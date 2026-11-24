"""M4 - hub location, PHC assignment, fleet and bays: MILP (HiGHS via scipy.optimize.milp).

Variables: y_j (open hub, binary), x_ij (PHC i served by hub j, binary), z_i (PHC i road-only,
binary), n_j drones, b_j bays, o_j operators per shift (integers).
"""
from __future__ import annotations
import sys, os
if __package__ in (None, ""):          # allow running this file directly (PyCharm Run button)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "skylink"
import os, json, math, itertools
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from . import data
from .m1_vehicle import Vehicle

UTIL = 0.70
EMERG_FLIGHT_BUDGET_MIN = 45 - 3 - 1 - 6     # 45 min target - load/take-off - drop - 6 min queue allowance
# Who can send what (stakeholder rule):
#   blood    -> stocked only at the district blood bank H0.
#               BLOOD_MODE = "relay"  : H0 restocks every open sub-district hospital hub (RESUPPLY_PER_DAY trips/day,
#                                       by drone if the H0->hub path is within range, else by road); each hospital hub
#                                       then serves blood to its PHCs like any other emergency.
#               BLOOD_MODE = "H0_only": blood is flown to PHCs only from H0 (PHCs out of H0's reach get blood by road).
#   other emergency supplies (antivenom ...), routine, lab samples -> any open hospital hub H0-H6
#   P09 (a PHC) -> holds no emergency stock; it is a hub only if P09_HUB = True
BLOOD_HUB = "H0"
BLOOD_MODE = "relay"
RESUPPLY_PER_DAY = 1  # ASSUMPTION: blood restock trips per day from H0 to each open sub-district hospital hub
BLOOD_FRAC = 0.5      # ASSUMPTION (H0_only mode): share of emergency orders that are blood
P09_HUB = False       # True = P09 may be a hub for antivenom/routine/lab (brief lists it as a candidate site)
RELAY_P09 = True      # P09 may act as a battery-swap relay for blood restock flights (no stock, no hub role)
RELAY_COST_INR = 100000 + 0   # ASSUMPTION: P09 relay upkeep per year (site + swap by PHC staff); bay and packs added below
RESTOCK_HOUR = 6      # restock flights leave H0 at 06:00 (and every 24/RESUPPLY_PER_DAY h after)


def wind_stats():
    w = data.wind()
    fly = w[~w.grounded]
    return dict(w_design=float(fly.mean_wind_ms.quantile(0.90)), w_mean=float(fly.mean_wind_ms.mean()),
                grounded_frac=float(w.grounded.mean()))


def vehicle_params(row, m_plate, fm, eb=150.0):
    """performance of a (snapped) design: still-air radius, speed, powers, usable energy"""
    x = np.array(json.loads(row.snap_x)) if isinstance(row.snap_x, str) else np.array([row.D_m, row.E_Wh, row.d_mm/1e3, row.t_mm/1e3, row.V_ms])
    v = Vehicle(m_plate=m_plate, fm=fm, eb=eb); o = v.evaluate(x)
    return dict(x=x, R_km=float(o["R"]), V=float(x[4]), MTOW=float(o["m"]), E_Wh=float(x[1]), Euse_Wh=float(o["Euse"])/3600,
                Pcl=float(o["Pcl"]), Pce=float(o["Pce"]), Phl=float(o["Phl"]), Phe=float(o["Phe"]))


def pair_data(vp, ws):
    """for every hub j and PHC i: feasibility, busy time per sortie (min), energy per sortie (Wh)"""
    D, _ = data.distance_matrix()
    S = data.sites(); hubs = [h for h in S.index[S.candidate_hub] if h != "P09" or P09_HUB]; phcs = list(data.demand().index)
    V, wd, wm = vp["V"], ws["w_design"], ws["w_mean"]
    L = D.loc[phcs, hubs].values * 1000                      # m, obstacle-avoiding one-way path
    # range with headwind at design wind: out-and-back time 2LV/(V^2-w^2) must fit still-air time 2R/V
    Lmax_range = vp["R_km"] * 1000 * (V ** 2 - wd ** 2) / V ** 2
    t_out_head = L / (V - wd) / 60                           # min, emergency leg flown into the design headwind
    feas = (L <= Lmax_range) & (t_out_head <= EMERG_FLIGHT_BUDGET_MIN)
    tf = 2 * L * V / (V ** 2 - wm ** 2)                      # s, at mean flyable wind
    busy = 3 + 1 + 6 + tf / 60                               # min
    e_Wh = ((vp["Pcl"] + vp["Pce"]) / 2 * tf + 60 * (vp["Phl"] + vp["Phe"])) / 3600
    same = np.array([[p == h for h in hubs] for p in phcs])  # PHC co-located with hub (P09): hand-carried
    busy[same] = 0; e_Wh[same] = 0; feas[same] = True
    return dict(L_km=L / 1000, feas=feas, busy=busy, e_Wh=e_Wh, hubs=hubs, phcs=phcs, Lmax_km=Lmax_range / 1000)


def costs_model(vp, ws, demand_mult=1.0, drone_price_mult=1.0):
    c = data.costs(); r = c["discount_rate"]
    dem = data.demand() * demand_mult; hp = data.hourly()
    lam = {k: dem[k].values for k in ("em", "ro", "lab")}           # per day
    if BLOOD_MODE == "H0_only":
        lam["blood"] = BLOOD_FRAC * lam["em"]; lam["em"] = (1 - BLOOD_FRAC) * lam["em"]   # em = antivenom; blood separate
    else:                                                                                  # relay: hubs hold restocked blood
        lam["blood"] = np.zeros_like(lam["em"])
    hub = c["hub"]["capex_inr"] * data.crf(r, 10) + c["hub"]["opex_inr_per_yr"]
    dr = c["drone"]; drone = drone_price_mult * (dr["base_capex_inr"] + dr["capex_per_kg_mtow_inr"] * vp["MTOW"]) * (data.crf(r, dr["life_yr"]) + dr["maintenance_frac_of_capex_per_yr"])
    bay = c["charging_bay"]["capex_inr"] * data.crf(r, c["charging_bay"]["life_yr"])
    st = c["staff"]; oper = st["cost_inr_per_person_yr"] * st["shifts_per_day"] * st["relief_factor"]
    spare = c["battery"]["cost_inr_per_wh"] * vp["E_Wh"] * data.crf(r, 3)   # 1 spare pack per drone, 3-yr calendar life (wear priced separately per Wh)
    # share of energy drawn in outage hours (14-17 Mon/Wed/Fri) -> genset price
    out_share = (3 / 7) * hp.loc[14:16].values.sum() / hp.values.sum()
    e_price = (c["energy"]["grid_inr_per_kwh"] * (1 - out_share) + c["energy"]["genset_inr_per_kwh"] * out_share) / 1000 / c["charging_bay"]["charge_efficiency"]
    wear = c["battery"]["cost_inr_per_wh"] / (c["battery"]["cycle_life"] * 0.8)   # INR per Wh throughput (equivalent full cycles)
    g = ws["grounded_frac"]
    road_e, road_rl = c["road_fallback"]["emergency_inr_per_trip"], c["road_fallback"]["routine_lab_inr_per_trip"]
    return dict(hub=hub, drone=drone, bay=bay, oper=oper, spare=spare, e_price=e_price, wear=wear, g=g, lam=lam,
                road_e=road_e, road_rl=road_rl, hp=hp, out_share=out_share)


def build(vp, ws, hub_subset=None, phc_subset=None, demand_mult=1.0, drone_price_mult=1.0, force_open=None):
    pdx = pair_data(vp, ws); cm = costs_model(vp, ws, demand_mult, drone_price_mult)
    hubs, phcs = pdx["hubs"], pdx["phcs"]
    J = [hubs.index(h) for h in (hub_subset or hubs)]; I = [phcs.index(p) for p in (phc_subset or phcs)]
    nJ, nI = len(J), len(I)
    lam = {k: v[I] for k, v in cm["lam"].items()}; hp = cm["hp"].values         # 24 x 3
    busy = pdx["busy"][np.ix_(I, J)]; e = pdx["e_Wh"][np.ix_(I, J)]; feas = pdx["feas"][np.ix_(I, J)]
    g = cm["g"]
    # hourly load (minutes of drone time and Wh) a PHC brings to a hub, per flyable hour
    rate_h = (lam["em"][:, None] * hp[None, :, 0] + lam["ro"][:, None] * hp[None, :, 1] + lam["lab"][:, None] * hp[None, :, 2])   # nI x 24
    load_min = rate_h[:, None, :] * busy[:, :, None] / (1 - g)        # nI x nJ x 24
    load_Wh = rate_h[:, None, :] * e[:, :, None] / (1 - g)
    daily = lam["em"] + lam["ro"] + lam["lab"]
    # annual variable cost of assigning i to j: energy + battery wear + road for grounded hours
    var = 365 * daily[:, None] * (1 - g) * e * (cm["e_price"] + cm["wear"])
    var += 365 * g * (lam["em"] * cm["road_e"] + 0.5 * (lam["ro"] + lam["lab"]) * cm["road_rl"])[:, None]
    road_only = 365 * (lam["em"] * cm["road_e"] + (lam["ro"] + lam["lab"]) * cm["road_rl"])
    # blood: always from the blood bank H0 (if H0 is in this instance), else by road
    hsub = [hubs[j] for j in J]
    jb = hsub.index(BLOOD_HUB) if BLOOD_HUB in hsub else None
    if jb is not None:
        feas_b = feas[:, jb].copy()
        rate_b = lam["blood"][:, None] * hp[None, :, 0]                                # nI x 24
        load_b_min = rate_b * busy[:, jb][:, None] / (1 - g); load_b_Wh = rate_b * e[:, jb][:, None] / (1 - g)
        var_b = 365 * lam["blood"] * ((1 - g) * e[:, jb] * (cm["e_price"] + cm["wear"]) + g * cm["road_e"])
    else:
        feas_b = np.zeros(nI, bool); load_b_min = np.zeros((nI, 24)); load_b_Wh = np.zeros((nI, 24)); var_b = np.zeros(nI)
    road_b = 365 * lam["blood"] * cm["road_e"]                                           # blood by road for the whole year
    # big-M: the most drones / bays a hub could ever need = all assignable demand at its busiest hour
    extra_min = np.zeros((nJ, 24)); extra_Wh = np.zeros((nJ, 24))
    if jb is not None:
        extra_min[jb] = (load_b_min * feas_b[:, None]).sum(0); extra_Wh[jb] = (load_b_Wh * feas_b[:, None]).sum(0)
    Nmax = np.ceil(((load_min * feas[:, :, None]).sum(0) + extra_min).max(1) / (UTIL * 60)).astype(int) + 1
    Bmax = np.ceil(((load_Wh * feas[:, :, None]).sum(0) + extra_Wh).max(1) / (UTIL * 1350)).astype(int) + 1
    Omax = np.ceil(Nmax / 4).astype(int)
    # variable layout: y_j | x_ij | z_i | n_j | b_j | o_j | u_i (blood flown from H0)
    iy = np.arange(nJ); ix = nJ + np.arange(nI * nJ).reshape(nI, nJ); iz = nJ + nI * nJ + np.arange(nI)
    off = nJ + nI * nJ + nI; i_n = off + np.arange(nJ); i_b = i_n + nJ; i_o = i_b + nJ; iu = i_o[-1] + 1 + np.arange(nI); nv = iu[-1] + 1
    cvec = np.zeros(nv)
    cvec[iy] = cm["hub"]; cvec[ix] = var; cvec[iz] = road_only
    resup = np.zeros(nJ); resup_mode = {}
    if BLOOD_MODE == "relay":                       # daily blood restock run H0 -> each other open hospital hub
        Dm, _ = data.distance_matrix()
        V, wd, wm = vp["V"], ws["w_design"], ws["w_mean"]
        for j, h in enumerate(hsub):
            if h == BLOOD_HUB or not h.startswith("H"): continue
            Lr = Dm.loc[BLOOD_HUB, h]
            if Lr <= pdx["Lmax_km"]:
                tf = 2 * Lr * 1000 * V / (V ** 2 - wm ** 2)
                er = ((vp["Pcl"] + vp["Pce"]) / 2 * tf + 60 * (vp["Phl"] + vp["Phe"])) / 3600
                resup[j] = 365 * RESUPPLY_PER_DAY * er * (cm["e_price"] + cm["wear"]); resup_mode[h] = "drone"
            else:
                resup[j] = 365 * RESUPPLY_PER_DAY * cm["road_rl"]; resup_mode[h] = "road"
        cvec[iy] += resup
    cvec[i_n] = cm["drone"] + cm["spare"]; cvec[i_b] = cm["bay"]; cvec[i_o] = cm["oper"]
    cvec[iu] = var_b - road_b              # flying blood replaces its road cost; the constant road_b.sum() is added in solve()
    const = float(road_b.sum())
    rows, lo, hi = [], [], []
    A = lil_matrix((nI + nI * nJ + nI + nJ * 24 * 2 + nJ * 6 + 1, nv)); k = 0
    for i in range(nI):                               # assignment: each PHC to one hub or road (antivenom, routine, lab)
        A[k, ix[i]] = 1; A[k, iz[i]] = 1; lo.append(1); hi.append(1); k += 1
    for i in range(nI):                               # x_ij <= y_j  (disaggregated: no big-M, tight LP)
        for j in range(nJ):
            A[k, ix[i, j]] = 1; A[k, iy[j]] = -1; lo.append(-np.inf); hi.append(0); k += 1
    if jb is not None:
        for i in range(nI):                           # blood by drone only if the blood bank hub is open
            A[k, iu[i]] = 1; A[k, iy[jb]] = -1; lo.append(-np.inf); hi.append(0); k += 1
    for j in range(nJ):
        for h in range(24):                           # drone time at <= 70% utilisation, every hour
            for i in range(nI): A[k, ix[i, j]] = load_min[i, j, h]
            if j == jb:
                for i in range(nI): A[k, iu[i]] = load_b_min[i, h]
            A[k, i_n[j]] = -UTIL * 60; lo.append(-np.inf); hi.append(0); k += 1
        for h in range(24):                           # bay charging at <= 70% utilisation
            for i in range(nI): A[k, ix[i, j]] = load_Wh[i, j, h]
            if j == jb:
                for i in range(nI): A[k, iu[i]] = load_b_Wh[i, h]
            A[k, i_b[j]] = -UTIL * 1350; lo.append(-np.inf); hi.append(0); k += 1
        # outage 14-17: 3 kW genset (2.7 kW into packs) for 3 h + one charged spare per drone
        for i in range(nI): A[k, ix[i, j]] = load_Wh[i, j, 14:17].sum()
        if j == jb:
            for i in range(nI): A[k, iu[i]] = load_b_Wh[i, 14:17].sum()
        A[k, i_n[j]] = -vp["Euse_Wh"]; lo.append(-np.inf); hi.append(3 * 2700); k += 1
        A[k, i_n[j]] = 1; A[k, iy[j]] = -Nmax[j]; lo.append(-np.inf); hi.append(0); k += 1   # big-M (justified above)
        A[k, i_b[j]] = 1; A[k, iy[j]] = -Bmax[j]; lo.append(-np.inf); hi.append(0); k += 1
        A[k, i_o[j]] = 4; A[k, i_n[j]] = -1; lo.append(0); hi.append(np.inf); k += 1          # 1 operator per 4 drones
        A[k, i_o[j]] = 1; A[k, iy[j]] = -1; lo.append(0); hi.append(np.inf); k += 1           # >= 1 per open hub
        A[k, i_n[j]] = 1; A[k, iy[j]] = -1; lo.append(0); hi.append(np.inf); k += 1           # open hub has >= 1 drone
    A = A[:k].tocsr()
    ub = np.ones(nv); ub[i_n] = Nmax; ub[i_b] = Bmax; ub[i_o] = Omax
    ub_x = ub[ix]; ub_x[~feas] = 0; ub[ix] = ub_x                       # range/time-infeasible pairs fixed to 0
    ub[iu] = feas_b.astype(float)                                        # blood: only PHCs within reach of H0
    lb = np.zeros(nv)
    if force_open:
        for h in force_open: lb[iy[[hubs[jj] for jj in J].index(h)]] = 1
    return dict(c=cvec, A=A, lo=np.array(lo), hi=np.array(hi), lb=lb, ub=ub, nv=nv, iy=iy, ix=ix, iz=iz, i_n=i_n, i_b=i_b, i_o=i_o, iu=iu,
                hubs=hsub, phcs=[phcs[i] for i in I], load_min=load_min, load_Wh=load_Wh, var=var, road_only=road_only,
                cm=cm, Nmax=Nmax, Bmax=Bmax, feas=feas, pdx=pdx, Euse=vp["Euse_Wh"], const=const, jb=jb, feas_b=feas_b,
                load_b_min=load_b_min, load_b_Wh=load_b_Wh, var_b=var_b, road_b=road_b, resup=resup, resup_mode=resup_mode)


def solve(M, relax=False, time_limit=120):
    integ = np.zeros(M["nv"]) if relax else np.ones(M["nv"])
    r = milp(M["c"], constraints=LinearConstraint(M["A"], M["lo"], M["hi"]), integrality=integ,
             bounds=Bounds(M["lb"], M["ub"]), options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    if r.x is not None:
        r.fun = float(r.fun + M.get("const", 0.0))       # add the all-blood-by-road constant back
    return r


def decode(M, r):
    x = r.x; out = dict(cost=r.fun, hubs={})
    for j, h in enumerate(M["hubs"]):
        if x[M["iy"][j]] > 0.5:
            served = [p for i, p in enumerate(M["phcs"]) if x[M["ix"][i, j]] > 0.5]
            out["hubs"][h] = dict(phcs=served, drones=int(round(x[M["i_n"][j]])), bays=int(round(x[M["i_b"][j]])),
                                  operators_per_shift=int(round(x[M["i_o"][j]])))
    out["road_only"] = [p for i, p in enumerate(M["phcs"]) if x[M["iz"][i]] > 0.5]
    out["blood_mode"] = BLOOD_MODE; out["blood_hub"] = BLOOD_HUB
    if BLOOD_MODE == "H0_only":
        out["blood_from_H0"] = [p for i, p in enumerate(M["phcs"]) if x[M["iu"][i]] > 0.5]
        out["blood_by_road"] = [p for p in M["phcs"] if p not in out["blood_from_H0"]]
        out["blood_frac"] = BLOOD_FRAC
    else:   # relay: blood is served by each PHC's own hospital hub (restocked from H0); simulated with the emergencies
        out["blood_from_H0"] = []; out["blood_by_road"] = list(out["road_only"]); out["blood_frac"] = 0.0
        out["resupply"] = {h: M["resup_mode"][h] for h in out["hubs"] if h in M["resup_mode"]}
    return out


def cost_breakdown(M, r):
    x = r.x; cm = M["cm"]
    return dict(hubs=cm["hub"] * x[M["iy"]].sum(), drones_and_spares=(cm["drone"] + cm["spare"]) * x[M["i_n"]].sum(),
                bays=cm["bay"] * x[M["i_b"]].sum(), operators=cm["oper"] * x[M["i_o"]].sum(),
                energy_wear_and_weather_road=float((M["var"] * x[M["ix"]]).sum()), road_only_PHCs=float((M["road_only"] * x[M["iz"]]).sum()),
                blood_by_drone=float((M["var_b"] * x[M["iu"]]).sum()), blood_by_road=float((M["road_b"] * (1 - x[M["iu"]])).sum()),
                blood_resupply=float((M["resup"] * x[M["iy"]]).sum()))


# ------------------------------------------------------------------ blood restock routing (after the MILP)
def restock_routing(sol, vp, ws, M):
    """Route the daily blood restock run H0 -> each open sub-district hospital hub over a chain of drone hops.
    A hop is allowed between two nodes whose obstacle-avoiding path fits the drone's reach at the design wind;
    nodes are the OPEN hubs (they have charged batteries) plus P09 as a battery-swap relay (if RELAY_P09).
    Hubs with no drone chain are restocked by road. The MILP priced every restock run by road (an upper bound);
    this re-prices the runs that can fly. Returns the updated cost and writes the routes into sol."""
    if BLOOD_MODE != "relay":
        return sol["cost"]
    import networkx as nx
    Dm, _ = data.distance_matrix(); cm = M["cm"]
    V, wm = vp["V"], ws["w_mean"]; Lmax = M["pdx"]["Lmax_km"]
    open_h = list(sol["hubs"])
    # battery-swap relay candidates: P09 (if RELAY_P09) and any hospital H1-H6 that is NOT an open hub
    # (a hospital has staff to swap a pack). Each relay used costs one bay + 2 spare packs + RELAY_COST_INR per year.
    cand = [n for n in Dm.index if n.startswith("H") and n not in open_h] + (["P09"] if RELAY_P09 and "P09" not in open_h else [])
    def leg_Wh(L_km):
        tf = 2 * L_km * 1000 * V / (V ** 2 - wm ** 2)
        return ((vp["Pcl"] + vp["Pce"]) / 2 * tf + 60 * (vp["Phl"] + vp["Phe"])) / 3600
    relay_fix = cm["bay"] + 2 * cm["spare"] + RELAY_COST_INR
    targets = [h for h in open_h if h != BLOOD_HUB and h.startswith("H")]
    old = 365 * RESUPPLY_PER_DAY * cm["road_rl"] * len(targets)
    best = None
    from itertools import combinations
    for k in range(len(cand) + 1):                      # try every set of relays (few candidates -> exact)
        for rel in combinations(cand, k):
            nodes = open_h + list(rel)
            G = nx.Graph(); G.add_nodes_from(nodes)
            for i, a_ in enumerate(nodes):
                for b_ in nodes[i + 1:]:
                    if Dm.loc[a_, b_] <= Lmax: G.add_edge(a_, b_, weight=float(Dm.loc[a_, b_]))
            tot, rts, used = len(rel) * relay_fix, {}, set()
            for h in targets:
                road = 365 * RESUPPLY_PER_DAY * cm["road_rl"]
                route = None
                if BLOOD_HUB in open_h:
                    try: route = nx.dijkstra_path(G, BLOOD_HUB, h)
                    except nx.NetworkXNoPath: route = None
                if route:
                    legs = [Dm.loc[route[q], route[q + 1]] for q in range(len(route) - 1)]
                    dr = 365 * RESUPPLY_PER_DAY * sum(leg_Wh(L) for L in legs) * (cm["e_price"] + cm["wear"])
                if route and dr <= road:
                    tot += dr; rts[h] = dict(mode="drone", route=route, km=round(float(sum(legs)), 1))
                    used |= set(route[1:-1]) & set(rel)
                else:
                    tot += road; rts[h] = dict(mode="road", route=[BLOOD_HUB, h], km=round(float(Dm.loc[BLOOD_HUB, h]), 1))
            if used != set(rel): continue                # a relay that no route uses is never worth paying for
            if best is None or tot < best[0] - 1e-6: best = (tot, rts, list(rel))
    new, routes, relays = best
    relay_used = "P09" in relays
    sol["relays"] = relays
    sol["resupply"] = routes; sol["relay_P09_used"] = relay_used
    sol["breakdown"]["blood_resupply"] = new
    sol["cost_milp"] = sol["cost"]; sol["cost"] = sol["cost"] - old + new
    return sol["cost"]


# ------------------------------------------------------------------ brute force verification
def brute_force(M):
    """Enumerate every assignment of the sub-instance's PHCs to {hub_1..hub_J, road} for antivenom/routine/lab,
    AND every choice of which PHCs get blood flown from H0 (vs road). For each combination the cheapest
    integers (n, b, o) per hub follow in closed form. Must equal the MILP optimum."""
    nI, nJ = len(M["phcs"]), len(M["hubs"]); cm = M["cm"]; jb = M["jb"]
    choices = np.array(list(itertools.product(range(nJ + 1), repeat=nI)))      # (nJ+1)^nI x nI
    feas_ext = np.c_[M["feas"], np.ones(nI, bool)]
    choices = choices[feas_ext[np.arange(nI)[None, :], choices].all(1)]
    fb = np.where(M["feas_b"])[0]
    U = np.zeros((2 ** len(fb), nI))
    for k, bits in enumerate(itertools.product((0, 1), repeat=len(fb))):
        U[k, fb] = bits
    cx = np.zeros(len(choices))
    for i in range(nI):
        is_road = choices[:, i] == nJ
        cx += np.where(is_road, M["road_only"][i], M["var"][i, np.minimum(choices[:, i], nJ - 1)])
    cu = U @ (M["var_b"] - M["road_b"]) + M["road_b"].sum()
    def hub_cost(lm, lw, used):
        n = np.ceil(np.round(lm.max(-1) / (UTIL * 60), 9))
        n = np.maximum(n, np.ceil(np.round((lw[..., 14:17].sum(-1) - 3 * 2700) / M["Euse"], 9)))
        n = np.maximum(n, 1)
        b = np.ceil(np.round(lw.max(-1) / (UTIL * 1350), 9))
        o = np.maximum(1, np.ceil(n / 4))
        return used * (cm["hub"] + (cm["drone"] + cm["spare"]) * n + cm["bay"] * b + cm["oper"] * o)
    total = cx[:, None] + cu[None, :]
    for j in range(nJ):
        mask = (choices == j).astype(float)
        lm = mask @ M["load_min"][:, j, :]; lw = mask @ M["load_Wh"][:, j, :]; used = mask.sum(1) > 0
        total += (used * M["resup"][j])[:, None] if j != jb else 0
        if j == jb:      # the blood bank hub also carries the blood sorties
            lm = lm[:, None, :] + (U @ M["load_b_min"])[None, :, :]
            lw = lw[:, None, :] + (U @ M["load_b_Wh"])[None, :, :]
            used = used[:, None] | (U.sum(1) > 0)[None, :]
            total += hub_cost(lm, lw, used)
        else:
            total += hub_cost(lm, lw, used)[:, None]
    k = int(np.argmin(total)); a, u = np.unravel_index(k, total.shape)
    return float(total[a, u]), choices[a], int(total.size)


# ------------------------------------------------------------------ driver
def run(designs: pd.DataFrame, m_plate, fm, tag="m4", make_figs=True):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ws = wind_stats(); rows, sols = [], {}
    for name, row in designs.iterrows():
        vp = vehicle_params(row, m_plate, fm)
        M = build(vp, ws)
        r = solve(M); rl = solve(M, relax=True)
        sol = decode(M, r); sol["breakdown"] = cost_breakdown(M, r); sol["vp"] = {k: v for k, v in vp.items() if k != "x"}; sol["x"] = list(map(float, vp["x"]))
        restock_routing(sol, vp, ws, M)
        sols[name] = sol
        rows.append(dict(design=name, MTOW_kg=vp["MTOW"], R_still_km=vp["R_km"], Lmax_design_wind_km=M["pdx"]["Lmax_km"],
                         annual_cost_MINR=sol["cost"] / 1e6, MILP_cost_MINR=r.fun / 1e6, LP_relax_MINR=rl.fun / 1e6, LP_gap_pct=100 * (r.fun - rl.fun) / r.fun,
                         n_hubs=len(sol["hubs"]), hubs=" ".join(sol["hubs"]), drones=sum(h["drones"] for h in sol["hubs"].values()),
                         bays=sum(h["bays"] for h in sol["hubs"].values()), road_only_PHCs=len(sol["road_only"]),
                         feasible_pairs=int(M["feas"].sum()), mip_status=r.message))
    df = pd.DataFrame(rows).set_index("design")
    # stakeholder variant: blood bank (H0) forced open as a hub, for the cheapest vehicle
    best = df.annual_cost_MINR.idxmin(); vpb = vehicle_params(designs.loc[best], m_plate, fm)
    Mh = build(vpb, ws, force_open=["H0"]); rh = solve(Mh); solh = decode(Mh, rh); solh["breakdown"] = cost_breakdown(Mh, rh)
    solh["vp"] = {k: v for k, v in vpb.items() if k != "x"}; restock_routing(solh, vpb, ws, Mh)
    sols[best + "_H0forced"] = solh
    df.loc[best, "cost_if_H0_forced_MINR"] = solh["cost"] / 1e6
    df.loc[best, "hubs_if_H0_forced"] = " ".join(solh["hubs"])
    # verification on a 4-hub, 8-PHC sub-instance (middle design)
    mid = designs.index[len(designs) // 2]
    vp = vehicle_params(designs.loc[mid], m_plate, fm)
    sub_h = ["H0", "H3", "H4", "H6"]; sub_p = ["P06", "P07", "P08", "P10", "P19", "P20", "P21", "P24"]   # brief: 4 hubs, 8 PHCs
    Ms = build(vp, ws, sub_h, sub_p); rs = solve(Ms); bf, bch, ncomb = brute_force(Ms)
    rsl = solve(Ms, relax=True)
    ver = dict(design=mid, sub_hubs=sub_h, sub_phcs=sub_p, milp=rs.fun, brute_force=bf, n_enumerated=ncomb,
               rel_diff=abs(rs.fun - bf) / bf, milp_assign=decode(Ms, rs), bf_assign={p: (sub_h + ["road"])[c] for p, c in zip(sub_p, bch)},
               sub_LP_gap_pct=100 * (rs.fun - rsl.fun) / rs.fun, wind=ws,
               bigM_note="N_max_j = ceil(peak-hour drone-minutes of ALL range-feasible PHCs / (0.7*60)) + 1: no feasible solution can need more, "
                         "so the constraint n_j <= N_max_j y_j never cuts off a feasible point, and it is the smallest such M (tightest LP).")
    json.dump(ver, open(os.path.join(data.OUT, "tables", f"{tag}_verification.json"), "w", encoding="utf-8"), indent=1, default=str)
    df.to_csv(os.path.join(data.OUT, "tables", f"{tag}_cost_vs_vehicle.csv"))
    json.dump(sols, open(os.path.join(data.OUT, "tables", f"{tag}_solutions.json"), "w", encoding="utf-8"), indent=1, default=float)
    if make_figs:
        best = df.annual_cost_MINR.idxmin()
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        if len(df) > 1:      # several vehicle designs: compare their network costs
            ax[0].plot(df.MTOW_kg, df.annual_cost_MINR, "o-", label="MILP optimum"); ax[0].plot(df.MTOW_kg, df.LP_relax_MINR, "x--", label="LP relaxation")
            for n, rr in df.iterrows(): ax[0].annotate(f"{n}: {rr.n_hubs} hubs, {rr.drones} drones", (rr.MTOW_kg, rr.annual_cost_MINR), fontsize=8, xytext=(3, 5), textcoords="offset points")
            ax[0].set_xlabel("vehicle MTOW (kg)"); ax[0].set_ylabel("annualised cost (M INR / yr)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Annual cost vs vehicle choice")
        else:                # one design: where the money goes
            bd = sols[best]["breakdown"]; r0 = df.loc[best]
            labels = {"operators": "operators (3 shifts)", "drones_and_spares": "drones + spare packs", "hubs": "hubs (capex + opex)",
                      "energy_wear_and_weather_road": "energy, battery wear,\nroad in grounded hours", "road_only_PHCs": "road-only PHCs", "bays": "charging bays",
                      "blood_by_drone": "blood flown from H0", "blood_by_road": "blood by road\n(PHCs out of H0's reach)",
                      "blood_resupply": "blood restock runs\nH0 -> hospital hubs"}
            items = sorted(((labels.get(k, k), v / 1e5) for k, v in bd.items()), key=lambda t: t[1])
            yy = np.arange(len(items))
            ax[0].barh(yy, [v for _, v in items], color="tab:blue")
            ax[0].set_yticks(yy); ax[0].set_yticklabels([n for n, _ in items])
            for i, (_, v) in enumerate(items): ax[0].text(v, i, f"  {v:.1f}", va="center", fontsize=9)
            ax[0].set_xlabel("INR lakh / year"); ax[0].grid(alpha=.3, axis="x")
            ax[0].set_title(f"Where the money goes: total Rs {r0.annual_cost_MINR/10:.2f} crore/yr\n"
                            f"{r0.n_hubs} hubs, {r0.drones} drones, {r0.bays} bays, {r0.road_only_PHCs} road-only PHCs | "
                            f"MTOW {r0.MTOW_kg:.1f} kg, LP gap {r0.LP_gap_pct:.1f}%", fontsize=10)
        draw_map(ax[1], sols[best], title=f"Network for {best} (cost {df.annual_cost_MINR[best]:.2f} M INR/yr)")
        plt.tight_layout(); plt.savefig(os.path.join(data.OUT, "figs", f"{tag}_network.png"), dpi=120); plt.close()
    return df, sols, ver


def draw_map(ax, sol, title=""):
    from shapely.geometry import Polygon
    S = data.sites()
    for zn, poly in data.nofly().items():
        xs, ys = poly.exterior.xy; ax.fill(xs, ys, color="red", alpha=.15); ax.text(poly.centroid.x, poly.centroid.y, zn, fontsize=7, ha="center", color="darkred")
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    for k, (h, info) in enumerate(sol["hubs"].items()):
        for p in info["phcs"]:
            if p == h: continue
            pts = np.array(data.path(h, p)); ax.plot(pts[:, 0], pts[:, 1], color=cols[k % 8], lw=1)
        ax.plot(S.loc[h, "x_km"], S.loc[h, "y_km"], "s", ms=12, color=cols[k % 8], mec="k")
        ax.annotate(f"{h}\n{info['drones']}d/{info['bays']}b", (S.loc[h, "x_km"], S.loc[h, "y_km"]), fontsize=8, xytext=(5, 5), textcoords="offset points", weight="bold")
    for h, rr in sol.get("resupply", {}).items():    # daily blood restock H0 -> hospital hubs
        if rr["mode"] == "drone":
            pts = np.vstack([np.array(data.path(rr["route"][k], rr["route"][k + 1])) for k in range(len(rr["route"]) - 1)])
            ax.plot(pts[:, 0], pts[:, 1], color="crimson", lw=2.2, ls="--", alpha=.85, zorder=4)
        else:
            a_, b_ = S.loc[rr["route"][0]], S.loc[rr["route"][-1]]
            ax.plot([a_.x_km, b_.x_km], [a_.y_km, b_.y_km], color="crimson", lw=1.2, ls=":", alpha=.6, zorder=3)
    for rl in sorted({n for rr in sol.get("resupply", {}).values() for n in rr["route"][1:-1] if n not in sol["hubs"]}):
        ax.plot(S.loc[rl, "x_km"], S.loc[rl, "y_km"], "D", ms=9, color="gold", mec="k", zorder=5)
        ax.annotate(f"{rl} swap relay", (S.loc[rl, "x_km"], S.loc[rl, "y_km"]), fontsize=7, xytext=(6, -12), textcoords="offset points", color="darkgoldenrod")
    for p in sol.get("blood_from_H0", []):          # blood sorties from the blood bank
        pts = np.array(data.path(BLOOD_HUB, p)); ax.plot(pts[:, 0], pts[:, 1], color="crimson", lw=0.8, ls="--", alpha=.7)
    if BLOOD_HUB in S.index:
        ax.plot(S.loc[BLOOD_HUB, "x_km"], S.loc[BLOOD_HUB, "y_km"], "*", ms=14, color="crimson", mec="k", zorder=5)
    for sid, r in S.iterrows():
        if sid.startswith("P"):
            ax.plot(r.x_km, r.y_km, "o", color="k" if sid not in sol.get("road_only", []) else "red", ms=4); ax.annotate(sid, (r.x_km, r.y_km), fontsize=6, xytext=(2, -7), textcoords="offset points")
        elif sid not in sol["hubs"]:
            ax.plot(r.x_km, r.y_km, "^", color="grey", ms=6); ax.annotate(sid, (r.x_km, r.y_km), fontsize=6, color="grey")
    ax.set_xlim(0, 60); ax.set_ylim(0, 45); ax.set_aspect("equal"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_title(title, fontsize=10); ax.grid(alpha=.2)
    if sol.get("resupply"):
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="crimson", lw=2.2, ls="--", label="blood restock by drone (hops, battery swap)"),
                           Line2D([], [], color="crimson", lw=1.2, ls=":", label="blood restock by road"),
                           Line2D([], [], color="grey", lw=1, label="hub -> PHC drone deliveries (all classes)")],
                  loc="lower left", fontsize=7, framealpha=.9)


if __name__ == "__main__":
    # Standalone M4 on the CURRENT M1 design: R_REQ from m1_vehicle.py, plate mass from the last M2 run.
    # FM: the M3 surrogate if m3_surrogate.py and oracle data are present, else the constant 0.65.
    from . import m1_vehicle as m1
    p2 = os.path.join(data.OUT, "tables", "m2_summary.json")
    m_plate = json.load(open(p2, encoding="utf-8"))["picked_plate_mass_kg"] if os.path.exists(p2) else 0.25
    fm, fm_note = None, "FM = 0.65 (constant)"
    try:
        from . import m3_surrogate as m3
        k, s3, _ = m3.fit_all(make_figs=False); fm = m3.fm_function(k)
        fm_note = "FM = M3 surrogate" + (" [MOCK oracle]" if s3["mock"] else " [oracle data]")
    except Exception:
        pass
    _, d, _ = m1.run(m_plate=m_plate, fm=fm, R_req=m1.R_REQ, tag="m1", quick=True)
    df, sols, ver = run(d, m_plate, fm)
    name = df.index[0]; sol = sols[name]; vp = sol["vp"]; r = df.loc[name]
    ws = ver["wind"]
    print(f"\n=== M4 hub network for R_REQ = {m1.R_REQ:g} km  (plate {m_plate*1000:.0f} g, {fm_note}) ===")
    print(f"Drone: MTOW {vp['MTOW']:.2f} kg, still-air radius {vp['R_km']:.1f} km, cruise {vp['V']:.1f} m/s, battery {vp['E_Wh']:.0f} Wh")
    print(f"Design wind (P90 of flyable hours) {ws['w_design']:.1f} m/s -> longest reachable flight path {r.Lmax_design_wind_km:.1f} km;"
          f" grounded {100*ws['grounded_frac']:.1f}% of hours")
    print(f"Annual cost {r.annual_cost_MINR:.3f} M INR (= Rs {r.annual_cost_MINR/10:.2f} crore/yr); LP relaxation {r.LP_relax_MINR:.3f}; LP gap {r.LP_gap_pct:.1f}%")
    print(f"Hubs opened: {r.hubs}  |  drones {r.drones}, bays {r.bays}  |  road-only PHCs: {r.road_only_PHCs} of 24")
    for h, v in sol["hubs"].items():
        print(f"  {h}: {v['drones']} drones, {v['bays']} bays, {v['operators_per_shift']} operator/shift -> {', '.join(v['phcs'])}")
    if sol["road_only"]: print(f"  road only (antivenom/routine/lab): {', '.join(sol['road_only'])}")
    if BLOOD_MODE == "H0_only":
        print(f"Blood ({100*BLOOD_FRAC:.0f}% of emergencies, only from {BLOOD_HUB}): by drone to {len(sol['blood_from_H0'])} PHCs "
              f"[{', '.join(sol['blood_from_H0'])}]; by road to {len(sol['blood_by_road'])} PHCs")
    else:
        print(f"Blood: stocked at {BLOOD_HUB}; restocked {RESUPPLY_PER_DAY}x/day to each open hospital hub, then each hub serves its own PHCs")
        for h, rr in sol.get("resupply", {}).items():
            print(f"  restock {BLOOD_HUB} -> {h}: " + (f"drone via {' -> '.join(rr['route'])} ({rr['km']} km, battery swap at each stop)"
                                                     if rr["mode"] == "drone" else f"road ({rr['km']} km path; no drone hop chain within reach, or road is cheaper)"))
        if sol.get("relays"): print(f"  battery-swap relays used (no stock, not hubs): {', '.join(sol['relays'])}")
        print(f"  MILP priced all restock runs by road: {sol['cost_milp']/1e6:.3f} M INR -> after routing {sol['cost']/1e6:.3f} M INR")
    print(f"P09 as hub: {'allowed' if P09_HUB else 'not allowed (no emergency stock)'}")
    print("Cost breakdown (INR/yr):", {k: round(v) for k, v in sol["breakdown"].items()})
    if "cost_if_H0_forced_MINR" in df.columns:
        print(f"If the district hospital H0 (blood bank) must be a hub: {r.cost_if_H0_forced_MINR:.3f} M INR, hubs {r.hubs_if_H0_forced}")
    print(f"\nVerification - brute force on hubs {ver['sub_hubs']} x PHCs {ver['sub_phcs']}: MILP {ver['milp']/1e6:.4f} vs brute force "
          f"{ver['brute_force']/1e6:.4f} M INR ({ver['n_enumerated']} assignments) -> {'MATCH' if ver['rel_diff'] < 1e-9 else 'MISMATCH'}; sub-instance LP gap {ver['sub_LP_gap_pct']:.1f}%")
    print("Big-M:", ver["bigM_note"])
    print("Saved: outputs/figs/m4_network.png, outputs/tables/m4_cost_vs_vehicle.csv, m4_solutions.json, m4_verification.json")
