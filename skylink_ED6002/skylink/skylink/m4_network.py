"""M4 - hub location, PHC assignment, fleet and bays: MILP (HiGHS via scipy.optimize.milp).

Variables: y_j (open hub, binary), x_ij (PHC i served by hub j, binary), z_i (PHC i road-only,
binary), n_j drones, b_j bays, o_j operators per shift (integers).
"""
from __future__ import annotations
import os, json, math, itertools
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from . import data
from .m1_vehicle import Vehicle

UTIL = 0.70
EMERG_FLIGHT_BUDGET_MIN = 45 - 3 - 1 - 6     # 45 min target - load/take-off - drop - 6 min queue allowance


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
    S = data.sites(); hubs = list(S.index[S.candidate_hub]); phcs = list(data.demand().index)
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
    # big-M: the most drones / bays a hub could ever need = all assignable demand at its busiest hour
    Nmax = np.ceil((load_min * feas[:, :, None]).sum(0).max(1) / (UTIL * 60)).astype(int) + 1
    Bmax = np.ceil((load_Wh * feas[:, :, None]).sum(0).max(1) / (UTIL * 1350)).astype(int) + 1
    Omax = np.ceil(Nmax / 4).astype(int)
    # variable layout
    iy = np.arange(nJ); ix = nJ + np.arange(nI * nJ).reshape(nI, nJ); iz = nJ + nI * nJ + np.arange(nI)
    off = nJ + nI * nJ + nI; i_n = off + np.arange(nJ); i_b = i_n + nJ; i_o = i_b + nJ; nv = i_o[-1] + 1
    cvec = np.zeros(nv)
    cvec[iy] = cm["hub"]; cvec[ix] = var; cvec[iz] = road_only
    cvec[i_n] = cm["drone"] + cm["spare"]; cvec[i_b] = cm["bay"]; cvec[i_o] = cm["oper"]
    rows, lo, hi = [], [], []
    A = lil_matrix((nI + nI * nJ + nJ * 24 * 2 + nJ * 6 + 1, nv)); k = 0
    for i in range(nI):                               # assignment: each PHC to one hub or road
        A[k, ix[i]] = 1; A[k, iz[i]] = 1; lo.append(1); hi.append(1); k += 1
    for i in range(nI):                               # x_ij <= y_j  (disaggregated: no big-M, tight LP)
        for j in range(nJ):
            A[k, ix[i, j]] = 1; A[k, iy[j]] = -1; lo.append(-np.inf); hi.append(0); k += 1
    for j in range(nJ):
        for h in range(24):                           # drone time at <= 70% utilisation, every hour
            for i in range(nI): A[k, ix[i, j]] = load_min[i, j, h]
            A[k, i_n[j]] = -UTIL * 60; lo.append(-np.inf); hi.append(0); k += 1
        for h in range(24):                           # bay charging at <= 70% utilisation
            for i in range(nI): A[k, ix[i, j]] = load_Wh[i, j, h]
            A[k, i_b[j]] = -UTIL * 1350; lo.append(-np.inf); hi.append(0); k += 1
        # outage 14-17: 3 kW genset (2.7 kW into packs) for 3 h + one charged spare per drone
        for i in range(nI): A[k, ix[i, j]] = load_Wh[i, j, 14:17].sum()
        A[k, i_n[j]] = -vp["Euse_Wh"]; lo.append(-np.inf); hi.append(3 * 2700); k += 1
        A[k, i_n[j]] = 1; A[k, iy[j]] = -Nmax[j]; lo.append(-np.inf); hi.append(0); k += 1   # big-M (justified above)
        A[k, i_b[j]] = 1; A[k, iy[j]] = -Bmax[j]; lo.append(-np.inf); hi.append(0); k += 1
        A[k, i_o[j]] = 4; A[k, i_n[j]] = -1; lo.append(0); hi.append(np.inf); k += 1          # 1 operator per 4 drones
        A[k, i_o[j]] = 1; A[k, iy[j]] = -1; lo.append(0); hi.append(np.inf); k += 1           # >= 1 per open hub
        A[k, i_n[j]] = 1; A[k, iy[j]] = -1; lo.append(0); hi.append(np.inf); k += 1           # open hub has >= 1 drone
    A = A[:k].tocsr()
    ub = np.ones(nv); ub[i_n] = Nmax; ub[i_b] = Bmax; ub[i_o] = Omax
    ub_x = ub[ix]; ub_x[~feas] = 0; ub[ix] = ub_x                       # range/time-infeasible pairs fixed to 0
    lb = np.zeros(nv)
    if force_open:
        for h in force_open: lb[iy[[hubs[jj] for jj in J].index(h)]] = 1
    return dict(c=cvec, A=A, lo=np.array(lo), hi=np.array(hi), lb=lb, ub=ub, nv=nv, iy=iy, ix=ix, iz=iz, i_n=i_n, i_b=i_b, i_o=i_o,
                hubs=[hubs[j] for j in J], phcs=[phcs[i] for i in I], load_min=load_min, load_Wh=load_Wh, var=var, road_only=road_only,
                cm=cm, Nmax=Nmax, Bmax=Bmax, feas=feas, pdx=pdx, Euse=vp["Euse_Wh"])


def solve(M, relax=False, time_limit=120):
    integ = np.zeros(M["nv"]) if relax else np.ones(M["nv"])
    r = milp(M["c"], constraints=LinearConstraint(M["A"], M["lo"], M["hi"]), integrality=integ,
             bounds=Bounds(M["lb"], M["ub"]), options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    return r


def decode(M, r):
    x = r.x; out = dict(cost=r.fun, hubs={})
    for j, h in enumerate(M["hubs"]):
        if x[M["iy"][j]] > 0.5:
            served = [p for i, p in enumerate(M["phcs"]) if x[M["ix"][i, j]] > 0.5]
            out["hubs"][h] = dict(phcs=served, drones=int(round(x[M["i_n"][j]])), bays=int(round(x[M["i_b"][j]])),
                                  operators_per_shift=int(round(x[M["i_o"][j]])))
    out["road_only"] = [p for i, p in enumerate(M["phcs"]) if x[M["iz"][i]] > 0.5]
    return out


def cost_breakdown(M, r):
    x = r.x; cm = M["cm"]
    return dict(hubs=cm["hub"] * x[M["iy"]].sum(), drones_and_spares=(cm["drone"] + cm["spare"]) * x[M["i_n"]].sum(),
                bays=cm["bay"] * x[M["i_b"]].sum(), operators=cm["oper"] * x[M["i_o"]].sum(),
                energy_wear_and_weather_road=float((M["var"] * x[M["ix"]]).sum()), road_only_PHCs=float((M["road_only"] * x[M["iz"]]).sum()))


# ------------------------------------------------------------------ brute force verification
def brute_force(M):
    """Enumerate every assignment of the sub-instance's PHCs to {hub_1..hub_J, road}; for each, the
    cheapest integers (n, b, o) per hub follow in closed form. Must equal the MILP optimum."""
    nI, nJ = len(M["phcs"]), len(M["hubs"]); cm = M["cm"]
    choices = np.array(list(itertools.product(range(nJ + 1), repeat=nI)))      # (nJ+1)^nI x nI
    feas_ext = np.c_[M["feas"], np.ones(nI, bool)]
    ok = feas_ext[np.arange(nI)[None, :], choices].all(1)
    choices = choices[ok]
    total = np.zeros(len(choices))
    for i in range(nI):
        is_road = choices[:, i] == nJ
        total += np.where(is_road, M["road_only"][i], M["var"][i, np.minimum(choices[:, i], nJ - 1)])
    for j in range(nJ):
        mask = (choices == j).astype(float)               # N x nI
        used = mask.sum(1) > 0
        lm = mask @ M["load_min"][:, j, :]                # N x 24
        lw = mask @ M["load_Wh"][:, j, :]
        n = np.ceil(np.round(lm.max(1) / (UTIL * 60), 9))
        n = np.maximum(n, np.ceil(np.round((lw[:, 14:17].sum(1) - 3 * 2700) / M["Euse"], 9)))
        n = np.maximum(n, 1)
        b = np.ceil(np.round(lw.max(1) / (UTIL * 1350), 9))
        o = np.maximum(1, np.ceil(n / 4))
        total += used * (cm["hub"] + (cm["drone"] + cm["spare"]) * n + cm["bay"] * b + cm["oper"] * o)
    k = int(np.argmin(total))
    return float(total[k]), choices[k], len(choices)


# ------------------------------------------------------------------ driver
def run(designs: pd.DataFrame, m_plate, fm, tag="m4", make_figs=True):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ws = wind_stats(); rows, sols = [], {}
    for name, row in designs.iterrows():
        vp = vehicle_params(row, m_plate, fm)
        M = build(vp, ws)
        r = solve(M); rl = solve(M, relax=True)
        sol = decode(M, r); sol["breakdown"] = cost_breakdown(M, r); sol["vp"] = {k: v for k, v in vp.items() if k != "x"}; sol["x"] = list(map(float, vp["x"]))
        sols[name] = sol
        rows.append(dict(design=name, MTOW_kg=vp["MTOW"], R_still_km=vp["R_km"], Lmax_design_wind_km=M["pdx"]["Lmax_km"],
                         annual_cost_MINR=r.fun / 1e6, LP_relax_MINR=rl.fun / 1e6, LP_gap_pct=100 * (r.fun - rl.fun) / r.fun,
                         n_hubs=len(sol["hubs"]), hubs=" ".join(sol["hubs"]), drones=sum(h["drones"] for h in sol["hubs"].values()),
                         bays=sum(h["bays"] for h in sol["hubs"].values()), road_only_PHCs=len(sol["road_only"]),
                         feasible_pairs=int(M["feas"].sum()), mip_status=r.message))
    df = pd.DataFrame(rows).set_index("design")
    # stakeholder variant: blood bank (H0) forced open as a hub, for the cheapest vehicle
    best = df.annual_cost_MINR.idxmin(); vpb = vehicle_params(designs.loc[best], m_plate, fm)
    Mh = build(vpb, ws, force_open=["H0"]); rh = solve(Mh); solh = decode(Mh, rh); solh["breakdown"] = cost_breakdown(Mh, rh)
    sols[best + "_H0forced"] = solh
    df.loc[best, "cost_if_H0_forced_MINR"] = rh.fun / 1e6
    df.loc[best, "hubs_if_H0_forced"] = " ".join(solh["hubs"])
    # verification on a 4-hub, 8-PHC sub-instance (middle design)
    mid = designs.index[len(designs) // 2]
    vp = vehicle_params(designs.loc[mid], m_plate, fm)
    sub_h = ["H0", "H3", "H4", "P09"]; sub_p = ["P05", "P06", "P07", "P08", "P10", "P20", "P21", "P23"]
    Ms = build(vp, ws, sub_h, sub_p); rs = solve(Ms); bf, bch, ncomb = brute_force(Ms)
    rsl = solve(Ms, relax=True)
    ver = dict(design=mid, sub_hubs=sub_h, sub_phcs=sub_p, milp=rs.fun, brute_force=bf, n_enumerated=ncomb,
               rel_diff=abs(rs.fun - bf) / bf, milp_assign=decode(Ms, rs), bf_assign={p: (sub_h + ["road"])[c] for p, c in zip(sub_p, bch)},
               sub_LP_gap_pct=100 * (rs.fun - rsl.fun) / rs.fun, wind=ws,
               bigM_note="N_max_j = ceil(peak-hour drone-minutes of ALL range-feasible PHCs / (0.7*60)) + 1: no feasible solution can need more, "
                         "so the constraint n_j <= N_max_j y_j never cuts off a feasible point, and it is the smallest such M (tightest LP).")
    json.dump(ver, open(os.path.join(data.OUT, "tables", f"{tag}_verification.json"), "w"), indent=1, default=str)
    df.to_csv(os.path.join(data.OUT, "tables", f"{tag}_cost_vs_vehicle.csv"))
    json.dump(sols, open(os.path.join(data.OUT, "tables", f"{tag}_solutions.json"), "w"), indent=1, default=float)
    if make_figs:
        best = df.annual_cost_MINR.idxmin()
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].plot(df.MTOW_kg, df.annual_cost_MINR, "o-", label="MILP optimum"); ax[0].plot(df.MTOW_kg, df.LP_relax_MINR, "x--", label="LP relaxation")
        for n, rr in df.iterrows(): ax[0].annotate(f"{n}: {rr.n_hubs} hubs, {rr.drones} drones", (rr.MTOW_kg, rr.annual_cost_MINR), fontsize=8, xytext=(3, 5), textcoords="offset points")
        ax[0].set_xlabel("vehicle MTOW (kg)"); ax[0].set_ylabel("annualised cost (M INR / yr)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Annual cost vs vehicle choice")
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
    for sid, r in S.iterrows():
        if sid.startswith("P"):
            ax.plot(r.x_km, r.y_km, "o", color="k" if sid not in sol.get("road_only", []) else "red", ms=4); ax.annotate(sid, (r.x_km, r.y_km), fontsize=6, xytext=(2, -7), textcoords="offset points")
        elif sid not in sol["hubs"]:
            ax.plot(r.x_km, r.y_km, "^", color="grey", ms=6); ax.annotate(sid, (r.x_km, r.y_km), fontsize=6, color="grey")
    ax.set_xlim(0, 60); ax.set_ylim(0, 45); ax.set_aspect("equal"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_title(title, fontsize=10); ax.grid(alpha=.2)


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(data.OUT, "tables", "m1_tmp2_designs.csv"), index_col=0)
    from . import m3_surrogate as m3
    k, s, t = m3.fit_all(make_figs=False)
    df, sols, ver = run(d, 0.1126, m3.fm_function(k))
    pd.set_option("display.width", 250); print(df.round(3).to_string())
    print({k: v for k, v in ver.items() if k in ("milp", "brute_force", "rel_diff", "n_enumerated", "sub_LP_gap_pct", "wind")})
    for n, s in sols.items(): print(n, {h: (v["drones"], v["bays"], len(v["phcs"])) for h, v in s["hubs"].items()}, s["breakdown"])
