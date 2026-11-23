"""M6 - simulation-based multi-objective optimisation: NSGA-II with common random numbers (CRN).

Decision vector (all integer after rounding):
  k            vehicle design index (V1..V5), which also selects that vehicle's M4 network
  dn_j, db_j   change of drones / bays at hub slot j relative to the M4 optimum (j = 1..7)
  s_j          spare battery packs at hub slot j
  r            emergency reserve fraction (0..0.5, step 0.05)
Objectives: annual cost (min), S45 (max), road-fallback share (min).
CRN: every candidate is simulated on the SAME 36 days (3 per month) with the SAME seeds, so
differences between candidates are not masked by sampling noise.
"""
from __future__ import annotations
import os, json, math
import numpy as np, pandas as pd
from multiprocessing import Pool
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from . import data, m5_sim as m5
from .m4_network import costs_model, wind_stats

NSLOT = 7
LO = np.r_[0, [-2] * NSLOT, [-1] * NSLOT, [0] * NSLOT, 0]
HI = np.r_[4.999, [3.999] * NSLOT, [2.999] * NSLOT, [8.999] * NSLOT, 10.999]
_G = {}


def setup(sols, designs, n_days=36, seed=2026):
    _G["sols"] = {k: v for k, v in sols.items() if k in designs}
    _G["names"] = list(designs)
    _G["days"] = m5.sample_days(n_days, seed=seed)
    _G["seeds"] = list(range(7000, 7000 + n_days))
    _G["ws"] = wind_stats()


def decode(z):
    z = np.floor(np.asarray(z)).astype(int)
    name = _G["names"][min(z[0], len(_G["names"]) - 1)]
    sol = json.loads(json.dumps(_G["sols"][name]))
    spares = {}
    for j, h in enumerate(sol["hubs"]):
        if j >= NSLOT: break
        sol["hubs"][h]["drones"] = max(1, sol["hubs"][h]["drones"] + z[1 + j])
        sol["hubs"][h]["bays"] = max(1, sol["hubs"][h]["bays"] + z[1 + NSLOT + j])
        spares[h] = int(z[1 + 2 * NSLOT + j])
    r = 0.05 * z[-1]
    return name, sol, spares, r


def annual_cost(sol, vp, spares, ev, demand_mult=1.0, drone_price_mult=1.0):
    cm = costs_model(vp, _G.get("ws") or wind_stats(), demand_mult, drone_price_mult)
    c = data.costs()
    n = sum(h["drones"] for h in sol["hubs"].values()); b = sum(h["bays"] for h in sol["hubs"].values())
    ops = sum(max(1, math.ceil(h["drones"] / 4)) for h in sol["hubs"].values())
    spare_pack = c["battery"]["cost_inr_per_wh"] * vp["E_Wh"] * data.crf(c["discount_rate"], 3)
    fixed = cm["hub"] * len(sol["hubs"]) + cm["drone"] * n + spare_pack * sum(spares.values()) + cm["bay"] * b + cm["oper"] * ops
    e = ev["energy_kWh_day"] - ev["genset_kWh_day"]
    energy = 365 * (e * c["energy"]["grid_inr_per_kwh"] + ev["genset_kWh_day"] * c["energy"]["genset_inr_per_kwh"])
    wear = 365 * ev["energy_kWh_day"] * 1000 * 0.9 * cm["wear"]
    road = 365 * ev["road_cost_day"]
    return fixed + energy + wear + road


def eval_one(z, days=None, seeds=None, demand_mult=1.0, availability=1.0, wind_scale=1.0, drone_price_mult=1.0, weekday_shift=0):
    name, sol, spares, r = decode(z)
    vp = sol["vp"]
    cfg = m5.Config(sol, vp, spares=spares, reserve=r, demand_mult=demand_mult, availability=availability, wind_scale=wind_scale, weekday_shift=weekday_shift)
    ev = m5.evaluate(cfg, days or _G["days"], seeds or _G["seeds"])
    cost = annual_cost(sol, vp, spares, ev, demand_mult, drone_price_mult)
    return np.array([cost / 1e6, -ev["S45"], ev["road_share"]]), ev


def _f(z):
    return eval_one(z)[0]


class Prob(Problem):
    def __init__(self, pool, nobj=3):
        super().__init__(n_var=len(LO), n_obj=nobj, xl=LO, xu=HI); self.pool = pool; self.nobj = nobj
    def _evaluate(self, X, out, *a, **k):
        F = np.array(self.pool.map(_f, list(X)))
        out["F"] = F[:, :self.nobj]


class WS(Problem):
    """weighted sum of normalised cost and S45 (two-objective slice)"""
    def __init__(self, pool, w, ref):
        super().__init__(n_var=len(LO), n_obj=1, xl=LO, xu=HI); self.pool, self.w, self.ref = pool, w, ref
    def _evaluate(self, X, out, *a, **k):
        F = np.array(self.pool.map(_f, list(X)))
        (c0, c1), (s0, s1) = self.ref
        out["F"] = self.w * (F[:, 0] - c0) / (c1 - c0) + (1 - self.w) * (F[:, 1] - s0) / (s1 - s0)
        self.last = F


def seeds_from_m4():
    """include the plain M4 solutions (no changes, 1 spare per drone, r=0.2) in the initial population"""
    X = []
    for k in range(len(_G["names"])):
        sol = _G["sols"][_G["names"][k]]
        z = np.r_[k + 0.5, [0.5] * NSLOT, [0.5] * NSLOT, [min(8, h["drones"]) + 0.5 for h in list(sol["hubs"].values())[:NSLOT]] + [0.5] * (NSLOT - min(NSLOT, len(sol["hubs"]))), 4.5]
        X.append(z)
    return np.array(X)


def nondominated(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            if dom.any(): keep[i] = False
    return keep


def knee_index(F2):
    """2-D (cost, -S45): point with the largest distance below the line joining the extremes (normalised)"""
    c = (F2[:, 0] - F2[:, 0].min()) / (np.ptp(F2[:, 0]) + 1e-12); s = (F2[:, 1] - F2[:, 1].min()) / (np.ptp(F2[:, 1]) + 1e-12)
    a, b = np.argmin(c), np.argmin(s)
    p1, p2 = np.array([c[a], s[a]]), np.array([c[b], s[b]])
    v = p2 - p1; q = np.c_[c, s] - p1
    d = (v[0] * q[:, 1] - v[1] * q[:, 0]) / (np.linalg.norm(v) + 1e-12)
    return int(np.argmax(np.abs(d)))


def run(sols, designs, pop=60, gens=35, procs=2, tag="m6", seed=1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tabd = os.path.join(data.OUT, "tables"); figd = os.path.join(data.OUT, "figs")
    setup(sols, designs)
    m5.static()
    with Pool(procs, initializer=_init, initargs=(sols, list(designs))) as pool:
        init = np.vstack([seeds_from_m4(), FloatRandomSampling().do(Prob(pool), pop - len(designs)).get("X")])
        res = minimize(Prob(pool), NSGA2(pop_size=pop, sampling=init), ("n_gen", gens), seed=seed, verbose=False, save_history=False)
        X3, F3 = res.X, res.F
        np.savez(os.path.join(tabd, f"{tag}_raw3.npz"), X=X3, F=F3)
        res2 = minimize(Prob(pool, 2), NSGA2(pop_size=40, sampling=init[:40]), ("n_gen", 20), seed=seed + 1, verbose=False)
        X2, F2 = res2.X, res2.F
        np.savez(os.path.join(tabd, f"{tag}_raw2.npz"), X=X2, F=F2)
        FF = np.vstack([F2, F3[:, :2]])                       # normalise on the union (2-obj front may be narrow)
        ref = ((FF[:, 0].min(), FF[:, 0].max()), (FF[:, 1].min(), FF[:, 1].max()))
        ws = []
        for w in (0.15, 0.35, 0.5, 0.65, 0.85):
            p = WS(pool, w, ref)
            r = minimize(p, GA(pop_size=16, sampling=init[:16]), ("n_gen", 10), seed=seed + 7, verbose=False)
            ws.append(dict(w=w, cost=float(_f(r.X)[0]), negS45=float(_f(r.X)[1]), x=r.X.tolist()))
    # front table (3-obj)
    rows = []
    for x, f in zip(X3, F3):
        name, sol, spares, r = decode(x)
        rows.append(dict(vehicle=name, cost_MINR=f[0], S45=-f[1], road_share=f[2], reserve=r,
                         drones=sum(h["drones"] for h in sol["hubs"].values()), bays=sum(h["bays"] for h in sol["hubs"].values()),
                         spares=sum(spares.values()), hubs=len(sol["hubs"]), x=json.dumps(list(map(float, x)))))
    front = pd.DataFrame(rows).sort_values("cost_MINR").reset_index(drop=True)
    # knee on the cost-S45 projection of the 3-obj front
    nd2 = nondominated(front[["cost_MINR"]].assign(n=-front.S45).values)
    sub = front[nd2].reset_index()
    kn = int(sub.loc[knee_index(sub[["cost_MINR"]].assign(n=-sub.S45).values), "index"])
    front["knee"] = False; front.loc[kn, "knee"] = True
    # verification: weighted-sum points vs 2-obj NSGA-II front (must lie on or behind)
    f2 = F2[np.argsort(F2[:, 0])]
    def front_S_at(c):   # best S45 the NSGA-II front achieves at cost <= c
        ok = f2[f2[:, 0] <= c + 1e-9]
        return -ok[:, 1].min() if len(ok) else -np.inf
    for d in ws:
        d["S45"] = -d["negS45"]; d["front_best_S45_at_same_or_lower_cost"] = front_S_at(d["cost"])
        d["behind_or_on_front"] = bool(d["S45"] <= d["front_best_S45_at_same_or_lower_cost"] + 1e-9)
    # re-simulate front points with independent days (stated error check)
    ind_days = list(np.random.default_rng(99).integers(1, 366, 108)); ind_seeds = list(range(90000, 90108))
    chk = []
    pick = sorted(set([kn] + list(np.linspace(0, len(front) - 1, 6).astype(int))))
    for i in pick:
        f, ev = eval_one(json.loads(front.x[i]), ind_days, ind_seeds)
        chk.append(dict(idx=i, cost_crn=front.cost_MINR[i], cost_ind=f[0], S45_crn=front.S45[i], S45_ind=-f[1],
                        S45_ind_halfwidth=ev["S45_halfwidth"], P95_ind=ev["P95_em_min"], road_crn=front.road_share[i], road_ind=f[2],
                        S45_within_CI=bool(abs(front.S45[i] + f[1]) <= ev["S45_halfwidth"] + 0.005)))
    chk = pd.DataFrame(chk)
    # P95 along the front (36 CRN days)
    p95 = []
    for i in range(len(front)):
        _, ev = eval_one(json.loads(front.x[i])); p95.append(ev["P95_em_min"])
    front["P95_em_min"] = p95
    front.to_csv(os.path.join(tabd, f"{tag}_front.csv"), index=False)
    chk.to_csv(os.path.join(tabd, f"{tag}_resim_check.csv"), index=False)
    pd.DataFrame(ws).to_csv(os.path.join(tabd, f"{tag}_weighted_sum.csv"), index=False)
    pd.DataFrame(F2, columns=["cost_MINR", "negS45"]).to_csv(os.path.join(tabd, f"{tag}_front2.csv"), index=False)
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sc = ax[0].scatter(front.cost_MINR, front.S45 * 100, c=front.road_share * 100, cmap="viridis", s=30)
    plt.colorbar(sc, ax=ax[0], label="road share (%)")
    k = front[front.knee].iloc[0]; ax[0].scatter(k.cost_MINR, k.S45 * 100, s=250, facecolors="none", edgecolors="r", lw=2, label=f"knee: {k.vehicle}, {k.drones} drones")
    ax[0].axhline(90, color="grey", ls=":"); ax[0].set_xlabel("annual cost (M INR)"); ax[0].set_ylabel("S45 (%)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("3-objective NSGA-II front (CRN, 36 days)")
    ax[1].plot(f2[:, 0], -f2[:, 1] * 100, "o-", ms=4, label="NSGA-II (cost, S45)")
    ax[1].scatter([d["cost"] for d in ws], [d["S45"] * 100 for d in ws], marker="*", s=180, c="r", label="weighted-sum GA", zorder=5)
    for d in ws: ax[1].annotate(f"w={d['w']}", (d["cost"], d["S45"] * 100), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax[1].set_xlabel("annual cost (M INR)"); ax[1].set_ylabel("S45 (%)"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("Two-objective slice: weighted sum vs NSGA-II")
    ax[2].scatter(front.S45 * 100, front.P95_em_min, c=front.cost_MINR, cmap="plasma"); ax[2].set_xlabel("S45 (%)"); ax[2].set_ylabel("P95 emergency time (min)")
    ax[2].axhline(45, color="grey", ls=":"); ax[2].set_title("P95 is bimodal: ~35-45 min, or 150 min once >5% go by road"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(figd, f"{tag}_pareto.png"), dpi=120); plt.close()
    return front, chk, pd.DataFrame(ws)


def _init(sols, designs):
    setup(sols, designs); m5.static()


if __name__ == "__main__":
    import sys
    sols = json.load(open(os.path.join(data.OUT, "tables", "m4_solutions.json")))
    designs = [k for k in sols if not k.endswith("forced")]
    pop, gens = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (60, 35)
    front, chk, ws = run(sols, designs, pop=pop, gens=gens)
    pd.set_option("display.width", 250)
    print(front.drop(columns="x").round(3).to_string()); print(chk.round(3)); print(ws.drop(columns="x").round(3))


def merge_fronts(sols, designs, tag="m6"):
    """Union of every final population (3-obj NSGA-II, 2-obj NSGA-II, weighted-sum GA), re-evaluated on the
    CRN days, reduced to the 3-objective non-dominated set. Needed because the 2-obj run found a point that
    dominates part of the 3-obj front's (cost, S45) projection: the 3-obj run had not fully converged."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tabd = os.path.join(data.OUT, "tables"); figd = os.path.join(data.OUT, "figs")
    setup(sols, designs); m5.static()
    X = [np.load(os.path.join(tabd, f"{tag}_raw3.npz"))["X"], np.load(os.path.join(tabd, f"{tag}_raw2.npz"))["X"]]
    wsd = pd.read_csv(os.path.join(tabd, f"{tag}_weighted_sum.csv"))
    X.append(np.array([json.loads(x) for x in wsd.x]))
    X = np.vstack(X); src = ["nsga3"] * len(X[0:0])
    rows = []
    n3 = len(np.load(os.path.join(tabd, f"{tag}_raw3.npz"))["X"]); n2 = len(np.load(os.path.join(tabd, f"{tag}_raw2.npz"))["X"])
    for k, x in enumerate(X):
        f, ev = eval_one(x)
        name, sol, spares, r = decode(x)
        rows.append(dict(source="nsga3" if k < n3 else ("nsga2obj" if k < n3 + n2 else "weighted-sum"), vehicle=name, cost_MINR=f[0], S45=-f[1],
                         road_share=f[2], reserve=r, drones=sum(h["drones"] for h in sol["hubs"].values()), bays=sum(h["bays"] for h in sol["hubs"].values()),
                         spares=sum(spares.values()), hubs=len(sol["hubs"]), P95_em_min=ev["P95_em_min"], x=json.dumps(list(map(float, x)))))
    allp = pd.DataFrame(rows).drop_duplicates(["cost_MINR", "S45", "road_share"]).reset_index(drop=True)
    F = np.c_[allp.cost_MINR, -allp.S45, allp.road_share]
    front = allp[nondominated(F)].sort_values("cost_MINR").reset_index(drop=True)
    nd2 = nondominated(np.c_[front.cost_MINR, -front.S45]); sub = front[nd2].reset_index()
    front["knee"] = False
    front.loc[int(sub.loc[knee_index(np.c_[sub.cost_MINR, -sub.S45]), "index"]) if len(sub) > 2 else int(sub["index"].iloc[0]), "knee"] = True
    front.to_csv(os.path.join(tabd, f"{tag}_front.csv"), index=False)
    # verification on the 2-objective slice: every weighted-sum point lies on or behind the merged 2-D front
    f2 = front[nd2]
    for i, d in wsd.iterrows():
        best = f2[f2.cost_MINR <= d.cost + 1e-9].S45.max()
        wsd.loc[i, "front_best_S45_at_same_or_lower_cost"] = best; wsd.loc[i, "behind_or_on_front"] = bool(d.S45 <= best + 1e-9)
    wsd.to_csv(os.path.join(tabd, f"{tag}_weighted_sum.csv"), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(allp.cost_MINR, allp.S45 * 100, c="lightgrey", s=12, label="all final-population points")
    sc = ax[0].scatter(front.cost_MINR, front.S45 * 100, c=front.road_share * 100, cmap="viridis", s=40, edgecolor="k", label="merged 3-obj front")
    plt.colorbar(sc, ax=ax[0], label="road share (%)")
    k = front[front.knee].iloc[0]
    ax[0].scatter(k.cost_MINR, k.S45 * 100, s=260, facecolors="none", edgecolors="r", lw=2, label=f"knee: {k.vehicle}, {k.drones} drones")
    ax[0].set_xlabel("annual cost (M INR)"); ax[0].set_ylabel("S45 (%) on 36 CRN days"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("Pareto front (S45 range is within ±2.5 pp sim noise)")
    ax[1].plot(f2.cost_MINR, f2.S45 * 100, "o-", label="merged front, (cost, S45) slice")
    ax[1].scatter(wsd.cost, wsd.S45 * 100, marker="*", s=180, c="r", label="weighted-sum GA", zorder=5)
    ax[1].set_xlabel("annual cost (M INR)"); ax[1].set_ylabel("S45 (%)"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("Weighted sum lies on/behind the front")
    ax[2].scatter(front.road_share * 100, front.cost_MINR, c=front.S45 * 100, cmap="plasma", s=40); ax[2].set_xlabel("road share (%)"); ax[2].set_ylabel("annual cost (M INR)")
    ax[2].set_title(f"P95 = {front.P95_em_min.min():.0f}-{front.P95_em_min.max():.0f} min on all front points:\n>5% of emergencies go by road on storm days"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(figd, f"{tag}_pareto.png"), dpi=120); plt.close()
    return front, allp, wsd
