"""M7 - robustness, reliability, sensitivity and the decision rule.

Run after M6:  python -m skylink.m7_decision      (from the skylink folder, ~10-15 min)
Uses the M6 Pareto front (outputs/tables/m6_front.csv) and the M6 vehicle set (m6set_solutions.json:
one M1 design + M4 network per radius in m6_moo.M1_SET).
"""
from __future__ import annotations
import sys, os
if __package__ in (None, ""):          # allow running this file directly (PyCharm Run button)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "skylink"
import os, json, math
import numpy as np, pandas as pd
from . import data, m5_sim as m5, m6_moo as m6
from . import m1_vehicle as m1, m4_network as m4


def wind_weeks():
    W = data.wind()
    g = W.groupby("day").agg(gr=("grounded", "sum"), w=("mean_wind_ms", "mean")).reset_index()
    g["gr7"] = g.gr.rolling(7).sum(); g["w7"] = g.w.rolling(7).mean()
    g = g.dropna()
    calm = int(g.sort_values(["gr7", "w7"]).day.iloc[0]); worst = int(g.sort_values(["gr7", "w7"], ascending=False).day.iloc[0])
    med = g.gr7.median(); typ = int(g.iloc[(g.gr7 - med).abs().argsort()].day.iloc[0])
    wk = lambda end: list(range(end - 6, end + 1))
    info = {k: dict(days=wk(e), grounded_h=int(g.set_index("day").gr7[e])) for k, e in [("calm", calm), ("typical", typ), ("worst monsoon", worst)]}
    return info


def pick_candidates(front):
    f = front.copy()
    knee = f[f.knee].index[0]
    c2 = f[(f.S45 >= 0.90) & (f.road_share <= 0.03)]
    c2 = c2.cost_MINR.idxmin() if len(c2) else f.S45.idxmax()
    kc = f.loc[knee, "cost_MINR"]
    c3 = f[f.cost_MINR <= 1.15 * kc].S45.idxmax()
    ids = list(dict.fromkeys([knee, c2, c3]))
    for i in f.sort_values("cost_MINR").index:   # fill to 3 with nearest-cost neighbours of the knee
        if len(ids) >= 3: break
        if i not in ids and abs(f.cost_MINR[i] - kc) < 3: ids.append(i)
    labels = {knee: "knee", c2: "cheapest with S45>=90% & road<=3%", c3: "best S45 within +15% of knee cost"}
    return [(i, labels.get(i, "neighbour of knee")) for i in ids[:3]]


def robustness(front, cands, reps=6):
    weeks = wind_weeks(); rows = []
    for i, lab in cands:
        x = json.loads(front.x[i])
        for wn, wi in weeks.items():
            S = []; costs = []; p95 = []
            for r in range(reps):
                f, ev = m6.eval_one(x, wi["days"], [800 + 100 * r + k for k in range(7)])
                S.append(-f[1]); costs.append(f[0]); p95.append(ev["P95_em_min"])
            rows.append(dict(candidate=i, role=lab, scenario=wn, grounded_h_in_week=wi["grounded_h"], S45_mean=np.mean(S), S45_worst=np.min(S),
                             P95_mean=np.mean(p95), cost_MINR_annualised=np.mean(costs)))
    return pd.DataFrame(rows), weeks


def _rel_task(t):
    x, days, seeds, dm, av = t
    f, _ = m6.eval_one(x, days, seeds, demand_mult=dm, availability=av)
    return -f[1]


def reliability(front, cands, n=300, seed=5, pool=None):
    """Monte Carlo: n samples per candidate (demand x0.75-1.25, availability 90-98 %, 7 random days each).
    Samples are drawn first (same random numbers as before), then simulated in parallel on all cores."""
    rng = np.random.default_rng(seed); rows = []
    for i, lab in cands:
        x = json.loads(front.x[i]); tasks = []
        for s in range(n):
            dm = rng.uniform(0.75, 1.25); av = rng.uniform(0.90, 0.98)
            days = list(map(int, rng.integers(1, 366, 7))); seeds = list(map(int, rng.integers(0, 10 ** 6, 7)))
            tasks.append((x, days, seeds, dm, av))
        S = pool.map(_rel_task, tasks, chunksize=8) if pool is not None else [_rel_task(t) for t in tasks]
        d = pd.DataFrame(dict(dm=[t[3] for t in tasks], av=[t[4] for t in tasks], S45=S)); p = (d.S45 >= 0.90).mean()
        rows.append(dict(candidate=i, role=lab, P_S45_ge_90=p, P_ci95=1.96 * math.sqrt(p * (1 - p) / n), n_samples=n,
                         S45_p05=d.S45.quantile(0.05), S45_median=d.S45.median(),
                         P_given_demand_hi=(d[d.dm > 1.1].S45 >= 0.9).mean(), P_given_avail_lo=(d[d.av < 0.92].S45 >= 0.9).mean()))
    return pd.DataFrame(rows)


def sensitivity(sols, chosen_x, designs, m_plate, fm, base_cost_vehicle):
    out = []
    ws = m4.wind_stats()
    def best_m4(dsg, dm=1.0, dp=1.0, label="", eb=150.0):
        best = None
        for n, row in dsg.iterrows():
            vp = m4.vehicle_params(row, m_plate, fm, eb=eb); M = m4.build(vp, ws, demand_mult=dm, drone_price_mult=dp); r = m4.solve(M)
            s = m4.decode(M, r); s["vp"] = {k: v for k, v in vp.items() if k != "x"}; s.setdefault("breakdown", {})
            cost = m4.restock_routing(s, vp, ws, M)                     # blood restock H0 -> hubs, as in M4
            if best is None or cost < best[1]: best = (n, cost, s)
        return dict(case=label, best_vehicle=best[0], M4_cost_MINR=best[1] / 1e6, hubs=" ".join(best[2]["hubs"]),
                    drones=sum(h["drones"] for h in best[2]["hubs"].values()))
    out.append(best_m4(designs, label="baseline"))
    # battery specific energy +20 % -> re-size vehicles (M1) then network (M4)
    d_eb = m1_set(m_plate, fm, eb=180.0)
    out.append(best_m4(d_eb, label="battery specific energy +20% (re-sized vehicles)", eb=180.0))
    out.append(best_m4(designs, dp=0.7, label="drone price -30%"))
    for yr in (1, 2, 3):
        out.append(best_m4(designs, dm=1.15 ** yr, label=f"demand +15%/yr, year {yr} (x{1.15**yr:.2f})"))
    sens = pd.DataFrame(out)
    # what happens to the CHOSEN operating system if nothing is changed
    days = m6._G["days"]; seeds = m6._G["seeds"]; fixed = []
    for lab, kw in [("baseline", {}), ("drone price -30%", dict(drone_price_mult=0.7))] + \
                   [(f"demand year {y} (x{1.15**y:.2f})", dict(demand_mult=1.15 ** y)) for y in (1, 2, 3)] + \
                   [(f"weekday of day 1 shifted by {k} (outage-day assumption)", dict(weekday_shift=k)) for k in (1, 2)]:
        f, ev = m6.eval_one(chosen_x, days, seeds, **kw)
        fixed.append(dict(case=lab, cost_MINR=f[0], S45=-f[1], road_share=f[2], util=ev["util"], P95=ev["P95_em_min"]))
    return sens, pd.DataFrame(fixed)


def m1_set(m_plate, fm, eb=m1.EB):
    """M1 design for every radius in m6.M1_SET (min take-off mass s.t. R >= R_req), indexed R18, R20, ..."""
    ds = []
    for R in m6.M1_SET:
        _, d, _ = m1.run(m_plate=m_plate, fm=fm, R_req=R, eb=eb, tag="m7_m1", make_figs=False, quick=True)
        ds.append(d.rename(index={d.index[0]: f"R{R:g}"}))
    return pd.concat(ds)


def run(sols, front, designs, m_plate, fm, tag="m7"):
    tabd = os.path.join(data.OUT, "tables")
    m6.setup(sols, [k for k in sols if not k.endswith("forced")]); m5.static()
    cands = pick_candidates(front)
    rob, weeks = robustness(front, cands)
    from multiprocessing import Pool
    procs = max(1, min(4, os.cpu_count() or 1))
    with Pool(procs, initializer=m6._init, initargs=(sols, [k for k in sols if not k.endswith("forced")])) as pool:
        rel = reliability(front, cands, pool=pool)
    # decision rule: lowest cost among candidates with typical-week S45 >= 90% and P(S45>=90%) >= 0.8;
    # if none qualifies, the candidate with the smallest maximum regret in S45 across the three wind weeks
    typ = rob[rob.scenario == "typical"].set_index("candidate")
    relx = rel.set_index("candidate")
    q = [i for i, _ in cands if typ.loc[i, "S45_mean"] >= 0.90 and relx.loc[i, "P_S45_ge_90"] >= 0.8]
    piv = rob.pivot_table(index="candidate", columns="scenario", values="S45_mean")
    regret = (piv.max(axis=0) - piv).max(axis=1)
    if q:
        chosen = min(q, key=lambda i: front.cost_MINR[i]); rule = "lowest cost meeting S45>=90% (typical week) and P(S45>=90%)>=0.8"
    else:
        chosen = int(regret.idxmin()); rule = "minimax regret in S45 across calm/typical/worst weeks (no candidate met the service rule)"
    chosen_x = json.loads(front.x[chosen])
    sens, fixed = sensitivity(sols, chosen_x, designs, m_plate, fm, front.vehicle[chosen])
    rob.to_csv(os.path.join(tabd, f"{tag}_robustness.csv"), index=False)
    rel.to_csv(os.path.join(tabd, f"{tag}_reliability.csv"), index=False)
    sens.to_csv(os.path.join(tabd, f"{tag}_sensitivity_network.csv"), index=False)
    fixed.to_csv(os.path.join(tabd, f"{tag}_sensitivity_chosen_system.csv"), index=False)
    name, sol, spares, r = m6.decode(chosen_x)
    summ = dict(candidates=[dict(idx=int(i), role=l, **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                                                          for k, v in front.loc[i].drop("x").items()}) for i, l in cands],
                chosen=int(chosen), rule=rule, regret=regret.to_dict(), weeks={k: dict(first_day=v["days"][0], grounded_h=v["grounded_h"]) for k, v in weeks.items()},
                chosen_system=dict(vehicle=name, reserve=r, hubs={h: dict(drones=v["drones"], bays=v["bays"], spares=spares.get(h, 0), phcs=v["phcs"]) for h, v in sol["hubs"].items()},
                                   road_only=sol.get("road_only", []), vp=sol["vp"]))
    json.dump(summ, open(os.path.join(tabd, f"{tag}_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    # live view of the CHOSEN system (replaces the M5 view of the M4 network, same story day)
    if tag == "m7":
        n_dr = int(sum(v["drones"] for v in sol["hubs"].values()))
        os.makedirs(os.path.join(data.OUT, "live"), exist_ok=True)
        m5.live_view(m5.Config(sol, sol["vp"], spares=spares, reserve=r), m5.pick_story_day(),
                     os.path.join(data.OUT, "live", "skylink_day.html"),
                     label=f"Chosen system (M7): {name[1:]} km drones, {n_dr} drones, reserve {r:.2f}")
        print("Live view of the chosen system: outputs/live/skylink_day.html")
    return summ, rob, rel, sens, fixed


if __name__ == "__main__":
    T = os.path.join(data.OUT, "tables")
    sols = {k: v for k, v in json.load(open(os.path.join(T, "m6set_solutions.json"), encoding="utf-8")).items() if not k.endswith("forced")}
    front = pd.read_csv(os.path.join(T, "m6_front.csv"))
    p2 = os.path.join(T, "m2_summary.json")
    m_plate = json.load(open(p2, encoding="utf-8"))["picked_plate_mass_kg"] if os.path.exists(p2) else 0.25
    fm = None
    try:
        from . import m3_surrogate as m3
        k, _, _ = m3.fit_all(make_figs=False); fm = m3.fm_function(k)
    except Exception:
        pass
    designs = m1_set(m_plate, fm)
    s7, rob, rel, sens, fixed = run(sols, front, designs, m_plate, fm)
    pd.set_option("display.width", 250)
    print("ROBUSTNESS\n", rob.round(3).to_string(index=False))
    print("\nRELIABILITY\n", rel.round(3).to_string(index=False))
    print("\nSENSITIVITY (network re-optimised)\n", sens.round(3).to_string(index=False))
    print("\nSENSITIVITY (chosen system unchanged)\n", fixed.round(3).to_string(index=False))
    print("\nDECISION:", s7["rule"])
    cs = s7["chosen_system"]; print("Chosen:", cs["vehicle"], "reserve", cs["reserve"])
    for h, v in cs["hubs"].items(): print(f"  {h}: {v['drones']} drones, {v['bays']} bays, {v['spares']} spares -> {v['phcs']}")
    print("Saved: outputs/tables/m7_robustness.csv, m7_reliability.csv, m7_sensitivity_network.csv, m7_sensitivity_chosen_system.csv, m7_summary.json")
