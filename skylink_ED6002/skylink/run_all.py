"""SkyLink - run the whole chain M0..M7 from one script.

    python run_all.py            # everything (M2 ~4 min, M6 ~15 min on 2 cores)
    python run_all.py --fast     # reuse cached M2 and M6 results if present

Each module's output feeds the next:
  data audit -> M1 (baseline, verification) -> M2 plate mass -> M1 (+plate) -> M3 FM surrogate
  -> M1 final (+plate +FM) designs V1..V5 -> M4 network per design -> M5 sim of cheapest network
  -> M6 Pareto front over vehicle/fleet/bays/spares/reserve -> M7 robustness, reliability, decision.
"""
import os, sys, json, time
import pandas as pd
from skylink import data, m1_vehicle as m1, m2_topopt as m2, m3_surrogate as m3, m4_network as m4
from skylink import m5_report, m6_moo as m6, m7_decision as m7

FAST = "--fast" in sys.argv
T = os.path.join(data.OUT, "tables")
os.makedirs(T, exist_ok=True); os.makedirs(os.path.join(data.OUT, "figs"), exist_ok=True)
t0 = time.time(); chain = {}
def stamp(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

stamp("data audit"); audit = data.write_audit(); chain["data_sources"] = data.sources(); chain["audit_items"] = len(audit)

stamp("M1 baseline (FM=0.65, plate 0.25 kg): MDA, MDF vs IDF, KKT, gradient check")
r_base, d_base, _ = m1.run(m_plate=0.25, tag="m1_base")
chain["M1_base"] = dict(R_km=r_base["mdf"]["R_km"], mdf_idf_rel=r_base["mdf_idf_R_rel_diff"], km_per_100g=r_base["kkt"]["km_lost_per_100g"],
                        km_per_100g_reopt=r_base["reopt_km_lost_per_100g"], active=r_base["kkt"]["active"], uncapped_R=r_base["uncapped_maxR"]["R_km"])

if FAST and os.path.exists(os.path.join(T, "m2_summary.json")):
    s2 = json.load(open(os.path.join(T, "m2_summary.json"))); stamp("M2 cached")
else:
    stamp("M2 SIMP: MBB benchmark, plate element check, volume-fraction sweep"); s2, _ = m2.run()
m_plate = s2["picked_plate_mass_kg"]; chain["M2"] = dict(volfrac=s2["picked_volfrac"], plate_kg=m_plate)

stamp("M1 re-run with M2 plate mass")
r_pl, _, _ = m1.run(m_plate=m_plate, tag="m1_plate", quick=True)
chain["M1_plate"] = dict(R_km=r_pl["mdf"]["R_km"], dR_km=r_pl["mdf"]["R_km"] - r_base["mdf"]["R_km"])

stamp("M3 surrogate (oracle responses if present, else MOCK)")
for k in (1, 2, 3):
    if not os.path.exists(os.path.join(m3.ORA, f"request_{k}.csv")):
        raise SystemExit(f"oracle/request_{k}.csv missing: run  python -m skylink.m3_surrogate request {k}")
kmod, s3, t3 = m3.fit_all(); fm = m3.fm_function(kmod)
chain["M3"] = s3

stamp("M1 final (plate + FM surrogate): designs V1..V5")
r_fin, designs, _ = m1.run(m_plate=m_plate, fm=fm, tag="m1")
chain["M1_final"] = dict(R_km=r_fin["mdf"]["R_km"], x=r_fin["mdf"], shift_vs_plate=dict(
    dR_km=r_fin["mdf"]["R_km"] - r_pl["mdf"]["R_km"], dV=r_fin["mdf"]["V_ms"] - r_pl["mdf"]["V_ms"], dE=r_fin["mdf"]["E_Wh"] - r_pl["mdf"]["E_Wh"]),
    mdf_idf_rel=r_fin["mdf_idf_R_rel_diff"], km_per_100g=r_fin["kkt"]["km_lost_per_100g"])
# trust region check: designs must sit below the surrogate's disk-loading limit
dl = (designs.MTOW_kg * 9.81 / 4) / (3.14159 * designs.D_m ** 2 / 4)
chain["M1_final"]["max_disk_loading"] = float(dl.max()); chain["M1_final"]["within_FM_trust_region"] = bool(dl.max() < s3["DL_trust_limit"])

stamp("M4 MILP per design + brute-force check")
df4, sols, ver = m4.run(designs, m_plate, fm)
best4 = df4.annual_cost_MINR.idxmin()
chain["M4"] = dict(best=best4, cost=df4.annual_cost_MINR.to_dict(), brute_force_match=ver["rel_diff"] < 1e-9, LP_gaps=df4.LP_gap_pct.to_dict(),
                   H0_forced_cost=float(df4.loc[best4, "cost_if_H0_forced_MINR"]))

stamp(f"M5 simulator on {best4} network: M/M/c check, CI study, live view")
s5 = m5_report.run(sols[best4], sols[best4]["vp"], label=f"{best4} network (M4 optimum)")
chain["M5"] = dict(days_for_CI=s5["ci"]["days"], S45=s5["ci"]["S45"], hw=s5["ci"]["S45_halfwidth"], P95=s5["ci"]["P95_em_min"],
                   mmc_rel_err=[m["rel_err"] for m in s5["mmc"]])

design_names = [k for k in sols if not k.endswith("forced")]
if FAST and os.path.exists(os.path.join(T, "m6_raw2.npz")):
    front = pd.read_csv(os.path.join(T, "m6_front.csv")); stamp("M6 cached")
else:
    stamp("M6 NSGA-II with CRN (slow)"); front, chk, ws = m6.run(sols, design_names, pop=48, gens=30)
stamp("M6 merge final populations into one non-dominated front")
front, allp, ws = m6.merge_fronts(sols, design_names)
chain["M6"] = dict(n_front=len(front), knee=front[front.knee].drop(columns="x").iloc[0].to_dict())

stamp("M7 robustness, reliability, sensitivity, decision")
s7, rob, rel, sens, fixed = m7.run(sols, front, designs, m_plate, fm)
chain["M7"] = dict(chosen=s7["chosen"], rule=s7["rule"], system=s7["chosen_system"]["vehicle"])
stamp("live views of the recommended system (normal outage day + storm day)")
from skylink import m5_sim as m5
name, sol, spares, r = m6.decode(json.loads(front.x[s7["chosen"]]))
cfg = m5.Config(sol, sol["vp"], spares=spares, reserve=r)
lab = f"Recommended system: {name}, {sum(h['drones'] for h in sol['hubs'].values())} drones, reserve {r:.0%}"
m5_report.live_view(cfg, m5_report.pick_story_day(), os.path.join(data.OUT, "live", "skylink_day.html"), label=lab)
g = data.wind().groupby("day").grounded.sum(); sd = int(g[(g >= 8) & (g <= 14)].index[0])
m5_report.live_view(cfg, sd, os.path.join(data.OUT, "live", "skylink_storm_day.html"), label=lab + " - storm day")
json.dump(chain, open(os.path.join(T, "chain_summary.json"), "w"), indent=1, default=str)
stamp("done -> outputs/")
