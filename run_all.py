"""Run the whole SkyLink chain M1 -> M7, one module after another.

    python run_all.py            runs all modules M1 -> M7 end-to-end (including M2 plate and M6 NSGA-II)
    python run_all.py --fast     fast mode: re-uses saved M2 and M6 results if available (~5 min)

Each module runs exactly as if you ran it on its own (python -m skylink.<module>), so the console output is
the same. Results go to outputs/figs and outputs/tables. Run from this folder (the one containing skylink/).
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "skylink")
T = os.path.join(HERE, "outputs", "tables")
FAST = "--fast" in sys.argv
t0 = time.time()


def stamp(msg):
    print(f"\n[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


def run(module, *args):
    stamp(f"python -m skylink.{module} {' '.join(args)}".rstrip())
    r = subprocess.run([sys.executable, "-m", f"skylink.{module}", *args], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\n{module} failed (exit code {r.returncode}) - fix it, then run again.")


def have(*files):
    return all(os.path.exists(os.path.join(T, f)) for f in files)


m2_name = "m2_new" if os.path.exists(os.path.join(PKG, "m2_new.py")) else "m2_topopt"
has_m3 = os.path.exists(os.path.join(PKG, "m3_surrogate.py"))

run("m1_vehicle")                                                    # M1: vehicle at R_REQ
if not FAST or not have("m2_summary.json"):
    run(m2_name)                                                     # M2: plate topology optimization
else:
    stamp(f"M2 skipped - using saved outputs/tables/m2_summary.json (run without --fast to re-run)")
if has_m3:
    run("m3_surrogate", "fit")                                       # M3: FM surrogate
else:
    stamp("M3 skipped - skylink/m3_surrogate.py not found (M4-M7 use the constant FM = 0.65)")
run("m4_network")                                                    # M4: hub network + blood restock
run("m5_sim")                                                        # M5: simulator, CI, live view
if not FAST or not have("m6_front.csv", "m6set_solutions.json"):
    run("m6_moo", "35", "15")                                        # M6: NSGA-II with CRN
else:
    stamp("M6 skipped - using saved outputs/tables/m6_front.csv (run without --fast to re-run)")
run("m7_decision")                                                   # M7: robustness, reliability, decision
stamp("All done. Live view: outputs/live/skylink_day.html")
