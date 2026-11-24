"""Run the whole SkyLink chain M1 -> M7, one module after another.

    python run_all.py            fast: re-uses the slow M2 and M6 results if they are already saved (~10-15 min)
    python run_all.py --full     also re-runs M2 topology optimisation and M6 NSGA-II (~1 h or more)

Each module runs exactly as if you ran it on its own (python -m skylink.<module>), so the console output is
the same. Results go to outputs/figs and outputs/tables. Run from this folder (the one containing skylink/).
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "skylink")
T = os.path.join(HERE, "outputs", "tables")
FULL = "--full" in sys.argv
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
has_m3 = os.path.exists(os.path.join(PKG, "m3_surrogate.py")) and os.path.exists(os.path.join(HERE, "oracle", "request_1.csv"))

run("m1_vehicle")                                                    # M1: vehicle at R_REQ
if FULL or not have("m2_summary.json"):
    run(m2_name)                                                     # M2: plate (slow)
else:
    stamp(f"M2 skipped - using saved outputs/tables/m2_summary.json (run_all.py --full to redo)")
if has_m3:
    run("m3_surrogate", "fit")                                       # M3: FM surrogate
else:
    stamp("M3 skipped - skylink/m3_surrogate.py or oracle/request_1.csv not found (M4-M7 use the constant FM = 0.65)")
run("m4_network")                                                    # M4: hub network + blood restock
run("m5_sim")                                                        # M5: simulator, CI, live view
if FULL or not have("m6_front.csv", "m6set_solutions.json"):
    run("m6_moo")                                                    # M6: NSGA-II (slow)
else:
    stamp("M6 skipped - using saved outputs/tables/m6_front.csv (run_all.py --full to redo)")
run("m7_decision")                                                   # M7: robustness, reliability, decision
stamp("All done. Live view: outputs/live/skylink_day.html")
