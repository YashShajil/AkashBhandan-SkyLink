# SkyLink – ED6002 end-semester project (M1–M7)

Drone delivery network for Kaveripuram district: vehicle sizing → hub plate → FM surrogate → hub network →
operations simulator → multi-objective tuning → decision.

## Setup
```
pip install -r requirements.txt
```
Run every command from this folder (the one that contains `skylink/`, `data/`, `oracle/`).

## Run
| Module | Command | Time | Main outputs |
|---|---|---|---|
| all (fast) | `python run_all.py` | ~5–15 min | everything; re-uses saved M2/M6 results if present |
| all (full) | `python run_all.py --full` | 1 h + | also re-runs M2 and M6 |
| M1 vehicle (MDF/IDF, min take-off mass, R ≥ `R_REQ`) | `python -m skylink.m1_vehicle` | < 1 min | `outputs/figs/m1_*`, `outputs/tables/m1_*` |
| M2 hub plate (SIMP topology optimisation, stresses, cut-outs) | `python -m skylink.m2_new` | several min | `m2_*` |
| M3 FM surrogate (oracle, kriging, adaptive requests) | `python -m skylink.m3_surrogate fit` | < 1 min | `m3_*` |
| M4 hub network MILP (+ blood restock H0 → hubs) | `python -m skylink.m4_network` | 1–2 min | `m4_network.png`, `m4_solutions.json` |
| M5 discrete-event simulator + live view | `python -m skylink.m5_sim` | ~1 min | `outputs/live/skylink_day.html`, `m5_*` |
| M6 NSGA-II with common random numbers | `python -m skylink.m6_moo [pop gens]` | 30–60 min (default 60 35) | `m6_pareto.png`, `m6_front.csv` |
| M7 robustness, reliability, sensitivity, decision | `python -m skylink.m7_decision` | 2–5 min (needs M6) | `m7_*` |

Each file can also be run directly (PyCharm Run button).

## Oracle (M3)
- `oracle/request_k.csv` – points sent to the TA (k = 1, 2, 3; budget 15 + 8 + 7 = 30).
- `oracle/response_k.csv` – the TA's reply, columns `D_m,T_N,FM`. Save each reply under this name.
- While a reply is missing, M3 uses `skylink/mock_oracle.py` for those points and marks them **MOCK**.
- `python -m skylink.m3_surrogate request 2` builds the next request from the data so far.

## Key settings
- `skylink/m1_vehicle.py`: `R_REQ = 22.0` km (required radius at full payload).
- `skylink/m4_network.py`: `BLOOD_HUB = "H0"`, `BLOOD_MODE = "relay"` (H0 is the only blood bank; hubs are
  restocked daily, drone via battery-swap stops where a hop chain fits, van otherwise), `RESUPPLY_PER_DAY = 1`.
- `skylink/m5_sim.py`: emergency road timer 90 min, routine/lab 6 h; forecast grounding (gust ≥ 14 m/s).
- `skylink/m6_moo.py`: `M1_SET = (18, 20, 22, 24, 26)` km vehicle set.

Data: `data/real/` (course data pack).
