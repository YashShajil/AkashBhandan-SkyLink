# SkyLink — Medical Drone Delivery Network (ED6002, Group 7)

ED6002 Optimization Methods in Engineering Design, IIT Madras — end-semester project.
Group 7: Priyanka G (AE22B066), Yash Shajil (AE22B110).

Kaveripuram district: 24 primary health centres (PHCs), 7 candidate hospital hubs (H0–H6) and P09.
Goal: emergency items (antivenom, blood) to every PHC within 45 minutes at a defensible annual cost.

## Run it

```bash
pip install -r requirements.txt
python run_all.py          # ~5 min: M1 -> M7, re-uses the saved M2 and M6 results
python run_all.py --full   # ~1 h: also re-runs M2 topology optimisation and M6 NSGA-II
```

Run from this folder. Results go to `outputs/figs` and `outputs/tables`; the live one-day view is
`outputs/live/skylink_day.html` (open in a browser). Each module also runs on its own:
`python -m skylink.m4_network`, etc.

## Modules

| Module | File | What it does |
|---|---|---|
| M0 data | `skylink/data.py` | Loads and audits the data pack (`outputs/tables/data_audit.csv`), obstacle-avoiding paths |
| M1 vehicle | `skylink/m1_vehicle.py` | Min take-off mass s.t. radius ≥ R_req (22 km); MDF and IDF, complex-step gradients, catalogue snap |
| M2 plate | `skylink/m2_new.py` | SIMP topology optimisation of the hub plate (Mindlin plate elements), cut-out comparison, stresses |
| M3 surrogate | `skylink/m3_surrogate.py` | Kriging / RBF / quadratic FM surrogates from the 30 oracle points, LOOCV |
| M4 network | `skylink/m4_network.py` | MILP hub location, PHC assignment, fleet and bays; brute-force check on 4 hubs × 8 PHCs; blood restock routing |
| M5 simulator | `skylink/m5_sim.py` | SimPy discrete-event simulation with hourly wind, grounding, outages, road fallback; M/M/c check |
| M6 multi-objective | `skylink/m6_moo.py` | NSGA-II with common random numbers over vehicle (18–26 km), fleet, spares, reserve; weighted-sum check |
| M7 decision | `skylink/m7_decision.py` | Robustness (calm/typical/monsoon weeks), 300-week reliability, sensitivity, minimax-regret decision |

## Data

- `data/real/` — course data pack (sites, demand, hourly profile, wind, no-fly zones, costs, components)
- `oracle/request_k.csv`, `oracle/response_k.csv` (k = 1–3) — the 30 rotor-oracle points used by M3

## Result (final run)

| | |
|---|---|
| Recommended system | 20 km drones, hubs H0 H1 H2 H3 H6, 16 drones, 9 charging bays, 16 spare packs |
| Annual cost | ₹2.18 crore |
| Emergencies within 45 min (typical week) | 95.6% (slowest 5% within 29 min) |
| Chance of a week with ≥ 90% on time | 76 ± 5% |
| Worst monsoon week (67 h grounded) | ~54% within 45 min; the rest by road |
| Decision rule | No design meets P ≥ 0.8, so minimax regret in S45 across calm/typical/monsoon weeks |

Blood: H0 is the only blood bank; hubs keep a small stock topped up daily at 06:00 by drone
(battery-swap relays) or van. This is a team modelling assumption, not part of the brief.

## Repository layout

```text
run_all.py            one script, M1 -> M7
requirements.txt
skylink/              modules (above)
data/real/            data pack
oracle/               oracle requests and responses
outputs/figs, tables  results of the final run
end_sem_ED6002_PROJECT.pdf
```
