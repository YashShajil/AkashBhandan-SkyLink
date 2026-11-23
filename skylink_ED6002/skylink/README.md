# SkyLink — medical drone delivery network (ED6002 project)

One script runs the whole chain:

```bash
pip install numpy scipy pandas matplotlib pyyaml networkx shapely scikit-learn simpy pymoo
python run_all.py          # full run (~25 min on 2 cores; M2 ~4 min, M6 ~15 min)
python run_all.py --fast   # reuse cached M2 / M6 results
```

Outputs go to `outputs/figs`, `outputs/tables`, `outputs/live/skylink_day.html` (live view of one day).

## ⚠️ Before submission — three things the code cannot do for you

1. **Real data pack.** `skylink_data.zip` was not available when this was built. `data/reconstructed/`
   holds the tables printed in the brief, plus a **synthetic** wind year (calibrated to the monthly table)
   and a reconstructed tube catalogue (only 7 of 30 tubes are printed). Unzip the real files into
   `data/real/`. The loader prefers them automatically, and `outputs/tables/data_audit.csv` will list every
   planted problem it finds and what was done about it. Re-run everything afterwards.
2. **Oracle.** `oracle/request_1.csv` is ready to email (replace `<ROLL_NO>`, or set `ROLL_NO=...`).
   Save each reply as `oracle/response_k.csv`, then generate the next request:
   `python -m skylink.m3_surrogate request 2` (and later `request 3`). Requests 2–3 are adaptive: they depend on reply 1.
   Until replies exist, a **MOCK oracle** (`skylink/mock_oracle.py`) fills in, and every M3 output is stamped MOCK.
3. **Formulation cards and predictions** must be written in your own words (course rule 1, rule 6).
   `docs/formulation_fact_sheet.md` lists the numbers and decisions to check against, but it is not a card.

## Chain

| Module | File | Feeds |
|---|---|---|
| data audit | `skylink/data.py` | every module (cleaned data, obstacle-avoiding distances) |
| M1 vehicle MDO | `skylink/m1_vehicle.py` | designs V1–V5 (`m1_designs.csv`) → M4, M6 |
| M2 plate SIMP | `skylink/m2_topopt.py` | plate mass → M1 |
| M3 FM surrogate | `skylink/m3_surrogate.py` | FM(D,T) → M1 final |
| M4 network MILP | `skylink/m4_network.py` | hubs/fleet per design → M5, M6 |
| M5 simulator | `skylink/m5_sim.py`, `m5_report.py` | evaluation engine for M6, M7 |
| M6 NSGA-II + CRN | `skylink/m6_moo.py` | Pareto front → M7 |
| M7 decision | `skylink/m7_decision.py` | recommended system |

## Verification index (one item per module)

| Module | Check | Output |
|---|---|---|
| data | audit log of every detected problem + fix | `tables/data_audit.csv` |
| M1 | MDA convergence plot; complex-step gradients vs FD at 1e-4 and 1e-7; MDF vs IDF | `figs/m1_mda_tradeoff.png`, `tables/m1_gradcheck.csv`, `tables/m1_summary.json` |
| M2 | MBB benchmark (density/sensitivity filter, no filter, p=1); plate element vs Timoshenko plate | `figs/m2_mbb_benchmark.png`, `tables/m2_summary.json` |
| M3 | LOOCV table, parity plot, replicate noise, non-smooth region | `tables/m3_loocv.csv`, `figs/m3_surrogate.png` |
| M4 | brute force 4 hubs × 8 PHCs = MILP; LP gap; big-M | `tables/m4_verification.json`, `tables/m4_cost_vs_vehicle.csv` |
| M5 | M/M/c vs Erlang C (3 cases); days needed for ±3 pp CI | `tables/m5_mmc_verification.csv`, `figs/m5_ci.png` |
| M6 | weighted sum on/behind NSGA-II; re-simulation with independent days | `tables/m6_weighted_sum.csv`, `tables/m6_resim_check.csv` |
| M7 | robustness / reliability / sensitivity tables | `tables/m7_*.csv` |
