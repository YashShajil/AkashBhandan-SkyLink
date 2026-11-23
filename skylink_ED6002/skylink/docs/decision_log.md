# Decision log — AI suggestions (draft for Priyanka to complete)

Rule 4 of the brief: for each non-trivial choice, record what the AI suggested, what **you** kept or changed, and **why**.
The first two columns below are what the AI (Claude) proposed and implemented. The last column is yours:
write it in your own words, after checking each item. Strike or change anything you disagree with, and change the code to match.

| # | Module | What the AI suggested / did | Evidence it produced | Your verdict (kept / changed) and why |
|---|---|---|---|---|
| 1 | data | Data pack was not attached at first, so the AI rebuilt it from the brief plus a synthetic wind year. When the real zip arrived, the loader switched to it automatically. | `data_audit.csv` lists sources = real | |
| 2 | data | Routine hourly shares sum to 1.1993. **Renormalise** to 1 (keep the daily totals in demand.csv; keep the shape). Alternative: treat demand.csv as wrong and scale it ×1.2. | audit row 1 | |
| 3 | data | Wind file has no weekday column, so the AI **assumed day 1 = Monday** for the outage days and tested shifts of +1 and +2 days in M7. | `m7_sensitivity_chosen_system.csv` | |
| 4 | data | Circle no-fly zone modelled as a **circumscribed** 64-gon (never cuts the circle) plus a visibility graph for shortest paths. Max detour factor 2.04. | `data.py`, M4 map | |
| 5 | data | Checked for planted errors: wind (gaps, duplicates, gust<mean, sentinels, spikes, month/day mismatch), catalogue (Wh/kg, ₹/Wh, tube mass vs geometry, motor mass vs model), sites inside zones, demand per-capita outliers. Only the routine-profile error was found in the files. | audit | |
| 6 | M1 | Unconstrained max-radius drives E, D, d and V all to their upper bounds (20 kg, 2000 Wh). The AI **reformulated**: max R subject to MTOW ≤ cap (MTOW drives drone price), and swept the cap 7–14 kg to get V1–V5. | `m1_summary.json` → `uncapped_maxR` | |
| 7 | M1 | Gradients by **complex step**, checked against forward FD at relative steps 1e-4 and 1e-7. | `m1_gradcheck.csv` | |
| 8 | M1 | SLSQP failed on IDF (line-search stalls); switched IDF to **trust-constr** with a consistent starting mass. MDF stays on SLSQP. | MDF/IDF agree to ~1e-9 | |
| 9 | M1 | Payload multiplier via KKT least squares plus the envelope theorem, **cross-checked** by re-optimising at +100 g. | `kkt`, `reopt_km_lost_per_100g` | |
| 10 | M1 | Catalogue snap: whole-inch props, 1–3 identical packs in parallel, lightest feasible tube, smallest motor with rating ≥ Tmax. M4 uses the **snapped** (buildable) designs. Note: snapped motor mass uses the formula at actual Tmax, not the catalogue rating. | `m1_designs.csv` | |
| 11 | M2 | Plate loads (thrust, battery weight) are **out of plane**, so the AI used a Mindlin plate-bending element rather than plane stress. Plane stress is kept only for the MBB benchmark. | | |
| 12 | M2 | The first plate element (full shear integration) **shear-locked**: −19% to −80% error vs Timoshenko. Fixed with 1-point shear + 2% full-shear stabilisation (removes 2 hourglass modes). | `m2_summary.json` → `plate_element_check` | |
| 13 | M2 | Corner arm-clamp patches and the central battery patch held **solid** (passive). Filter radius 6 mm for the minimum member width. Pick = lightest vf whose **thresholded** (waterjet) design stays ≤ 2.5× solid compliance. | `m2_plate_sweep.csv` | |
| 14 | M2 | Plate load = 2 × battery weight (6.1 kg) for the 2g max-thrust case. Compliance scales linearly, so the vf choice does not depend on this number. | | |
| 15 | M3 | Request 1 = 13-point maximin LHS + the ROI centre point sent twice (to measure noise). Requests 2–3 adaptive: kriging std × ROI weight × gradient magnitude (finds steep zones), with "kriging believer" spacing. Request 3 also seeds the M1 design hover points. | `oracle/request_*.csv` | |
| 16 | M3 | Non-smooth region handled by a **disk-loading trust limit**: fit only the smooth regime and check that M1 designs stay below the limit. (Mock oracle until real replies arrive.) | `m3_summary.json` | |
| 17 | M4 | Range feasibility uses the **90th-percentile flyable-hour wind** as a pure headwind. Emergency feasibility: outbound leg into that headwind ≤ 35 min (45 − 3 − 1 − 6 min queue allowance). | `m4_network.py` | |
| 18 | M4 | Capacity at ≤70% utilisation enforced **for every hour** (peak-hour, not daily average), with demand inflated by 1/(1 − grounded share). Bays likewise. Outage: 3 h × 2.7 kW + one charged spare per drone. | | |
| 19 | M4 | x_ij ≤ y_j disaggregated (no big-M). Big-M only on n_j, b_j, set to the smallest provably valid value. | `m4_verification.json` | |
| 20 | M4 | Battery wear priced per Wh of throughput (22 ₹/Wh ÷ (500 × 0.8)). Spare pack: capex over a 3-yr calendar life. Grounded-hour emergencies go by road; 50% of grounded routine/lab orders go by road. | | |
| 21 | M4 | Added a stakeholder variant: **H0 (blood bank) forced open**. | `m4_cost_vs_vehicle.csv` | |
| 22 | M5 | Charging on a 1-min tick with plug power shared (genset 3 kW cap during outages). Headwind from the hourly wind direction projected on the hub→PHC bearing. Routine/lab go by road after 6 h of waiting. | | |
| 23 | M5 | M/M/c check: 3 cases, 40 reps × 4000 h, independent seeds per case (an earlier version reused seeds and showed a correlated −6% error). | `m5_mmc_verification.csv` | |
| 24 | M5 | Days for CI: sequential, add 12 random days until the 95% CI half-width on S45 < 3 pp (ratio estimator, days as batches). | `m5_ci_days.csv` | |
| 25 | M6 | NSGA-II with CRN (36 days, 3 per month, fixed seeds). Genome = vehicle, Δdrones, Δbays, spares per hub slot, reserve. M4 solutions seeded into the initial population. | | |
| 26 | M6 | Weighted-sum GA normalised on the union of both fronts (2-objective front was too narrow). NumPy-2 `cross` bug fixed. | | |
| 27 | M7 | Decision rule: lowest cost among candidates with typical-week S45 ≥ 90% **and** P(S45 ≥ 90%) ≥ 0.8; otherwise minimax regret across wind weeks. | `m7_summary.json` | |

## Oracle request log (fill in when sending)

| Request | Date sent | Points | Why these points | What you expected to learn | What you actually learned |
|---|---|---|---|---|---|
| 1 | | 15 | space-filling LHS + replicate | global trend, noise level, any non-smooth zone | |
| 2 | | 8 | adaptive (std × ROI × gradient) | | |
| 3 | | 7 | adaptive + M1 design hover points | | |

## Predict-then-run (rule 6): write your prediction BEFORE opening the result file

| Change | Your prediction (direction, rough size, why) | Result | Were you right? Why / why not |
|---|---|---|---|
| Battery specific energy +20% | | `m7_sensitivity_network.csv` | |
| Drone price −30% | | | |
| Demand +15%/yr, year 3 | | `m7_sensitivity_chosen_system.csv` | |
| Payload +100 g | | `m1_summary.json` | |
