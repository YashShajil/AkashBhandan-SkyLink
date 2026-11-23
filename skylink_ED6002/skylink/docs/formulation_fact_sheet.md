# Formulation fact sheet (NOT a formulation card)

The brief requires formulation cards **in your own words** (rule 1: "Do not paste AI output into a formulation card").
This sheet only lists what the code implements: variables, bounds, numbers and where to find them, so you can
check your cards against the code. Write the cards (and the reasons why each constraint is a constraint and not an objective) yourself.

## M0 — what the code assumes
- Top-level objectives used downstream: annual cost (min), S45 (max), road-fallback share (min).
- Wishes converted to numeric constraints in the code: 45-min emergency target (in M4 feasibility: outbound flight ≤ 35 min into the design headwind);
  70% peak-hour utilisation; MTOW cap (M1); compliance ≤ 2.5× solid (M2); member width ≥ 6 mm; emergency → road after 30 min; routine/lab → road after 6 h.
- Decision hierarchy implemented: vehicle (M1–M3) → hubs/assignment/fleet (M4) → fleet, spares and policy tuning (M6).
- Variable types: D, E, d, t, V continuous; plate densities continuous (then thresholded); y, x binary; n, b, operators, spares integer; vehicle choice categorical; reserve fraction discretised (0.05 steps).

## M1
| Var | Unit | Bounds | Type |
|---|---|---|---|
| D | m | 0.35–0.80 | cont. |
| E | Wh | 200–2000 | cont. |
| d | m | 0.015–0.040 | cont. |
| t | m | 0.0005–0.0025 | cont. |
| V | m/s | 8–22 | cont. |
| m (IDF only) | kg | 4–30 | cont. (coupling) |

Objective in code: max R at 2.5 kg, s.t. bending, deflection, frequency, d/t ≤ 40, **MTOW ≤ cap** (12 kg baseline; caps 7/8.5/10/12/14 kg give V1–V5).
Why not plain max R: `m1_base_summary.json → uncapped_maxR` (all bounds active, 20.1 kg, 27.3 km).
Baseline (FM 0.65, plate 0.25 kg): R = 22.24 km; active = frequency, MTOW cap, D upper, d upper; **0.282 km lost per +100 g** (re-optimised: 0.283).
Final (plate 0.113 kg + FM surrogate): R = 24.83 km at 12 kg; 0.316 km/100 g.

## M2
Mindlin plate, 64×64 elements (2.5 mm), 3 mm 6061-T6 (E 68.9 GPa, ν 0.33), corner clamp patches 10 mm, battery patch 40×40 mm, filter radius 6 mm, p = 3.
Pick: vf = 0.35, 113 g, thresholded compliance 2.16× solid. vf = 0.30 → 2.63× (fails); vf = 0.20 disconnects after thresholding.

## M3
Domain D 0.35–0.80 m, T 8–40 N; 15 + 8 + 7 points. Trust limit on disk loading (mock: 140 N/m²); M1 designs reach 68 N/m².

## M4
Sets: 8 candidate hubs, 24 PHCs. Costs (annualised, CRF 10%): hub ₹6.93 L; drone (₹2 L + ₹45k/kg) × (CRF₃ + 0.15); bay ₹23.7k;
operator slot ₹15.12 L (3 shifts × 1.2 relief); spare pack 22·E·CRF₃; energy grid/genset blend; wear ₹0.055/Wh; road ₹2500/₹500.
Design wind (P90 of flyable hours) = see `m4_verification.json → wind`.

## M5 / M6 / M7
See README verification index; days for CI = `m5_ci_days.csv`; CRN day set = 36 days (3 per month), seeds 7000+.
