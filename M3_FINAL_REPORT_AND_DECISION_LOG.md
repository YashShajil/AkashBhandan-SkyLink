# SkyLink Module 3: Rotor Surrogate Modeling from Scarce Oracle Data
**Course:** ED6002 — Engineering Design Optimization | Indian Institute of Technology Madras  
**Authors:** Yash Shajil & Priyanka  
**Date of Submission:** November 25, 2026  

---

## 1. Module Overview & Objective
In the baseline quadcopter sizing module ($M_1$), the rotor aerodynamic Figure of Merit was assumed to be an invariant constant:
$$\text{FM}_0 = 0.6500$$
In realistic low-Reynolds rotor aerodynamics, the true Figure of Merit $\text{FM}(D, T)$ depends strongly on rotor diameter $D$ (scale/tip speed/chord Reynolds number) and thrust per rotor $T$ (collective pitch, blade angle of attack $\alpha$, and induced velocity).

**The Objective of $M_3$:**  
Replace the constant $\text{FM}_0 = 0.65$ with a data-driven surrogate model $\widehat{\text{FM}}(D, T)$ constructed under a strict budget constraint:
$$\text{Total Evaluations } N \le 30 \quad \text{across at most } 3 \text{ sequential oracle queries}$$

---

## 2. Formulation Card: Module 3 (Rotor Surrogate)

| Element | Specification | Description & Mathematical Definition |
| :--- | :--- | :--- |
| **Input Variables** | $D \in [0.35, 0.80]\text{ m}$<br/>$T \in [8.0, 40.0]\text{ N}$ | Continuous rotor diameter (lower bound: ground clearance/tip speed limits; upper bound: arm structural length).<br/>Continuous hover/maneuver thrust per rotor (lower bound: empty cruise; upper bound: 2g maneuver MTOW). |
| **Output Response** | $\text{FM} \in (0, 1)$ | Rotor Figure of Merit: $\text{FM} = \frac{P_{\text{ideal}}}{P_{\text{actual}}} = \frac{T v_i}{P_{\text{shaft}}}$ where $v_i = \sqrt{\frac{T}{2\rho A}}$. |
| **Objective** | $\min \text{LOOCV RMSE}$ | Minimize the Leave-One-Out Cross-Validation error across the scarce sample points, prioritizing accuracy in the quadcopter flight region ($DL < 100\text{ N/m}^2$). |
| **Budget Constraint** | $\sum_{k=1}^3 N_k \le 30$ | Total calls to the expensive CFD/BEMT oracle cannot exceed 30 evaluations across $\le 3$ submissions. |
| **Domain Constraint** | $D_{\text{lo}} \le D \le D_{\text{hi}}$<br/>$T_{\text{lo}} \le T \le T_{\text{hi}}$ | Points outside the physical box are returned un-evaluated by the TA and forfeited. |
| **Why Budget is a Constraint** | Fixed Computational Cost | High-fidelity rotor simulations require significant CPU time; the 30-point cap mimics real-world scarce-data engineering environments. |

---

## 3. Sequential Sampling Strategy & Oracle Request Log

The 30-point evaluation budget was apportioned into three structured stages:
- **Request 1 ($N_1 = 15$ points):** Initial Space-Filling DoE & Noise Quantification.
- **Request 2 ($N_2 = 8$ points):** Exploration, Peak Efficiency Hunting & Stall Boundary Search.
- **Request 3 ($N_3 = 7$ points):** Mission Exploitation at 22 km Operating Conditions & Phenomonological Bracketing.

### Complete 30-Point Oracle Evaluation Log

| # | Req | Diameter $D$ (m) | Thrust $T$ (N) | Disk Loading $DL$ ($\text{N/m}^2$) | Evaluated $\text{FM}$ | Physical Classification & Aerodynamic Insight |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1 | 0.7574 | 17.83 | 39.6 | **0.5870** | Cruise transition regime at large diameter |
| 2 | 1 | 0.6612 | 34.47 | 100.7 | **0.6853** | High-efficiency hover plateau |
| 3 | 1 | 0.3885 | 36.79 | 310.4 | **0.2113** | **Severe Aerodynamic Stall Cliff** (Extreme flow separation) |
| 4 | 1 | 0.6974 | 11.71 | 30.6 | **0.5508** | Moderate disk loading cruise |
| 5 | 1 | 0.5926 | 13.11 | 47.5 | **0.5941** | Mid-diameter cruise condition |
| 6 | 1 | 0.4285 | 19.38 | 134.4 | **0.6711** | High disk loading attached flow |
| 7 | 1 | 0.5318 | 23.46 | 105.6 | **0.6738** | High efficiency mid-thrust zone |
| 8 | 1 | 0.4702 | 8.48 | 48.8 | **0.5931** | Low thrust boundary |
| 9 | 1 | 0.5920 | 29.88 | 108.6 | **0.6751** | Operational hover zone |
| 10 | 1 | 0.7983 | 39.66 | 79.2 | **0.6796** | High thrust large diameter hover |
| 11 | 1 | 0.4911 | 30.59 | 161.5 | **0.6967** | High attached-flow peak |
| 12 | 1 | 0.6623 | 21.72 | 63.1 | **0.6421** | Operational flight centroid |
| 13 | 1 | 0.3759 | 25.94 | 233.7 | **0.4343** | **Moderate Stall Region** (Onset of boundary layer separation) |
| 14 | 1 | 0.7874 | 26.00 | 53.4 | **0.6230** | Replicate point 1 (Vehicle hover operating point) |
| 15 | 1 | 0.7874 | 26.00 | 53.4 | **0.6282** | Replicate point 2 (Quantifies oracle noise variance: $\sigma = 0.00368$) |
| 16 | 2 | 0.5800 | 36.00 | 136.3 | **0.6924** | High-efficiency ridge confirmation |
| 17 | 2 | 0.6400 | 38.00 | 118.1 | **0.6971** | **Global Peak Efficiency Island** ($\alpha \approx 5^\circ$ peak $C_l/C_d$) |
| 18 | 2 | 0.5000 | 40.00 | 203.7 | **0.6718** | **Upper Attached Flow Limit** (Proves attached flow up to $204\text{ N/m}^2$) |
| 19 | 2 | 0.3500 | 8.00 | 83.2 | **0.6175** | Lower diameter boundary |
| 20 | 2 | 0.8000 | 8.00 | 15.9 | **0.4986** | **Low-Thrust Profile Drag Dip** (Small collective pitch penalty) |
| 21 | 2 | 0.3500 | 18.00 | 187.1 | **0.6579** | Small rotor moderate thrust |
| 22 | 2 | 0.7200 | 8.00 | 19.6 | **0.5161** | Low disk loading profile drag corroboration |
| 23 | 2 | 0.4000 | 14.00 | 111.4 | **0.6652** | Mid-range verification |
| 24 | 3 | 0.7874 | 30.68 | 63.0 | **0.6317** | **Exact 22 km Loaded Hover Operating Condition** ($T \approx W_{\text{loaded}}/4$) |
| 25 | 3 | 0.7874 | 17.50 | 35.9 | **0.5711** | **Exact 22 km Cruise Forward Flight Condition** ($T \approx W_{\text{cruise}}/4$) |
| 26 | 3 | 0.7500 | 24.00 | 54.3 | **0.6107** | Empty return flight hover state |
| 27 | 3 | 0.8000 | 13.50 | 26.9 | **0.5566** | **Profile Drag Knee** (Bridges the 8N dip to the hover plateau) |
| 28 | 3 | 0.6800 | 28.00 | 77.1 | **0.6537** | Uncertainty reduction in operational centroid |
| 29 | 3 | 0.7200 | 34.00 | 83.5 | **0.6722** | High-efficiency ridge interpolation |
| 30 | 3 | 0.4500 | 35.00 | 220.1 | **0.5289** | **Stall Onset Inflection Point** (Exact flow separation threshold) |

---

## 4. Analysis of Experimental Phenonema & Physics

### A. Oracle Noise Quantification (Replicates Analysis)
Points 14 and 15 evaluated identical coordinates: $D = 0.7874\text{ m}, T = 26.00\text{ N}$.
$$\text{FM}_1 = 0.6230, \quad \text{FM}_2 = 0.6282$$
$$\sigma_{\text{noise}} = \frac{|\text{FM}_2 - \text{FM}_1|}{\sqrt{2}} = 0.00368 \implies \text{Relative Noise } \pm 0.59\%$$
- **Finding:** The oracle exhibits slight numerical convergence jitter ($\sim 0.6\%$) characteristic of iterative blade element convergence tolerances.
- **Model Treatment:** We applied a noise nugget $\alpha = (\sigma_{\text{noise}} / \sigma_y)^2 \approx 10^{-3}$ on the covariance diagonal:
  $$\mathbf{K}_{\text{reg}} = \mathbf{K} + \alpha \mathbf{I}$$
  This regularizes the Gram matrix, prevents numerical singularities, and ensures the surrogate does not overfit numerical noise.

### B. Discovery and Mapping of the Aerodynamic Stall Cliff
The combined dataset revealed a severe, highly localized aerodynamic discontinuity:
- At $DL = 161.5\text{ N/m}^2$: $\text{FM} = 0.6967$ (healthy attached flow)
- At $DL = 203.7\text{ N/m}^2$: $\text{FM} = 0.6718$ (peak unstalled loading)
- At $DL = 220.1\text{ N/m}^2$: $\text{FM} = \mathbf{0.5289}$ (**Stall Onset**: flow separation begins)
- At $DL = 233.7\text{ N/m}^2$: $\text{FM} = \mathbf{0.4343}$ (moderate separation)
- At $DL = 310.4\text{ N/m}^2$: $\text{FM} = \mathbf{0.2113}$ (**Deep Stall Collapse**: complete boundary layer breakdown)

### C. Low-Thrust Profile Drag Dip
At large diameters ($D = 0.80\text{ m}$) under low thrust ($T = 8.0\text{ N}$), $\text{FM}$ drops to **$0.4986$**.
- **Physical Cause:** Figure of Merit is defined as $\text{FM} = \frac{T v_i}{T v_i + P_0}$. As $T \to 0$, induced power $T v_i \propto T^{3/2}$ vanishes much faster than profile drag power $P_0 \propto \Omega^3 R^5$. At small collective pitch angles, blade airfoils operate at low lift coefficients where drag dominates, reducing efficiency.

---

## 5. Multi-Model Benchmark: Leave-One-Out Cross-Validation (LOOCV)

Four distinct surrogate architectures were benchmarked across the complete 30-point dataset using **Leave-One-Out Cross-Validation (LOOCV)**:

$$e_i = y_i - \widehat{y}_{-i}(x_i), \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N e_i^2}$$

| Surrogate Architecture | Mathematical Formulation | Regularity | Global LOOCV RMSE | Max Residual Error | ROI RMSE ($DL < 100$) | Aerodynamic Evaluation & Fit Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Kriging + Nugget (Gaussian / RBF)** | $k(d) = \exp(-0.5 d^2) + \alpha \delta_{ij}$ | $C^\infty$ | **`0.03635`** | **`0.09509`** | **`0.01937`** | **Best Overall Surrogate.** With 30 points, spatial coverage eliminated earlier edge ripples. Global RMSE dropped by $73\%$. |
| **RBF (Thin-Plate Spline)** | $\phi(r) = r^2 \ln(r) + \lambda \mathbf{I}$ | $C^\infty$ | **`0.05528`** | `0.21034` | **`0.01074`** | **Outstanding Flight Zone Fit.** Bending energy minimization provides smooth curvature with $1.07\%$ operational error. |
| **Kriging + Nugget (Matérn 3/2)** | $k(d) = (1 + \sqrt{3}d)\exp(-\sqrt{3}d)$ | **$C^1$** | **`0.05786`** | `0.24673` | **`0.01199`** | **Best Discontinuity Handler.** Rougher sample paths capture the stall cliff without disturbing the unstalled plateau. |
| **Smooth-Regime Kriging ($DL < 215$)** | $C^\infty$ on 27 unstalled pts | $C^\infty$ | **`0.02022`** | **`0.04254`** | **`0.01239`** | **Direct MDO Model.** Isolates unstalled physics. $< 1.2\%$ relative error across the flight envelope. |
| **Quadratic Response Surface (QRS)** | $\beta_0 + \beta_1 D + \beta_2 T + \beta_3 D^2 + \dots$ | $C^\infty$ | `0.08966` | `0.32942` | `0.02291` | **Polynomial Stiffness Flaw.** A rigid degree-2 parabola cannot fit profile drag dip, peak ridge, and stall cliff simultaneously. |

---

## 6. M1 Quadcopter MDO Re-Run: How Far Did the Optimum Move?

We re-ran the full multidisciplinary design optimization (**MDF: minimize MTOW subject to structural margins and radius $R \ge R_{\text{req}}$**) using exact complex-step gradients, comparing baseline constant $\text{FM} = 0.65$ against the trained 30-point surrogate:

### Optimization Results Comparison Table

| Formulation & Scenario | MTOW (kg) | $\Delta\text{MTOW}$ | Rotor Diameter $D$ (m) | Battery Energy $E$ (Wh) | Cruise Speed $V$ (m/s) | Hover FM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **15 km Baseline (Constant $\text{FM}=0.65$)** | **`8.137`** | — | `0.7247` | `441.4` | `14.82` | `0.6500` |
| **15 km with Quadratic RS** | **`8.200`** | $+0.063\text{ kg } (+0.77\%)$ | `0.6437` ($-81.0\text{ mm}$) | `478.8` ($+37.4\text{ Wh}$) | `15.51` | `0.6701` |
| **15 km with 30-Point Kriging** | **`8.649`** | **`+0.513 kg (+6.30%)`** | `0.6873` ($-37.4\text{ mm}$) | `527.0` ($+85.6\text{ Wh}$) | `15.97` | `0.6261` |
| **22 km Baseline (Constant $\text{FM}=0.65$)** | **`11.805`** | — | `0.8000` (Upper Bnd) | `890.9` | `17.29` | `0.6500` |
| **22 km with Quadratic RS** | **`11.832`** | $+0.027\text{ kg } (+0.23\%)$ | `0.7114` ($-88.6\text{ mm}$) | `945.7` ($+54.8\text{ Wh}$) | `17.95` | `0.6812` |
| **22 km with 30-Point Kriging** | **`12.551`** | **`+0.745 kg (+6.31%)`** | **`0.7484` ($-51.6\text{ mm}$)** | **`1022.6` ($+131.7\text{ Wh}$)** | **`18.41`** | `0.6523` |

### Physical Justification of the Optimum Movement:
1. **Battery Capacity & MTOW Increase ($+131.7\text{ Wh}$, $+745\text{ g}$):**  
   Point 25 in Request 3 measured cruise forward flight efficiency ($T \approx 17.5\text{ N}$) to be **$\text{FM} = 0.5711$**, well below the idealized $0.6500$. Because the drone spends $>85\%$ of mission energy in cruise, power draw is higher than originally estimated. To guarantee the 22 km mission radius, the optimizer increases battery capacity from $890.9\text{ Wh}$ to **$1022.6\text{ Wh}$**, causing vehicle MTOW to rise by $+6.31\%$ to **$12.551\text{ kg}$**.
2. **Rotor Diameter Contraction ($0.8000\text{ m} \to 0.7484\text{ m}$):**  
   In the baseline, the rotor was pegged against the upper boundary ($0.80\text{ m}$) because ideal momentum theory does not penalize large rotors. With real aerodynamic data, the optimizer discovers that shrinking the rotor to $0.7484\text{ m}$ avoids the low-disk-loading profile drag penalty while shortening the motor arms and significantly reducing structural carbon fiber mass.

---

## 7. Deliverables & Artifact References
- **Master 30-Point Dataset:** `Project/oracle/final_response_30.csv`
- **Interactive 3D Dashboard:** `Project/outputs_22km/live/kriging_explorer.html`
- **4-Panel Static 3D Diagnostic Figures:** `Project/outputs_22km/figs/m3_surrogate_3d.png`
- **Summary Metrics & JSON:** `Project/outputs_22km/tables/m3_summary.json`
