# SkyLink (AkashBhandan) — Formulation Cards (Modules M0 to M7)
**Course:** ED6002 — Engineering Design Optimization | IIT Madras  
**Authors:** Yash Shajil & Priyanka  
**Date:** November 25, 2026  

---

## Module 0: Problem Context & Data Preprocessing
- **Domain:** Kaveripuram district ($60\text{ km} \times 45\text{ km}$), 1 District Hospital ($H_0$), 6 Sub-District Hospitals ($H_1$–$H_6$), 24 PHCs ($P_{01}$–$P_{24}$).
- **Candidate Hubs:** 8 candidate locations ($H_0$–$H_6$ and $P_{09}$ with spare land).
- **Demand Classes:** Emergency ($2.5\text{ kg}$, $<45\text{ min}$ deadline), Routine ($2.5\text{ kg}$, same day), Lab samples (reverse flow).
- **No-Fly Zones:** $8\text{ km}$ circle around civil airstrip + reserved forest polygon. Formulated via 64-gon circumscription and tangent visibility graph (shortest detour paths, maximum detour factor $2.04$).
- **Environmental Stochastics:** Monsoons/gusts grounding flights above threshold; scheduled 3-hour grid outages (MWF 14:00–17:00, $2.7\text{ kW}$ generator cap).

---

## Module 1: Vehicle Multidisciplinary Design Optimization (MDO)

### 1. Variables
| Variable | Description | Units | Bounds | Type |
| :---: | :--- | :---: | :---: | :---: |
| $D$ | Propeller / Rotor diameter | $\text{m}$ | $[0.35, 0.80]$ | Continuous |
| $E$ | Battery pack total energy | $\text{Wh}$ | $[200.0, 2000.0]$ | Continuous |
| $d$ | Arm tube outer diameter | $\text{m}$ | $[0.015, 0.040]$ | Continuous |
| $t$ | Arm tube wall thickness | $\text{m}$ | $[0.0005, 0.0025]$ | Continuous |
| $V$ | Cruise forward speed | $\text{m/s}$ | $[8.0, 22.0]$ | Continuous |
| $m$ | Coupled takeoff mass (IDF coupling) | $\text{kg}$ | $[4.0, 30.0]$ | Continuous |

### 2. Objective
$$\min_{\mathbf{x}} \quad \text{MTOW}(\mathbf{x})$$
Subject to required operational radius $R(\mathbf{x}) \ge R_{\text{req}}$ ($22\text{ km}$ target mission, $15\text{ km}$ course baseline) and full $2.5\text{ kg}$ payload.  
*(Alternative unconstrained $\max R$ was rejected because it pushes all variables $E, D, d, V$ to their upper bounds, resulting in an impractical $20.1\text{ kg}$ vehicle).*

### 3. Constraints & Defenses (Why Constraint, Not Objective)
1. **Mission Range ($R(\mathbf{x}) \ge R_{\text{req}}$):** Hard operational mandate. A drone unable to reach the most distant assigned PHC with 60 s hover reserve leaves health centers without emergency blood/antivenom.
2. **Arm Bending Stress ($\sigma_{\text{max}} \le \sigma_{\text{allow}} = 200\text{ MPa}$):** Physical survival under $2g$ thrust maneuver ($T_{\text{max}} = 2mg/4$). Yielding or structural fracture causes catastrophic in-flight loss.
3. **Tip Deflection ($\delta_{\text{tip}} \le 0.01 L$):** Aerodynamic blade-tip clearance and control authority. Excessive arm bending tilts rotor thrust vectors and degrades attitude stabilization.
4. **Arm First Natural Frequency ($f_1 \ge 1.15 f_{\text{rot}}$):** Resonance avoidance. If arm fundamental bending frequency couples with 1/rev rotor harmonics ($f_{\text{rot}} = V_{\text{tip}} / \pi D$), aeroelastic flutter destroys the arm.
5. **Tube Slenderness ($d / t \le 40$):** Local thin-walled shell buckling prevention under peak cantilever bending moments.

---

## Module 2: Center Hub Plate Topology Optimization

### 1. Variables
- Element artificial density field $\rho_e \in [0, 1]$ over a $64 \times 64$ mesh of Mindlin-Reissner plate bending finite elements ($2.5\text{ mm}$ grid).
- Thresholded binary density $\widetilde{\rho}_e \in \{0, 1\}$ for physical CNC/waterjet manufacturing.

### 2. Objective
$$\min_{\boldsymbol{\rho}} \quad c(\boldsymbol{\rho}) = \mathbf{u}^T \mathbf{K}(\boldsymbol{\rho}) \mathbf{u} \quad (\text{Compliance / Strain Energy})$$
Subject to volume fraction constraint $V(\boldsymbol{\rho}) \le V_{\text{target}}$ (swept from $0.20$ to $0.50$).

### 3. Constraints & Defenses
1. **Volume Fraction ($V \le V_{\text{target}}$):** Aircraft empty weight budget. Minimizing compliance without volume bounds yields a solid block ($V=1.0$), defeating the lightweighting goal.
2. **Filter Radius ($r_{\text{min}} = 6\text{ mm}$):** Minimum member width constraint. Prevents mesh-dependent checkerboarding and guarantees that all arms and cutouts can be cut with standard waterjet toolpaths.
3. **Stiffness Retention ($c_{\text{opt}} \le 2.5 c_{\text{solid}}$):** Maximum allowable compliance limit to prevent battery and arm mounting compliance from destabilizing the flight controller.

---

## Module 3: Rotor Aerodynamic Surrogate Modeling

### 1. Variables
- Rotor diameter $D \in [0.35, 0.80]\text{ m}$ (continuous).
- Thrust per rotor $T \in [8.0, 40.0]\text{ N}$ (continuous).

### 2. Objective
$$\min \quad \text{LOOCV RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \text{FM}_i - \widehat{\text{FM}}_{-i}(D_i, T_i) \right)^2}$$

### 3. Constraints & Defenses
1. **Total Oracle Budget ($\sum N_k \le 30$ in $\le 3$ submissions):** Hard computational cost ceiling set by the oracle harness.
2. **Physical Evaluation Box ($0.35 \le D \le 0.80$, $8 \le T \le 40$):** Oracle hardware/BEMT solver validity limits.
3. **Disk Loading Cutoff ($DL \le 215\text{ N/m}^2$):** Aerodynamic stall boundary. Operating above this boundary induces massive flow separation where smooth surrogates breakdown.

---

## Module 4: Hub Network Location & Fleet Sizing (MILP)

### 1. Variables
| Variable | Description | Units | Bounds / Domain | Type |
| :---: | :--- | :---: | :---: | :---: |
| $y_j$ | Open candidate hub site $j$ | — | $\{0, 1\}$ | Binary |
| $x_{ij}$ | Assign PHC $i$ to hub $j$ | — | $\{0, 1\}$ | Binary |
| $n_j$ | Number of operational drones at hub $j$ | drones | $\mathbb{Z}_{\ge 0}$ | Integer |
| $b_j$ | Number of charging bays at hub $j$ | bays | $\mathbb{Z}_{\ge 0}$ | Integer |
| $s_j$ | Spare battery packs allocated to hub $j$ | packs | $\mathbb{Z}_{\ge 0}$ | Integer |

### 2. Objective
$$\min \quad \text{Annual Cost} = \sum_j \left( C_{\text{hub}} y_j + C_{\text{drone}} n_j + C_{\text{bay}} b_j + C_{\text{op}} O_j + C_{\text{batt}} s_j \right) + C_{\text{energy}} + C_{\text{road}}$$

### 3. Constraints & Defenses
1. **Single Hub Assignment ($\sum_j x_{ij} = 1$):** Clear chain of custody and accountability. A primary health center must have an unambiguous dispatch origin.
2. **Headwind Flight Reachability ($d_{ij}^{\text{NFZ}} \le R_{\text{usable}}(V_{\text{wind}}^{90})$):** Range feasibility under the 90th percentile headwind. Prevents running out of battery before reaching the delivery site.
3. **Emergency Outbound Flight Time ($t_{\text{outbound}} \le 35\text{ min}$):** Hard deadline constraint. Guarantees that combined processing (3 min), takeoff/landing (1 min), queue delay (6 min), and flight time remain $< 45\text{ min}$.
4. **Peak-Hour Utilization ($\rho_j \le 0.70$):** Queue stability under stochastic Poisson arrivals. Above 70% utilization, M/M/c queueing delay grows asymptotically.
5. **Power Outage Battery Reserve ($b_j P_{\text{bay}} \le 2.7\text{ kW}$ or $s_j \ge n_j$):** Continuity of operations during 3-hour grid outages on MWF.

---

## Module 5: Stochastic Discrete-Event Simulation

### 1. Entities & Random Processes
- **Arrivals:** Non-homogeneous Poisson process across 24 hours per demand class (Emergency, Routine, Lab).
- **Wind Vector:** Real hourly wind speeds and directional vectors projected onto hub-to-PHC flight bearings.
- **Outage Events:** Scheduled deterministic outages (MWF 14:00–17:00) + stochastic wind gusts grounding flights $> 12\text{ m/s}$.

### 2. Objectives (Performance Trackers)
- Emergency Service Level $S_{45} = \frac{\text{Emergencies Delivered in } \le 45\text{ min}}{\text{Total Emergency Orders}}$.
- Total annual operational expenditure (drone wear, electricity, road fallback penalties).

### 3. Constraints (Operational Policy)
1. **Emergency Dispatch Preemption:** Emergency medical orders jump to the head of the dispatch queue, reserving $20\%$ of idle drones.
2. **Road Fallback Timeout:** Orders waiting $> 30\text{ min}$ (emergency) or $> 6\text{ h}$ (routine) automatically transfer to ground ambulance.

---

## Module 6: Multi-Objective Optimization (NSGA-II)

### 1. Variables (Decision Vector)
- Vehicle choice index $k \in \{V_1, V_2, \dots, V_5\}$.
- Incremental drone allocation $\Delta n_j \in [0, 3]$.
- Incremental bay allocation $\Delta b_j \in [0, 2]$.
- Emergency drone reserve fraction $\alpha_{\text{res}} \in [0.10, 0.40]$.

### 2. Dual Objectives
$$\min \quad f_1 = \text{Annualized Total Cost of Ownership (₹/year)}$$
$$\max \quad f_2 = \text{Emergency Service Level } S_{45} \quad (\%)$$

### 3. Constraints
1. **Pareto Feasibility:** $S_{45} \ge 85\%$ minimum floor for any deployable public health system.
2. **Common Random Numbers (CRN):** Synchronized stochastic seeds across candidate evaluations to ensure true Pareto dominance rather than lucky weather draws.

---

## Module 7: Multi-Criteria Decision & Robustness Evaluation

### 1. Decision Criteria
- Select one definitive system configuration to recommend to the District Health Officer (DHO).
- Decision Rule: **Minimax Regret** across 52 stochastic weather weeks and demand surges.

### 2. Robustness Checks & Sensitivities
1. **Battery Degradation (+20% battery cost, -10% capacity):** Ensures fleet remains viable across 3-year battery life.
2. **Demand Growth (+15%/year):** Validates capacity headroom for year 3 operations without requiring new hub land acquisitions.
3. **Outage Day Shift (+1 day):** Proves policy is insensitive to changes in utility maintenance schedules.
