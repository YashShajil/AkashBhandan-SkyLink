# SkyLink: Recommendation for Drone Delivery of Emergency Medical Supplies

**To:** District Health Officer, Kaveripuram  
**From:** ED6002 Project SkyLink Optimization Team  
**Date:** November 2026  

---

## 1. Executive Summary & Recommendation in One Paragraph

Deploy a high-performance network of **four drone hubs** located at SDH Vadakarai (H2), SDH Melmalai (H3), SDH Kadalpuram (H4), and PHC Arasampatti (H5). The district fleet operates **13–14 long-range medical delivery quadcopters** supported by 12 dedicated fast-charging bays, 17–18 hot-swappable battery packs, and a 20% dedicated emergency reserve buffer. Every one of the district's 24 Primary Health Centres (PHCs) is reachable within payload regulations (2.5 kg). On a typical day, blood units, antivenom, and critical cardiac packs reach peripheral clinics in **20–25 minutes flight time** (versus 1.5–3 hours by road across the ghat terrain). In full annual operations, the system achieves **97.6% on-time emergency delivery ($\le$ 45 min)** during typical weeks and **95.5% annual average reliability** at an affordable annual cost of **₹2.02–2.05 crore/year (₹20.2–20.5 M INR)**, including 24×7 technical operators, equipment amortization, electricity, battery degradation, and backup road ambulances during adverse monsoon storms.

---

## 2. District Service Outcomes & Weather Limits

| Performance Metric | Calm Week | Typical Week | Worst Monsoon Week (July) | Annual Average |
| :--- | :---: | :---: | :---: | :---: |
| **Emergencies Delivered $\le$ 45 min ($S_{45}$)** | **100.0%** | **97.6%** | **54.5%** | **95.5%** |
| **95th Percentile Emergency Turnaround** | **22.7 min** | **24.6 min** | **131.2 min** (Road failover) | **34.2 min** |
| **Hours Grounded by High Winds / Gusts** | 0 h | 4 h | 67 h (40% of daytime) | 5.4% of all hours |
| **Orders Diverted to Road Ambulance** | < 1% | 1.8% | ~45% | **2.4%** |
| **Annualized Operating Budget** | ₹18.8 M / yr | ₹19.8 M / yr | ₹43.2 M / yr (road surged) | **₹20.2–20.5 M / yr** |

### Key Operational Insights for the DHO:
1. **Exceptional On-Time Delivery (97.6% in Typical Weeks):**  
   Under normal weather conditions, the system easily surpasses the district's 90% SLA target, delivering 97.6% of emergency medical supplies within 45 minutes, with an average turnaround under 25 minutes.
2. **Weather is the Fundamental Limit, Not Fleet Size:**  
   Service interruptions are strictly governed by aerodynamic gust limits (grounded when wind gusts exceed safety margins), which ground flights in 5.4% of annual hours and up to 40% of daytime hours during peak monsoon (late July). Adding drones beyond 14 does not improve monsoon on-time rates because grounded aircraft cannot take off.
3. **Road Ambulance Failover is Essential:**  
   The drone system and the 108 road ambulance fleet are complementary. Any emergency order that waits more than 30 minutes due to weather grounding automatically shifts to an expedited road dispatch.
4. **Reliability & Queueing Stability:**  
   Across 300 Monte Carlo stress-test scenarios varying patient demand by $\pm 25\%$ and drone hardware availability between 90% and 98%, the system maintained stable queueing behavior with average fleet utilization at 31.5%, providing ample surge margin for epidemic outbreaks.

---

## 3. Financial Breakdown & Cost Drivers

The optimized system costs **₹20.20 to ₹20.49 million/year (₹2.02 to ₹2.05 crore/yr)**.

```
Annual Budget Allocation:
├── Personnel & Hub Operations (42%): 24×7 pilot/operator staffing across 3 shifts + site rent & gensets
├── Aircraft Capital Amortization (28%): 13-14 drones @ ₹7.77 lakh each, amortized over 3-year lifespan
├── Battery Cycle Degradation & Spares (15%): Wh-throughput wear based on true low-Re rotor surrogate
├── Bad-Weather Road Ambulance Contracts (10%): Guaranteed contractor retainers for road backup
└── Grid Electricity & Charging Infrastructure (5%): Hub chargers and diesel backup during grid outages
```

---

## 4. Strategic Policy Options for District Leadership

1. **Option A: Four-Hub vs. Five-Hub Configuration (Recommended):**  
   - **4 Hubs (H2, H3, H4, H5):** Optimal affordability (**₹2.02–2.05 crore/yr**, 13–14 drones) delivering 95.5% on-time emergency delivery.
   - **5 Hubs (opening H0 Central Blood Bank):** Stations drones directly at the Kaveripuram District Hospital Blood Bank. Cost increases modestly to **₹2.43–2.48 crore/yr** (18–20 drones), eliminating road transfer shuttles for rare blood units.
2. **Option B: Growth Planning (Years 1 to 3 Demand Growth):**  
   If district healthcare demand grows at 15% per year (+52% cumulative demand in Year 3):
   - The *current 14-drone system* maintains high reliability ($S_{45} = 96.7\%$, utilization rises safely from 31.5% to 44.3%), with annual cost reaching ₹25.4 M due to battery wear.
   - Re-optimizing in Year 3 expands the network to **27 drones across 6 hubs** (opening H0, H1, H2, H3, H5, H6) at ₹3.34 crore/year.
3. **Option C: Technology Dividends (Cheaper Drones / Advanced Batteries):**  
   - A **30% reduction in COTS drone prices** saves **₹19.4 lakh/year (-9.6%)**, reducing total cost to **₹1.83 crore/yr**.
   - Next-generation solid-state batteries (**+20% specific energy**, 150 $\to$ 180 Wh/kg) reduce drone MTOW from 11.8 kg to 10.4 kg, saving **₹8.35 lakh/year**.

---

## 5. Technical Rigor & Engineering Validation

This recommendation is grounded in an end-to-end Multidisciplinary Design Optimization (MDO) pipeline:
- **Aerodynamic Fidelity ($M_3$):** Calibrated against 30 physical oracle evaluations using thin-plate spline and Kriging surrogates, capturing non-linear low-thrust profile drag and stall limits.
- **Topology Optimization ($M_2$):** Lightened carbon-fiber central chassis plates (102 g waterjet-cut) validated against Mindlin plate shear-locking benchmarks.
- **Multi-Objective Pareto Optimization ($M_6$):** 3-objective NSGA-II optimization across Cost, $S_{45}$, and Road Fallback evaluated across 48 Common Random Number days.
- **Decision Rule ($M_7$):** Selected via a formal Minimax Regret criterion across Calm, Typical, and Worst Monsoon meteorological records.
