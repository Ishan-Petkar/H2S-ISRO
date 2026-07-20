# Data Dictionary — H2S-ISRO Project

| Variable | Symbol | Type | Unit | Source | Provenance |
|---|---|---|---|---|---|
| L-band CPR | CPR_L | float32 | dimensionless | DFSAR L2 | PRADAN (A8) |
| S-band CPR | CPR_S | float32 | dimensionless | DFSAR L2 | PRADAN (A8) |
| Degree of polarisation | DOP | float32 | dimensionless | DFSAR L2 Stokes | PRADAN (A8) |
| LOLA elevation | h(p) | float32 | metres | PDS LOLA | A10 |
| LOLA slope | s(p) | float32 | degrees | Computed from h(p) | src/planner_execution.py |
| DEM roughness | ρ(p) | float32 | metres RMS | Computed from h(p) | src/planner_execution.py |
| CPR reference mean (sunlit) | μ_R^CPR | float64 | dimensionless | **ACTION REQUIRED** A9 | Compute from Mini-RF mosaic |
| DOP reference mean | μ_R^DOP | float64 | dimensionless | **ACTION REQUIRED** | Cannot come from Mini-RF; see Phase 1 Task 1.3 |
| Ice fraction | f_ice | float32 | dimensionless [0,1] | Eq. 13, coefficients a,b from Sahu 2024 | **Must verify a,b values from A13** |
| Traction coefficient | k | float64 | dimensionless | Prior; sensitivity 0.03–0.10 | Stated in text |
| Flat-terrain locomotion cost | e_0 | float64 | Wh/m | VIPER-class reference | **Verify exact NASA source** |
| Relay window | T_relay | float64 | hours | Proxy orbit simulation | planner_execution.py |
