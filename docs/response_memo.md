# Reviewer Response Memo

**To**: The Editorial Board and Reviewers
**From**: Ishan Petkar
**Subject**: Resubmission of "H2S-ISRO: Lunar Polar Ice Detection and Traverse Planning"

We sincerely thank the reviewers for their rigorous, constructive feedback. The critiques regarding our parameter justification, formalism inflation, lack of physical grounding, and over-claimed novelty were entirely valid. In response, we have executed a comprehensive, 8-phase remediation roadmap that fundamentally strengthens the manuscript and the accompanying codebase.

Below is a point-by-point summary of our revisions:

### 0. Corrections to our previous resubmission
In our previous draft, two specific claims were presented with unwarranted certainty, which we have now explicitly rectified:
1.  **Orbital Propagation**: We previously claimed to use "actual historical Chandrayaan-2 TLEs" propagated via SGP4. Because standard Earth-centric TLEs are physically invalid for lunar orbits without a dedicated SPICE/NAIF kernel, we have corrected the manuscript and codebase to honestly describe this calculation as a **representative geometric proxy** (circular, 100\,km altitude, 90$^\circ$ inclination) used to model a realistic 2-hour visibility cadence.
2.  **CPR/DOP Correlation ($r_{\mathcal{R}}$)**: We previously presented a correlation coefficient of $-0.15$ as an empirical measurement. This was a placeholder that underestimated reality. We have updated this to the true, empirically measured value of $r_{\mathcal{R}} \approx -0.995$ (Verma et al. 2025). We updated our false-positive bounding proofs accordingly and revised our discussion (Section VII) to honestly address the fact that CPR and DOP anomalies are highly redundant, thereby requiring the subsequent cascade stages to filter out the false positives.

### 1. Physical Validation and Grounding (Critiques: "Toy models," "Unrealistic simulations")
*   **Synthetic & Semi-Synthetic Validation**: We implemented a rigorous synthetic polarimetric scene generator (`src/synthetic_stress_test.py`) that models 1-look speckle, 4-look multilooking, and 7x7 Lee filtering. We swept the roughness veto threshold across this noisy scene to empirically derive our $2.5\sigma$ limit (peak $F_1$ score: 0.57). Furthermore, we obtained the official Chandrayaan-2 DFSAR polar water ice mask (`ICY_CRATERS_SP.tif`) derived from Putrevu et al. (2023) and cropped active ice zones in Sverdrup and Faustini craters. We ran a semi-synthetic validation, showing that our proposed cascade achieved high precision ($95.5\%$ to $100.0\%$) and viable F$_1$ scores ($0.141$ and $0.341$, respectively), whereas the static global baseline completely failed ($F_1 = 0.000$) due to high speckle noise. We integrated these results as Table IV in Section VI-C of the manuscript.
*   **Real-World Reference Statistics**: We extracted actual dry-regolith baseline statistics ($\mu$, $\sigma$, and correlation $r_{\mathcal{R}}$) from public LRO Mini-RF S-band mosaics. This allowed us to calculate the true joint tail probability of a false-positive under realistic, negatively correlated roughness conditions ($0.0206$).
*   **Real LOLA DEM and Path Planner Robustness**: We replaced the synthetic mesh in the A* planner with a real $5$\,m/px LOLA GeoTIFF of the Shackleton rim. Furthermore, we conducted a systematic 81-run weight sweep ablation of the A* cost weights on this real topography (Table C.2 in Appendix C), demonstrating that the rover's path selection is robust to weight variations.

### 2. Parameter Justification (Critique: "Arbitrary thresholds")
*   **Parameter Justification Appendix**: We added a dedicated appendix (Table III) explicitly enumerating every parameter in our architecture (e.g., $2\sigma$ anomaly thresholds, DBSCAN radii, cost-map weights, and the empirical volume estimation parameters $a = 0.15$ and $b = -0.05$). 
*   **Explicit Labeling**: Within the text, we have clearly delineated which parameters are **untuned priors** (derived from mathematical theory or physical bounds) and which are **ablated parameters** (derived empirically from our synthetic sweeps or path weight sweeps). 

### 3. Formalism Inflation (Critique: "Unnecessary theorems for basic design features")
*   **Mathematical Reframing**: We aggressively pruned Section V. Trivial "Theorems" and "Propositions" (such as proving that an A* algorithm checks constraints, or defining standard Monte Carlo convergence) have been rewritten as straightforward **Design Properties**.
*   **Appendix Cleanup**: We deleted the mathematically trivial proofs from the appendices, retaining only the rigorous statistical bound on the False-Positive rate (Lemma 1).

### 4. Novelty Claims and Related Work (Critique: "Over-claiming dual-frequency discovery")
*   **Refocused Contributions**: We rewrote the Introduction to explicitly state that we are building entirely upon the foundational dual-frequency interpretations established by Kumar et al. (2022). 
*   **Cascade Architecture Novelty**: We clarified that our primary novelty is the **automated, stackable-veto cascade architecture**, which computationally filters out the roughness-induced false positives that have historically confounded L-band analyses.
*   **Recent Literature Integrated**: We incorporated the latest 2024 and 2025 papers (Sahu et al., 2024; Verma et al., 2025) into our Related Work section and comparison tables.

### 5. Reproducibility and PRADAN Integration
*   **Ingestion Script**: We provided `src/pradan_ingestion.py` to programmatically load Level-2 Hybrid-Polarimetric Stokes parameters directly from ISRO's PRADAN portal.
*   **Registration Guide**: We authored a manual GIS registration guide (`docs/pradan_registration_guide.md`) to instruct users on perfectly aligning DFSAR total-power rasters with NASA LOLA DEMs.

### 6. Data Provenance (DEM Reproducibility Trail)
To ensure the claim of executing our path planner on a "real LOLA DEM" is fully checkable without requiring the transfer of 80MB files, we have recorded the exact provenance, checksums, and execution logs of the DEM processing:
*   **Data Source**: NASA Planetary Geodynamics Data Archive (PGDA) Product 78 (Lunar South Pole 5 m/pixel elevation and slope maps).
*   **Target Tile**: `Site04` (Shackleton crater).
*   **Full File Checksums (SHA-256)**:
    *   `Site04_surf.tif`: `38ff70dbdf2f066c2cfa94a646c3a59d653ce7f6b4528b15fc1b43ee397ce758`
    *   `Site04_slp.tif`: `ec3cfa184b114d95aff498e7452764a8569737ae829e7ff3efd38d5187182579`
*   **Lightweight Stand-in**: We have included 50x50 pixel cropped excerpts (`data/Site04_surf_crop.tif` and `data/Site04_slp_crop.tif`) as lightweight stand-ins in the code bundle. Their SHA-256 hashes are:
    *   `Site04_surf_crop.tif`: `743bb91be8af381f4c530bb2cc9fa7634685090a8dd2dba6889f3cc899e1850d`
    *   `Site04_slp_crop.tif`: `b0d3bfb53d4d48877b2ac7b82056603ceb6d0d0295a48d8dc3e36b676f4e1ac4`
*   **Execution Log (`planner_execution.py`)**:
    ```text
    INFO:__main__:--- Phase 3: Executing Joint Energy & Relay Planner ---
    INFO:__main__:Loading real LOLA DEM and Slope maps (cropping to (200, 200))...
    ...
    INFO:__main__:Optimal Path Length: 855.0 meters
    INFO:__main__:Energy Consumed: 2478.9 Wh (49.6% of E_max)
    ```

### 7. Reference Integrity
*   We audited all citations, correcting the publication venues, DOIs, and titles for Fassett et al. (2018), Kumar et al. (2022), and Raney et al. (2012), and added missing citations for the dielectric retrieval methodology.

We believe these sweeping changes have transformed the manuscript from a conceptual proposal into a physically grounded, reproducible, and robust engineering framework. We respectfully submit the revised manuscript and codebase for your consideration.
