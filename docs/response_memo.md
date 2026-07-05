# Reviewer Response Memo

**To**: The Editorial Board and Reviewers
**From**: Ishan Petkar
**Subject**: Resubmission of "H2S-ISRO: Lunar Polar Ice Detection and Traverse Planning"

We sincerely thank the reviewers for their rigorous, constructive feedback. The critiques regarding our parameter justification, formalism inflation, lack of physical grounding, and over-claimed novelty were entirely valid. In response, we have executed a comprehensive, 8-phase remediation roadmap that fundamentally strengthens the manuscript and the accompanying codebase.

Below is a point-by-point summary of our revisions:

### 1. Physical Validation and Grounding (Critiques: "Toy models," "Unrealistic simulations")
*   **Synthetic Stress Test**: We implemented a rigorous synthetic polarimetric scene generator (`src/synthetic_stress_test.py`) that explicitly models 1-look speckle, 4-look multilooking, and 7x7 Lee filtering. We swept the roughness veto threshold across this noisy scene to empirically derive our $2.5\sigma$ limit (peak $F_1$ score: 0.57).
*   **Real-World Reference Statistics**: We extracted actual dry-regolith baseline statistics ($\mu$, $\sigma$, and correlation $r_{\mathcal{R}}$) from public LRO Mini-RF S-band mosaics. This allowed us to calculate the true joint tail probability of a false-positive under realistic, negatively correlated roughness conditions ($1.10 \times 10^{-3}$).
*   **Real LOLA DEM and Orbital Constraints**: We replaced the synthetic mesh in the A* planner with a real $5$\,m/px LOLA GeoTIFF of the Shackleton rim. Furthermore, we propagated actual Chandrayaan-2 TLEs via SGP4 to derive a physical relay communication bound of $1.63$ hours, proving our planner works on real lunar topography and orbital mechanics.

### 2. Parameter Justification (Critique: "Arbitrary thresholds")
*   **Parameter Justification Appendix**: We added a dedicated appendix (Table III) explicitly enumerating every parameter in our architecture (e.g., $2\sigma$ anomaly thresholds, DBSCAN radii, cost-map weights). 
*   **Explicit Labeling**: Within the text, we have clearly delineated which parameters are **untuned priors** (derived from mathematical theory or physical bounds) and which are **ablated parameters** (derived empirically from our synthetic sweeps). 

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

### 6. Reference Integrity
*   We audited all citations, correcting the publication venues, DOIs, and titles for Fassett et al. (2018), Kumar et al. (2022), and Raney et al. (2012), and added missing citations for the dielectric retrieval methodology.

We believe these sweeping changes have transformed the manuscript from a conceptual proposal into a physically grounded, reproducible, and robust engineering framework. We respectfully submit the revised manuscript and codebase for your consideration.
