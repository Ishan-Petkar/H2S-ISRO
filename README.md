# BAH 2026 — Lunar Ice Detection & Rover Traverse Planning
## Reproducibility Bundle

---

## Repository Structure

```
bah2026-ice-detection/
├── src/
│   ├── preprocessing/
│   │   ├── calibrate.py          # Radiometric + polarimetric calibration
│   │   ├── geocode.py            # Co-registration to South Polar Stereographic
│   │   └── speckle_filter.py     # Refined-Lee filter implementation
│   ├── features/
│   │   ├── cpr.py                # CPR computation (L + S band)
│   │   ├── dop.py                # Degree of polarisation
│   │   └── mchi.py               # m-chi decomposition
│   ├── detection/
│   │   ├── reference_pop.py      # Reference population selection
│   │   ├── anomaly_score.py      # Stage-2 z-score computation
│   │   ├── roughness_veto.py     # Stage-3 DEM-roughness veto
│   │   ├── dbscan_filter.py      # Stage-4 spatial coherence
│   │   └── dual_freq_corr.py     # Stage-5 S/L-band corroboration
│   ├── landing_site/
│   │   ├── illumination.py       # Horizon ray-trace over LOLA DEM
│   │   └── site_optimizer.py     # Constrained landing-site search
│   ├── traverse/
│   │   ├── energy_budget.py      # Energy ledger + feasibility check
│   │   ├── relay_window.py       # Orbital relay visibility computation
│   │   └── astar_planner.py      # A* on cost map with hard constraints
│   └── volume/
│       ├── ice_fraction.py       # Empirical CPR-to-ice-fraction mapping
│       └── monte_carlo.py        # MC uncertainty propagation (n=10000)
├── data/
│   ├── download_dfsar.sh         # PRADAN portal download script placeholder
│   ├── download_lola.sh          # PDS Geosciences Node DEM download
│   └── README_data.md            # Dataset descriptions and licences
├── experiments/
│   ├── run_full_pipeline.sh      # End-to-end experiment script
│   ├── ablation_dbscan.py        # DBSCAN parameter ablation
│   ├── ablation_roughness.py     # Roughness veto threshold sweep
│   └── baseline_comparison.py   # Global threshold vs proposed
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_detection_stages.ipynb
│   └── 04_traverse_planning.ipynb
├── results/                      # Auto-generated; gitignored
├── paper/
│   ├── main.tex
│   └── references.bib
├── LICENSE                       # MIT
└── README.md                     # This file
```

---

## Dependencies

```
python>=3.10
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3      # DBSCAN
gdal>=3.7              # Geospatial raster I/O
rasterio>=1.3
shapely>=2.0
matplotlib>=3.7
tqdm>=4.65
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Hardware

- CPU: 8+ cores recommended (parallelised Monte Carlo)
- RAM: 32 GB for full south polar scene
- GPU: Not required
- Disk: ~50 GB for full DFSAR + LOLA datasets
- Estimated runtime: < 6 hours end-to-end

---

## Reproducing Experiments

### Step 1 — Download data
```bash
bash data/download_lola.sh          # ~8 GB, PDS Geosciences Node
bash data/download_dfsar.sh         # ~20 GB, ISRO PRADAN portal
                                    # (manual registration required)
```

### Step 2 — Run full pipeline
```bash
python src/preprocessing/calibrate.py --scene SC001_L2 --seed 42
python src/preprocessing/geocode.py   --crs EPSG:32761
python src/features/cpr.py            --band L S
python src/detection/anomaly_score.py --sigma 2.0
python src/detection/dbscan_filter.py --eps 30 --minpts 4
python src/traverse/astar_planner.py  --energy-margin 0.20
python src/volume/monte_carlo.py      --n-samples 10000 --seed 42
```

Or run the full pipeline script:
```bash
bash experiments/run_full_pipeline.sh --seed 42
```

### Step 3 — Ablation studies
```bash
python experiments/ablation_dbscan.py
python experiments/ablation_roughness.py
```

---

## Random Seed

All stochastic components (Monte Carlo, DBSCAN initialisation) use
`seed=42` by default. Pass `--seed <int>` to override.

---

## Dataset Licences

| Dataset | Licence | Source |
|---|---|---|
| Chandrayaan-2 DFSAR | ISRO Open Data Policy | pradan.issdc.gov.in |
| LOLA DEM | NASA PDS Open Access | pds-geosciences.wustl.edu |
| LRO Mini-RF CPR | NASA PDS Open Access | pds-geosciences.wustl.edu |

---

## Humanizing GitHub Resources Applied

The paper writing style was informed by:
- **github.com/humanize/humanize** — Python text humanization utilities
  (number formatting, time deltas); concepts of making numbers
  human-readable extended to making uncertainty ranges human-readable
  in the paper prose.
- **Academic writing style principles** (narrative motivation, varied
  sentence length, explicit signposting, empathetic limitation framing)
  documented in open academic-writing guides.
