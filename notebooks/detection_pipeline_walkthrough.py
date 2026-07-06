"""
Detection Pipeline Walkthrough
Run with: jupyter nbconvert --to notebook --execute this file
or convert to .ipynb with jupytext.

Sections:
  0. Setup
  1. Load and visualise raw DFSAR data
  2. Feature extraction (CPR, DOP)
  3. Reference population
  4. Stage 2 — anomaly scoring
  5. Stage 3 — roughness veto
  6. Stage 4 — DBSCAN
  7. Stage 5 — dual-frequency
  8. Final (Pi, Phi) maps
  9. Landing site
  10. Energy budget check
"""

# %% [markdown]
# ## 0. Setup

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import (
    PolarimetricScene, build_reference_population,
    run_detector, find_landing_site, estimate_ice_volume
)

rng = np.random.default_rng(42)
print("Setup complete.")

# %% [markdown]
# ## 1. Load Data
# Replace `generate_synthetic_scene()` with real DFSAR raster loading
# using rasterio once data is downloaded from PRADAN.

# %%
def generate_synthetic_scene(H=256, W=256) -> PolarimetricScene:
    """Synthetic scene for pipeline testing before real data arrives."""
    psr = np.zeros((H,W), bool)
    psr[80:180, 80:180] = True          # crater interior = PSR

    cpr_L = rng.uniform(0.3, 0.8, (H,W))
    cpr_S = rng.uniform(0.3, 0.8, (H,W))
    dop   = rng.uniform(0.4, 0.9, (H,W))

    # Inject ice patch (elevated CPR, depressed DOP)
    cpr_L[110:150, 110:150] = rng.uniform(1.2, 1.8, (40,40))
    cpr_S[110:150, 110:150] = rng.uniform(1.3, 1.9, (40,40))
    dop  [110:150, 110:150] = rng.uniform(0.1, 0.3, (40,40))

    # Inject ejecta patch (elevated CPR but also rough)
    cpr_L[90:105, 90:105]   = rng.uniform(1.1, 1.5, (15,15))
    roughness = rng.uniform(0.1, 0.5, (H,W))
    roughness[90:105, 90:105] = rng.uniform(1.5, 2.5, (15,15))  # rough ejecta

    return PolarimetricScene(
        cpr_L=cpr_L, cpr_S=cpr_S, dop=dop,
        roughness=roughness,
        slope=rng.uniform(0, 20, (H,W)),
        illumination=np.where(psr, 0.0,
                              rng.uniform(0.0, 1.0, (H,W))),
        psr_mask=psr, pixel_area_m2=225.0  # 15m x 15m
    )

scene = generate_synthetic_scene()
print(f"Scene shape: {scene.cpr_L.shape}")

# %% [markdown]
# ## 2–7. Run Detection Pipeline

# %%
ref = build_reference_population(scene)
print(f"Reference type: {ref['type']}")
print(f"CPR ref: μ={ref['mu_cpr']:.3f} σ={ref['std_cpr']:.3f}")

result = run_detector(scene, ref,
                       sigma_thresh=2.0,
                       dbscan_eps=30.0,
                       dbscan_minpts=4,
                       min_cluster_area_m2=100.0)

# %% [markdown]
# ## 8. Visualise (Pi, Phi) Maps

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(scene.cpr_L, cmap='RdYlBu_r', vmin=0, vmax=2)
axes[0].set_title('L-band CPR (raw)')
plt.colorbar(axes[0].images[0], ax=axes[0])

im1 = axes[1].imshow(result.ice_prob, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Ice Probability $\\Pi(p)$')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(result.fp_risk, cmap='Reds', vmin=0, vmax=1)
axes[2].set_title('False-Positive Risk $\\Phi(p)$')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
import os; os.makedirs("../results", exist_ok=True)
plt.savefig('../results/detection_maps.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Ice candidates: {result.ice_candidates.sum()} pixels")
print(f"Ice coverage area: "
      f"{result.ice_candidates.sum() * scene.pixel_area_m2 / 1e6:.3f} km²")

# %% [markdown]
# ## 9–10. Volume Estimate & Energy Check

# %%
vol = estimate_ice_volume(
    scene.cpr_L,
    result.ice_candidates,
    pixel_area_m2=scene.pixel_area_m2,
    n_mc=10_000
)
print(f"\nIce Volume Estimate:")
print(f"  Headline: {vol['headline_m3']:.0f} m³")
print(f"  90% CI:  [{vol['ci_90_lo']:.0f}, {vol['ci_90_hi']:.0f}] m³")
print(f"  Uncertainty: ±{vol['uncertainty_pct']:.1f}%")
