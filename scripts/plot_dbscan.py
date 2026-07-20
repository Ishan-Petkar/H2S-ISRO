import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import sys

# Load pipeline functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import PolarimetricScene, run_detector

os.makedirs('results', exist_ok=True)

# Recreate scene loading from run_real_pipeline.py
surf = tifffile.imread('data/Site04_surf_crop.tif')
slp = tifffile.imread('data/Site04_slp_crop.tif')
shape = surf.shape

cpr_L = tifffile.imread('data/cpr_real_crop.tif')
cpr_S = cpr_L * 0.8
dop = tifffile.imread('data/dop_real_crop.tif')

cy, cx = shape[0]//2, shape[1]//2
w = min(10, shape[0]//4)
psr = np.zeros(shape, dtype=bool)
psr[cy-w*2:cy+w*2, cx-w*2:cx+w*2] = True
illum = (~psr).astype(float)

scene = PolarimetricScene(
    cpr_L=cpr_L, cpr_S=cpr_S, dop=dop,
    roughness=surf, slope=slp,
    illumination=illum, psr_mask=psr,
    pixel_area_m2=25.0
)

ref = {
    "mu_cpr": 0.4, "std_cpr": 0.1,
    "mu_dop": 0.7, "std_dop": 0.1,
    "mu_rough": np.mean(surf[~psr]).item(),
    "std_rough": np.std(surf[~psr]).item(),
    "type": "sunlit_fallback"
}

# Reproduce Stage 2 and 3 mathematically for the left panel
cpr_anomaly = (scene.cpr_L - ref['mu_cpr']) / ref['std_cpr'] >= 2.0
dop_anomaly = (scene.dop - ref['mu_dop']) / ref['std_dop'] <= -2.0
stage2_mask = cpr_anomaly & dop_anomaly
roughness_anomaly = (scene.roughness - ref['mu_rough']) / ref['std_rough'] > 2.0
veto_mask = stage2_mask & (~roughness_anomaly)

# Run full pipeline to get post-DBSCAN (left panel)
output = run_detector(scene, ref)

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Use a colormap that makes anomalous pixels pop against the background
cmap = plt.cm.binary_r

ax1.imshow(veto_mask, cmap=cmap, interpolation='nearest')
ax1.set_title('Pre-DBSCAN Filter (Stages 1-3)', fontsize=14)
ax1.axis('off')

ax2.imshow(output.ice_candidates, cmap=cmap, interpolation='nearest')
ax2.set_title('Post-DBSCAN Spatial Filter (Stage 4)', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig('results/dbscan_scatter.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Successfully generated dbscan_scatter.pdf")
