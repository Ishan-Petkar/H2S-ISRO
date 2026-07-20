import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
import os
import sys

# Load pipeline functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import PolarimetricScene, run_detector

os.makedirs('results', exist_ok=True)

# Load data
surf = tifffile.imread('data/Site04_surf_crop.tif')
slp = tifffile.imread('data/Site04_slp_crop.tif')
cpr_L = tifffile.imread('data/cpr_real_crop.tif')
dop = tifffile.imread('data/dop_real_crop.tif')

shape = surf.shape
cy, cx = shape[0]//2, shape[1]//2
w = min(10, shape[0]//4)
psr = np.zeros(shape, dtype=bool)
psr[cy-w*2:cy+w*2, cx-w*2:cx+w*2] = True
illum = (~psr).astype(float)

scene = PolarimetricScene(
    cpr_L=cpr_L, cpr_S=cpr_L*0.8, dop=dop,
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

output = run_detector(scene, ref)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

H, W = shape
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

# We will plot surfaces at different Z heights to stack them
z_offset = 200

# Layer 1: Topography (Lowest)
z1 = surf - np.mean(surf) # Center around 0
ax.plot_surface(X, Y, z1, cmap='gray', alpha=0.9, antialiased=False, linewidth=0)
ax.text2D(0.05, 0.1, "Layer 1: LOLA DEM Topography", transform=ax.transAxes, fontsize=12)

# Layer 2: CPR L-band
z2 = np.zeros_like(surf) + z_offset
norm_cpr = (cpr_L - np.nanmin(cpr_L)) / (np.nanmax(cpr_L) - np.nanmin(cpr_L))
ax.plot_surface(X, Y, z2, facecolors=plt.cm.magma(norm_cpr), alpha=0.7, antialiased=False, linewidth=0)
ax.text2D(0.05, 0.35, "Layer 2: DFSAR CPR L-band", transform=ax.transAxes, fontsize=12)

# Layer 3: Roughness / Slope
z3 = np.zeros_like(surf) + 2 * z_offset
norm_slp = (slp - np.min(slp)) / (np.max(slp) - np.min(slp))
ax.plot_surface(X, Y, z3, facecolors=plt.cm.viridis(norm_slp), alpha=0.7, antialiased=False, linewidth=0)
ax.text2D(0.05, 0.65, "Layer 3: LOLA Slope Map", transform=ax.transAxes, fontsize=12)

# Layer 4: Final Output Mask (Highest)
z4 = np.zeros_like(surf) + 3 * z_offset
# Mask out everything except ice candidates
color_mask = np.zeros((*shape, 4))
color_mask[output.ice_candidates] = [0, 1, 1, 1] # Cyan
color_mask[~output.ice_candidates] = [0, 0, 0, 0.1] # Faint black
ax.plot_surface(X, Y, z4, facecolors=color_mask, antialiased=False, linewidth=0)
ax.text2D(0.05, 0.9, "Layer 4: Validated Ice Candidates ($\Pi$)", transform=ax.transAxes, fontsize=12, color='blue')

# Draw vertical lines to show stacking
ax.plot([W//2, W//2], [H//2, H//2], [0, 3*z_offset], color='white', linestyle='--', linewidth=1)
ax.plot([10, 10], [10, 10], [0, 3*z_offset], color='white', linestyle='--', linewidth=1)
ax.plot([W-10, W-10], [H-10, H-10], [0, 3*z_offset], color='white', linestyle='--', linewidth=1)

ax.set_zlim(np.min(z1), 3*z_offset + 50)
ax.axis('off')
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('results/datacube.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Successfully generated datacube.pdf")
