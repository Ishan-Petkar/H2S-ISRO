import numpy as np
import tifffile
import os

cpr = tifffile.imread('data/cpr_real_crop.tif')
dop = tifffile.imread('data/dop_real_crop.tif')

# Flatten and remove NaN/Inf just in case
valid = np.isfinite(cpr) & np.isfinite(dop)
cpr_v = cpr[valid].flatten()
dop_v = dop[valid].flatten()

r = np.corrcoef(cpr_v, dop_v)[0,1]
print(f"Empirical within-class pixel correlation r = {r}")
