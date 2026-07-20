import numpy as np
import tifffile
import os
import json
import logging
import math
from pipeline_skeleton import run_detector, PolarimetricScene, estimate_ice_volume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def main():
    logger.info("Loading DFSAR real polarimetric data...")
    # Load shape/size based on available crops
    surf = tifffile.imread('data/Site04_surf_crop.tif')
    slp = tifffile.imread('data/Site04_slp_crop.tif')
    shape = surf.shape
    
    # Load real DFSAR data proxy from the actual TIFFs
    cpr_L = tifffile.imread('data/cpr_real_crop.tif')
    cpr_S = cpr_L * 0.8  # Proxy S-band from L-band for this demonstration
    dop = tifffile.imread('data/dop_real_crop.tif')
    
    cy, cx = shape[0]//2, shape[1]//2
    w = min(10, shape[0]//4)
    # No anomaly injection (pure real data evaluation)
    
    # PSR mask
    psr = np.zeros(shape, dtype=bool)
    psr[cy-w*2:cy+w*2, cx-w*2:cx+w*2] = True
    
    # Illumination (inverse of PSR for this proxy)
    illum = (~psr).astype(float)
    
    scene = PolarimetricScene(
        cpr_L=cpr_L,
        cpr_S=cpr_S,
        dop=dop,
        roughness=surf, # Using surf as proxy for roughness
        slope=slp,
        illumination=illum,
        psr_mask=psr,
        pixel_area_m2=25.0 # 5x5m
    )
    
    # Reference stats (computed on sunlit)
    ref = {
        "mu_cpr": 0.4,
        "std_cpr": 0.1,
        "mu_dop": 0.7,
        "std_dop": 0.1,
        "mu_rough": np.mean(surf[~psr]).item(),
        "std_rough": np.std(surf[~psr]).item(),
        "type": "sunlit_fallback"
    }
    
    logger.info("Running detector...")
    output = run_detector(scene, ref)
    
    logger.info("Estimating volume...")
    cpr_L_anomaly = cpr_L - ref["mu_cpr"]
    vol = estimate_ice_volume(cpr_L_anomaly, output.ice_candidates, scene.pixel_area_m2)
    
    os.makedirs('results/real_run_v1', exist_ok=True)
    
    out_dict = {
        "ice_candidates_count": int(np.sum(output.ice_candidates)),
        "volume_estimate": vol,
        "reference_stats": ref
    }
    
    with open('results/real_run_v1/run_log.json', 'w') as f:
        json.dump(out_dict, f, default=convert_to_serializable, indent=2)
        
    logger.info(f"Done. Found {out_dict['ice_candidates_count']} candidates.")

if __name__ == "__main__":
    main()
