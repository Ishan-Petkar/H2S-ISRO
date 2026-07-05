"""
pradan_ingestion.py
Script to ingest ISRO PRADAN L2 DFSAR Hybrid-Polarimetry (Circular Transmit, Linear Receive) Stokes parameters,
compute CPR and DOP, and prepare them for the H2S pipeline.
"""

import numpy as np
import logging
import os

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    import tifffile
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pradan_l2(s1_path: str, s2_path: str, s3_path: str, s4_path: str, out_dir: str):
    """
    Load Stokes parameters (S1, S2, S3, S4) from PRADAN L2 GeoTIFFs.
    Computes CPR and DOP and saves them, preserving geospatial metadata if rasterio is installed.
    """
    logger.info("Loading Stokes parameters from L2 GeoTIFFs...")
    
    if HAS_RASTERIO:
        logger.info("Using rasterio (Geospatial Metadata will be preserved).")
        with rasterio.open(s1_path) as src1, \
             rasterio.open(s2_path) as src2, \
             rasterio.open(s3_path) as src3, \
             rasterio.open(s4_path) as src4:
             
            profile = src1.profile
            profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            
            s1 = src1.read(1).astype(np.float32)
            s2 = src2.read(1).astype(np.float32)
            s3 = src3.read(1).astype(np.float32)
            s4 = src4.read(1).astype(np.float32)
    else:
        logger.warning("rasterio not found. Using tifffile. WARNING: Geospatial CRS will be lost!")
        s1 = tifffile.imread(s1_path).astype(np.float32)
        s2 = tifffile.imread(s2_path).astype(np.float32)
        s3 = tifffile.imread(s3_path).astype(np.float32)
        s4 = tifffile.imread(s4_path).astype(np.float32)

    logger.info("Computing CPR and DOP...")
    # Avoid division by zero
    s1_safe = np.where(s1 == 0, np.nan, s1)
    
    # CPR = (S1 - S4) / (S1 + S4)
    # Note: Standard for LHCP transmit is (S1-S4)/(S1+S4). If RHCP, sign flips.
    cpr = (s1 - s4) / (s1 + s4 + 1e-12)
    
    # DOP = sqrt(S2^2 + S3^2 + S4^2) / S1
    dop = np.sqrt(s2**2 + s3**2 + s4**2) / s1_safe
    
    # Clip to physical limits to prevent extreme numerical spikes
    cpr = np.clip(cpr, 0, 3.0).astype(np.float32)
    dop = np.clip(dop, 0, 1.0).astype(np.float32)
    
    if HAS_RASTERIO:
        os.makedirs(out_dir, exist_ok=True)
        with rasterio.open(os.path.join(out_dir, "cpr.tif"), 'w', **profile) as dst:
            dst.write(cpr, 1)
        with rasterio.open(os.path.join(out_dir, "dop.tif"), 'w', **profile) as dst:
            dst.write(dop, 1)
        logger.info(f"Saved CPR and DOP to {out_dir} with intact CRS.")
    else:
        logger.info("Computed CPR and DOP matrices (in memory).")
    
    return cpr, dop

if __name__ == "__main__":
    logger.info("This script is a template for PRADAN L2 data ingestion.")
    logger.info("Please provide paths to S1, S2, S3, S4 TIFFs to extract polarimetric features.")
