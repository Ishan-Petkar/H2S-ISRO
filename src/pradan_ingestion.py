"""
pradan_ingestion.py
Script to ingest ISRO PRADAN L2 DFSAR Hybrid-Polarimetry (Circular Transmit, Linear Receive) Stokes parameters,
compute CPR and DOP, and prepare them for the H2S pipeline.
"""

import numpy as np
import tifffile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stokes_from_pradan(s1_path: str, s2_path: str, s3_path: str, s4_path: str):
    """
    Load Stokes parameters (S1, S2, S3, S4) from PRADAN L2 GeoTIFFs.
    """
    logger.info("Loading Stokes parameters from L2 GeoTIFFs...")
    s1 = tifffile.imread(s1_path)
    s2 = tifffile.imread(s2_path)
    s3 = tifffile.imread(s3_path)
    s4 = tifffile.imread(s4_path)
    return s1, s2, s3, s4

def compute_cpr_and_dop(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray, s4: np.ndarray):
    """
    Compute Circular Polarization Ratio (CPR) and Degree of Polarization (DOP)
    from Hybrid-Pol Stokes parameters.
    """
    logger.info("Computing CPR and DOP...")
    # Avoid division by zero
    s1_safe = np.where(s1 == 0, np.nan, s1)
    
    # CPR = (S1 - S4) / (S1 + S4)
    # Note: Standard for LHCP transmit is (S1-S4)/(S1+S4). If RHCP, sign flips.
    cpr = (s1 - s4) / (s1 + s4 + 1e-12)
    
    # DOP = sqrt(S2^2 + S3^2 + S4^2) / S1
    dop = np.sqrt(s2**2 + s3**2 + s4**2) / s1_safe
    
    # Clip to physical limits to prevent extreme numerical spikes
    cpr = np.clip(cpr, 0, 3.0)
    dop = np.clip(dop, 0, 1.0)
    
    return cpr, dop

if __name__ == "__main__":
    # Example usage placeholder
    logger.info("This script is a template for PRADAN L2 data ingestion.")
    logger.info("Please provide paths to S1, S2, S3, S4 TIFFs to extract polarimetric features.")
