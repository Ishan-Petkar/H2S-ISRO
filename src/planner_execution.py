import numpy as np
import scipy.ndimage as ndimage
import sys
import os
import logging
from skyfield.api import load, EarthSatellite, wgs84

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add root to sys.path to import pipeline_skeleton
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_skeleton import astar_traverse, compute_energy, PolarimetricScene

SEED = 42
rng = np.random.default_rng(SEED)

def load_real_lola_dem(shape=(200, 200), pixel_size_m=5.0):
    """Loads a real LOLA GeoTIFF and crops a representative region."""
    import tifffile
    logger.info(f"Loading real LOLA DEM and Slope maps (cropping to {shape})...")
    surf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Site04_surf.tif')
    slp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Site04_slp.tif')
    
    if not os.path.exists(surf_path) or not os.path.exists(slp_path):
        logger.warning(f"Missing full LOLA files. Falling back to crops.")
        surf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Site04_surf_crop.tif')
        slp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Site04_slp_crop.tif')
        if not os.path.exists(surf_path) or not os.path.exists(slp_path):
            raise FileNotFoundError(f"Missing LOLA files at {surf_path} or {slp_path}.")
        
    surf = tifffile.imread(surf_path)
    slp = tifffile.imread(slp_path)
    
    # Take a crop from the center
    cy, cx = surf.shape[0] // 2, surf.shape[1] // 2
    y0, y1 = cy - shape[0]//2, cy + shape[0]//2
    x0, x1 = cx - shape[1]//2, cx + shape[1]//2
    
    dem = surf[y0:y1, x0:x1]
    slope_map = slp[y0:y1, x0:x1]
    
    # Replace any NaNs if present
    dem = np.nan_to_num(dem, nan=np.nanmedian(dem))
    slope_map = np.nan_to_num(slope_map, nan=np.nanmedian(slope_map))
    
    return dem, slope_map

def compute_chandrayaan2_relay_window():
    """
    Simulates a representative polar orbit (circular, 100 km altitude, 90 deg inclination, 
    mean motion ~ 12 revs/day) as a proxy to model the ~2-hour orbit visibility cadence 
    over a South Pole target (-89.9 deg).
    
    Note: Real lunar TLEs are not tracked under the Earth-centric SGP4 model.
    This representative orbit acts as a clean, labeled geometric proxy for mission planning.
    """
    logger.info("Computing representative orbital relay window cadence...")
    ts = load.timescale()
    t0 = ts.utc(2026, 1, 1)
    t1 = ts.utc(2026, 1, 3)
    
    # Representative TLE representing a circular polar orbit (90 deg inc, 12 revs/day)
    line1 = '1 99999U 26001A   26001.00000000  .00000000  00000-0  00000-0 0  9997'
    line2 = '2 99999  90.0000   0.0000 0001000   0.0000   0.0000 12.00000000    10'
    chandrayaan2_representative_orbit = EarthSatellite(line1, line2, 'CHANDRAYAAN-2-REPRESENTATIVE', ts)
    
    # Target: Shackleton rim equivalent (-89.9 S)
    target = wgs84.latlon(-89.9, 0.0)
    
    # Find events (rise above 5 deg elevation)
    t, events = chandrayaan2_representative_orbit.find_events(target, t0, t1, altitude_degrees=5.0)
    
    # Calculate max blackout time
    blackout_hours = []
    last_loss = None
    for ti, event in zip(t, events):
        # event: 0=rise, 1=culminate, 2=set
        if event == 2:
            last_loss = ti
        elif event == 0 and last_loss is not None:
            blackout_hours.append((ti.tt - last_loss.tt) * 24.0)
            
    max_blackout = max(blackout_hours) if blackout_hours else 2.0
    logger.info(f"Simulated representative orbital blackout (relay constraint): {max_blackout:.2f} hours")
    return max_blackout

def main():
    logger.info("--- Phase 3: Executing Joint Energy & Relay Planner ---")
    
    # 1. Compute Relay Constraint
    t_relay_hours = compute_chandrayaan2_relay_window()
    
    # 2. Setup Terrain
    pixel_size_m = 5.0
    shape = (200, 200) # 1000m x 1000m
    dem, slope_map = load_real_lola_dem(shape=shape, pixel_size_m=pixel_size_m)
    
    # Cost map (alpha, beta, gamma, delta weights from paper: 0.4, 0.3, 0.2, 0.1)
    # Since we are just testing the A* energy pruning, we'll use slope as the main cost driver
    cost_map = 0.3 * (slope_map / 30.0) + 0.1 # Base cost
    
    # Start (lander) and Targets (ice patches)
    start_pos = (10, 10)
    targets = np.array([[180, 180], [175, 185]]) # Deep in the PSR
    
    # 3. Rover Specifications (VIPER-class)
    e_max_wh = 5000.0 # 5 kWh battery
    e_science_stops = 5.0 # Wh per stop
    n_stops = 3
    p_rhu_comms = 3.0 # W
    
    # Hard bounds
    usable_energy = e_max_wh * 0.80 # 20% contingency
    usable_time = t_relay_hours
    
    # We deduct static thermal/comms overhead and science overhead from usable energy
    overhead_energy = (usable_time * p_rhu_comms) + (n_stops * e_science_stops)
    locomotion_energy_allowance = usable_energy - overhead_energy
    
    logger.info(f"Rover Battery: {e_max_wh} Wh")
    logger.info(f"Locomotion Energy Allowance: {locomotion_energy_allowance:.1f} Wh")
    
    # 4. Run A* Traverse
    # The A* function from pipeline_skeleton needs 'e_max_wh' to mean 'usable_energy'
    output = astar_traverse(
        cost_map=cost_map,
        start=start_pos,
        targets=targets,
        e_max_wh=e_max_wh, # It multiplies by (1-0.2) = 0.8 internally
        t_relay_hours=usable_time,
        pixel_size_m=pixel_size_m,
        slope_map=slope_map,
        contingency=0.20
    )
    
    if output.feasible:
        path, e_consumed, _ = output.paths[0]
        length_m = sum(
            np.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2) * pixel_size_m
            for i in range(len(path)-1)
        )
        logger.info(f"--- SUCCESS ---")
        logger.info(f"Optimal Path Length: {length_m:.1f} meters")
        logger.info(f"Energy Consumed: {e_consumed:.1f} Wh ({(e_consumed/e_max_wh)*100:.1f}% of E_max)")
        logger.info(f"Margin Remaining: {usable_energy - e_consumed:.1f} Wh")
    else:
        logger.error("Failed to find feasible path.")

if __name__ == "__main__":
    main()
