"""Sweeps cost function weights in the A* planner and outputs ablation results."""
import numpy as np
import os, sys, csv
import scipy.ndimage as ndimage

# Add root to sys.path to import pipeline_skeleton
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import astar_traverse, compute_energy, PolarimetricScene

def load_crop_dem(shape=(200, 200)):
    import tifffile
    surf_path = os.path.join(os.path.dirname(__file__), "..", "data", "Site04_surf_crop.tif")
    slp_path = os.path.join(os.path.dirname(__file__), "..", "data", "Site04_slp_crop.tif")
    
    if not os.path.exists(surf_path) or not os.path.exists(slp_path):
        rng = np.random.default_rng(42)
        dem = np.zeros(shape)
        slope_map = rng.uniform(0, 15, size=shape)
        return dem, slope_map
        
    dem = tifffile.imread(surf_path)
    slope_map = tifffile.imread(slp_path)
    
    return dem, slope_map

def main():
    print("Starting A* cost-map weight ablation sweep...")
    dem, slope_map = load_crop_dem()
    H, W = dem.shape
    print(f"Loaded DEM shape: {H}x{W}")
    
    # Generate mock Pi (ice probability) and Phi (false positive risk) maps
    rng = np.random.default_rng(42)
    pi_map = np.zeros((H, W))
    # Inject an ice-likely channel in the center (use fractional coords)
    r0, r1 = int(H * 0.25), int(H * 0.75)
    c0, c1 = int(W * 0.25), int(W * 0.75)
    pi_map[r0:r1, c0:c1] = rng.uniform(0.5, 0.9, (r1 - r0, c1 - c0))
    
    phi_map = np.zeros((H, W))
    # High risk zone
    r2, r3 = int(H * 0.6), int(H * 0.85)
    c2, c3 = int(W * 0.6), int(W * 0.85)
    phi_map[r2:r3, c2:c3] = rng.uniform(0.4, 0.8, (r3 - r2, c3 - c2))
    
    # We will compute roughness as local slope standard deviation
    roughness = ndimage.generic_filter(slope_map, np.std, size=3)
    roughness = np.nan_to_num(roughness, nan=0.1)
    
    start_pos = (2, 2) # Adjust start for 50x50 grid
    targets = np.array([[int(H*0.9), int(W*0.9)], [int(H*0.88), int(W*0.92)]])
    
    e_max_wh = 5000.0
    t_relay_hours = 1.63
    pixel_size_m = 5.0
    
    # Sweep ranges
    alpha_vals = [0.2, 0.4, 0.6]
    beta_vals = [0.1, 0.3, 0.5]
    gamma_vals = [0.1, 0.2, 0.3]
    delta_vals = [0.05, 0.10, 0.20]
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_csv = os.path.join(os.path.dirname(__file__), "..", "results", "weight_ablation.csv")
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "gamma", "delta", "feasible", "path_length_m", "energy_wh"])
        
        count = 0
        for alpha in alpha_vals:
            for beta in beta_vals:
                for gamma in gamma_vals:
                    for delta in delta_vals:
                        # Construct cost map
                        # C(p) = alpha*(1-Pi(p)) + beta*(s(p)/s_max) + gamma*Phi(p) + delta*rho(p)
                        s_max = 30.0
                        rho_max = max(1e-4, roughness.max())
                        cost_map = (
                            alpha * (1.0 - pi_map) +
                            beta * (slope_map / s_max) +
                            gamma * phi_map +
                            delta * (roughness / rho_max)
                        )
                        
                        output = astar_traverse(
                            cost_map=cost_map,
                            start=start_pos,
                            targets=targets,
                            e_max_wh=e_max_wh,
                            t_relay_hours=t_relay_hours,
                            pixel_size_m=pixel_size_m,
                            slope_map=slope_map,
                            contingency=0.20
                        )
                        
                        if output.feasible:
                            path, e_consumed, _ = output.paths[0]
                            length_m = len(path) * pixel_size_m
                            writer.writerow([alpha, beta, gamma, delta, True, length_m, round(e_consumed, 2)])
                        else:
                            writer.writerow([alpha, beta, gamma, delta, False, 0.0, 0.0])
                        
                        count += 1
                        
    print(f"Weight ablation sweep complete. Wrote {count} rows to {out_csv}.")

if __name__ == "__main__":
    main()
