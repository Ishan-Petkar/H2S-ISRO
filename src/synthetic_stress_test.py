import sys
import os
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support

# Add project root to sys.path to import from pipeline_skeleton
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import PolarimetricScene, build_reference_population, run_detector

# Set random seed for reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

def refined_lee_filter(img, window_size=7):
    """
    Standard Lee Filter (speckle filter) implementation.
    Smoothes speckle noise while preserving edges based on local statistics.
    """
    local_mean = ndimage.uniform_filter(img, size=window_size)
    local_sqr_mean = ndimage.uniform_filter(img**2, size=window_size)
    local_var = local_sqr_mean - local_mean**2
    local_var = np.clip(local_var, 0, None)
    
    # 4-look SAR speckle variance is 1 / L = 0.25
    sigma_v = 0.5
    var_v = sigma_v**2
    
    denom = local_var
    numerator = var_v * (local_mean**2)
    w = np.zeros_like(img)
    mask = denom > 0
    w[mask] = 1.0 - numerator[mask] / denom[mask]
    w = np.clip(w, 0, 1)
    
    filtered = local_mean + w * (img - local_mean)
    return filtered

def generate_synthetic_scene(grid_size=500):
    """
    Generates a synthetic polarimetric scene on a 500x500 grid.
    Injects ice-bearing patches, rocky ejecta, and isolated noise spikes.
    Applies 1-look speckle noise, 4-look averaging, and Lee filtering.
    """
    H = W = grid_size
    # Create high-res grid (1000x1000) for 1-look simulation
    H_hr = W_hr = grid_size * 2
    
    # Initialize base maps
    cpr_L_base = np.full((H_hr, W_hr), 0.35)
    cpr_S_base = np.full((H_hr, W_hr), 0.30)
    dop_base = np.full((H_hr, W_hr), 0.75)
    roughness_base = np.full((H_hr, W_hr), 0.35) # RMS roughness in metres
    slope = np.zeros((H, W))                     # Stated size maps
    illumination = np.full((H, W), 0.90)
    psr_mask = np.zeros((H, W), dtype=bool)
    
    # Define circular permanently shadowed region (PSR) in the center of 500x500 grid
    # Radius = 150 pixels
    r_center, c_center = 250, 250
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((Y - r_center)**2 + (X - c_center)**2)
    psr_mask[dist_from_center <= 150] = True
    illumination[psr_mask] = 0.0
    
    # Ground truth ice mask on the 500x500 grid
    gt_ice_mask = np.zeros((H, W), dtype=bool)
    
    # Ice patch 1: Large cluster (>100 m^2) - high ice content
    # Center (200, 200), radius 20 pixels on 500x500 (40 pixels on 1000x1000)
    r1, c1, rad1 = 200, 200, 20
    gt_ice_mask[np.sqrt((Y - r1)**2 + (X - c1)**2) <= rad1] = True
    
    # Ice patch 2: Medium cluster (>100 m^2) - moderate ice content
    # Center (280, 220), radius 12 pixels
    r2, c2, rad2 = 280, 220, 12
    gt_ice_mask[np.sqrt((Y - r2)**2 + (X - c2)**2) <= rad2] = True
    
    # Ice patch 3: Small cluster (>100 m^2) - low ice content
    # Center (240, 290), radius 6 pixels
    r3, c3, rad3 = 240, 290, 6
    gt_ice_mask[np.sqrt((Y - r3)**2 + (X - c3)**2) <= rad3] = True
    
    # Ice patch 4: Sub-threshold patch (<100 m^2)
    # Center (320, 320), single pixel spike (represented as True on 500x500)
    gt_ice_mask[320, 320] = True
    
    # Map the ice patches back to 1000x1000 high-res grid to inject values
    for r in range(H_hr):
        for c in range(W_hr):
            r500, c500 = r // 2, c // 2
            if gt_ice_mask[r500, c500]:
                cpr_L_base[r, c] = 0.95
                cpr_S_base[r, c] = 0.75
                dop_base[r, c] = 0.32
    
    # Inject rock/ejecta patch: high CPR, high roughness, but high DOP (no ice)
    # Center (350, 250), radius 22 pixels on 500x500 grid
    r_ejecta, c_ejecta, rad_ejecta = 350, 250, 22
    for r in range(H_hr):
        for c in range(W_hr):
            r500, c500 = r // 2, c // 2
            if np.sqrt((r500 - r_ejecta)**2 + (c500 - c_ejecta)**2) <= rad_ejecta:
                cpr_L_base[r, c] = 0.85
                cpr_S_base[r, c] = 0.80
                dop_base[r, c] = 0.72
                roughness_base[r, c] = 1.1
                
    # Inject isolated single-pixel noise spikes (not ice) in the PSR
    # e.g., at (220, 270) and (260, 180)
    for r, c in [(220, 270), (260, 180)]:
        cpr_L_base[r*2:(r+1)*2, c*2:(c+1)*2] = 1.1
        dop_base[r*2:(r+1)*2, c*2:(c+1)*2] = 0.25
        
    # --- Simulate 1-look SAR Backscatter & Speckle Noise ---
    I_oc_L = rng.exponential(scale=1.0, size=(H_hr, W_hr))
    I_sc_L = rng.exponential(scale=cpr_L_base, size=(H_hr, W_hr))
    
    I_oc_S = rng.exponential(scale=1.0, size=(H_hr, W_hr))
    I_sc_S = rng.exponential(scale=cpr_S_base, size=(H_hr, W_hr))
    
    # --- Multilooking: 4-look average (2x2 spatial binning) ---
    I_oc_L_multilook = I_oc_L.reshape(H, 2, W, 2).mean(axis=(1, 3))
    I_sc_L_multilook = I_sc_L.reshape(H, 2, W, 2).mean(axis=(1, 3))
    
    I_oc_S_multilook = I_oc_S.reshape(H, 2, W, 2).mean(axis=(1, 3))
    I_sc_S_multilook = I_sc_S.reshape(H, 2, W, 2).mean(axis=(1, 3))
    
    cpr_L_noisy = I_sc_L_multilook / np.clip(I_oc_L_multilook, 1e-4, None)
    cpr_S_noisy = I_sc_S_multilook / np.clip(I_oc_S_multilook, 1e-4, None)
    
    # DOP is degree of polarization, simulated on 500x500 with Gaussian noise
    dop_noisy = dop_base.reshape(H, 2, W, 2).mean(axis=(1, 3)) + rng.normal(0.0, 0.08, size=(H, W))
    dop_noisy = np.clip(dop_noisy, 0.0, 1.0)
    
    # Roughness is simulated on 500x500 with small scale Gaussian noise
    roughness_noisy = roughness_base.reshape(H, 2, W, 2).mean(axis=(1, 3)) + rng.normal(0.0, 0.05, size=(H, W))
    roughness_noisy = np.clip(roughness_noisy, 0.05, None)
    
    # --- Apply 7x7 Speckle Filter (Lee Filter) ---
    cpr_L_filtered = refined_lee_filter(cpr_L_noisy, window_size=7)
    cpr_S_filtered = refined_lee_filter(cpr_S_noisy, window_size=7)
    dop_filtered = refined_lee_filter(dop_noisy, window_size=7)
    
    # Create the final PolarimetricScene object
    scene = PolarimetricScene(
        cpr_L=cpr_L_filtered,
        cpr_S=cpr_S_filtered,
        dop=dop_filtered,
        roughness=roughness_noisy,
        slope=slope,
        illumination=illumination,
        psr_mask=psr_mask,
        pixel_area_m2=225.0
    )
    
    return scene, gt_ice_mask

def run_roughness_ablation(scene, gt_mask):
    """
    Sweeps the roughness veto threshold from 1.5 to 3.0 sigma.
    Computes precision, recall, and F1 at each step to generate Figure 5 data.
    """
    ref = build_reference_population(scene)
    thresholds = np.linspace(1.5, 3.0, 7)
    precision_list = []
    recall_list = []
    f1_list = []
    
    print("\n--- Roughness Veto Ablation Sweep ---")
    print("Veto (sigma) | Precision | Recall | F1 Score")
    print("--------------------------------------------")
    
    eval_gt = gt_mask.copy()
    eval_gt[320, 320] = False # Exclude sub-threshold single pixel from positive class
    
    for thresh in thresholds:
        # Stages 1-5 Custom Run
        rough_thresh = ref["mu_rough"] + thresh * ref["std_rough"]
        z_cpr = (scene.cpr_L - ref["mu_cpr"]) / ref["std_cpr"]
        z_dop = (scene.dop - ref["mu_dop"]) / ref["std_dop"]
        stage2 = scene.psr_mask & (z_cpr >= 2.0) & (z_dop <= -2.0)
        
        roughness_anomaly = scene.roughness > rough_thresh
        stage3 = stage2 & ~roughness_anomaly
        
        coords = np.column_stack(np.where(stage3))
        stage4_mask = np.zeros_like(stage3, dtype=bool)
        if len(coords) >= 4:
            pixel_size = np.sqrt(scene.pixel_area_m2)
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=30.0, min_samples=4).fit_predict(coords * pixel_size)
            for lbl in set(labels):
                if lbl == -1:
                    continue
                cluster = coords[labels == lbl]
                if len(cluster) * scene.pixel_area_m2 >= 1000.0:
                    stage4_mask[cluster[:, 0], cluster[:, 1]] = True
        
        stage5_mask = stage4_mask.copy()
        
        # Evaluate
        y_true = eval_gt.flatten()
        y_pred = stage5_mask.flatten()
        
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        print(f"    {thresh:.2f}     |   {p:.3f}   |  {r:.3f} |  {f:.3f}")
        
    return thresholds, precision_list, recall_list, f1_list

def run_stage_waterfall(scene):
    """
    Computes candidate pixel survival counts through each of the 5 stages.
    """
    ref = build_reference_population(scene)
    counts = []
    
    total_psr = scene.psr_mask.sum()
    counts.append(total_psr)
    
    z_cpr = (scene.cpr_L - ref["mu_cpr"]) / ref["std_cpr"]
    z_dop = (scene.dop - ref["mu_dop"]) / ref["std_dop"]
    stage2 = scene.psr_mask & (z_cpr >= 2.0) & (z_dop <= -2.0)
    counts.append(stage2.sum())
    
    rough_thresh = ref["mu_rough"] + 2.0 * ref["std_rough"]
    roughness_anomaly = scene.roughness > rough_thresh
    stage3 = stage2 & ~roughness_anomaly
    counts.append(stage3.sum())
    
    coords = np.column_stack(np.where(stage3))
    stage4_mask = np.zeros_like(stage3, dtype=bool)
    if len(coords) >= 4:
        pixel_size = np.sqrt(scene.pixel_area_m2)
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=30.0, min_samples=4).fit_predict(coords * pixel_size)
        for lbl in set(labels):
            if lbl == -1:
                continue
            cluster = coords[labels == lbl]
            if len(cluster) * scene.pixel_area_m2 >= 1000.0:
                stage4_mask[cluster[:, 0], cluster[:, 1]] = True
    counts.append(stage4_mask.sum())
    
    stage5_mask = stage4_mask.copy()
    counts.append(stage5_mask.sum())
    
    stage_names = ["PSR pixels", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
    print("\n--- Pixel Survival Counts ---")
    for name, count in zip(stage_names, counts):
        print(f"  {name:12s}: {count:6d} ({count/total_psr*100:6.2f}%)")
        
    return stage_names, counts

def run_dbscan_ablation(scene, gt_mask):
    """
    Sweeps DBSCAN epsilon in {15, 30, 60}m and minPts in {2, 4, 8}.
    """
    ref = build_reference_population(scene)
    eps_vals = [15.0, 30.0, 60.0]
    minpts_vals = [2, 4, 8]
    
    eval_gt = gt_mask.copy()
    eval_gt[320, 320] = False
    
    z_cpr = (scene.cpr_L - ref["mu_cpr"]) / ref["std_cpr"]
    z_dop = (scene.dop - ref["mu_dop"]) / ref["std_dop"]
    stage2 = scene.psr_mask & (z_cpr >= 2.0) & (z_dop <= -2.0)
    rough_thresh = ref["mu_rough"] + 2.0 * ref["std_rough"]
    stage3 = stage2 & ~(scene.roughness > rough_thresh)
    
    results = []
    
    print("\n--- DBSCAN Parameter Ablation ---")
    print("Eps (m) | minPts | Precision | Recall | F1 Score")
    print("-------------------------------------------------")
    
    for eps in eps_vals:
        for minpts in minpts_vals:
            coords = np.column_stack(np.where(stage3))
            stage4_mask = np.zeros_like(stage3, dtype=bool)
            if len(coords) >= minpts:
                pixel_size = np.sqrt(scene.pixel_area_m2)
                from sklearn.cluster import DBSCAN
                labels = DBSCAN(eps=eps, min_samples=minpts).fit_predict(coords * pixel_size)
                for lbl in set(labels):
                    if lbl == -1:
                        continue
                    cluster = coords[labels == lbl]
                    if len(cluster) * scene.pixel_area_m2 >= 1000.0:
                        stage4_mask[cluster[:, 0], cluster[:, 1]] = True
                        
            y_true = eval_gt.flatten()
            y_pred = stage4_mask.flatten()
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            results.append((eps, minpts, p, r, f))
            print(f"  {eps:2.0f}m   |   {minpts:2d}   |   {p:.3f}   |  {r:.3f} |  {f:.3f}")
            
    return results

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("Generating synthetic polarimetric scene...")
    scene, gt_mask = generate_synthetic_scene()
    
    thresholds, p_vals, r_vals, f1_vals = run_roughness_ablation(scene, gt_mask)
    stage_names, counts = run_stage_waterfall(scene)
    dbscan_results = run_dbscan_ablation(scene, gt_mask)
    
    # Save CSVs of results
    with open("results/roughness_ablation.csv", "w") as f:
        f.write("threshold_sigma,precision,recall,f1\n")
        for t, p, r, f1 in zip(thresholds, p_vals, r_vals, f1_vals):
            f.write(f"{t:.2f},{p:.4f},{r:.4f},{f1:.4f}\n")
            
    with open("results/waterfall_survival.csv", "w") as f:
        f.write("stage,count,fraction_psr\n")
        for name, count in zip(stage_names, counts):
            f.write(f"{name},{count},{count/counts[0]:.4f}\n")
            
    with open("results/dbscan_ablation.csv", "w") as f:
        f.write("epsilon_m,min_pts,precision,recall,f1\n")
        for eps, minpts, p, r, f1 in dbscan_results:
            f.write(f"{eps:.1f},{minpts},{p:.4f},{r:.4f},{f1:.4f}\n")
            
    print("\nAll synthetic test results computed and saved to results/ folder.")
