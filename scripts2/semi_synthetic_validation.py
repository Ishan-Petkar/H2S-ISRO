"""Runs semi-synthetic validation of the cascade detector using real-geometry spacecraft ice masks."""
import numpy as np
import os, sys, csv
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support

# Add root to sys.path to import pipeline_skeleton
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import PolarimetricScene, build_reference_population, run_detector

SEED = 42
rng = np.random.default_rng(SEED)

def refined_lee_filter(img, window_size=7):
    local_mean = ndimage.uniform_filter(img, size=window_size)
    local_sqr_mean = ndimage.uniform_filter(img**2, size=window_size)
    local_var = local_sqr_mean - local_mean**2
    local_var = np.clip(local_var, 0, None)
    
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

def generate_semi_synthetic_scene(mask_path):
    gt_ice_mask = np.load(mask_path).astype(bool)
    H, W = gt_ice_mask.shape
    
    # High-res grid (2x) for multilook simulation
    H_hr, W_hr = H * 2, W * 2
    gt_ice_mask_hr = np.repeat(np.repeat(gt_ice_mask, 2, axis=0), 2, axis=1)
    
    cpr_L_base = np.full((H_hr, W_hr), 0.35)
    cpr_S_base = np.full((H_hr, W_hr), 0.30)
    dop_base = np.full((H_hr, W_hr), 0.75)
    roughness_base = np.full((H_hr, W_hr), 0.35)
    
    # Inject ice signature in mask regions
    cpr_L_base[gt_ice_mask_hr] = 0.95
    cpr_S_base[gt_ice_mask_hr] = 0.75
    dop_base[gt_ice_mask_hr] = 0.32
    
    # Inject a rocky ejecta zone (non-ice anomaly) to test specificity
    # Put it in a corner where there is no ice
    for r in range(H_hr):
        for c in range(W_hr):
            if r // 2 > 180 and c // 2 < 70 and not gt_ice_mask_hr[r, c]:
                cpr_L_base[r, c] = 0.85
                cpr_S_base[r, c] = 0.80
                dop_base[r, c] = 0.72
                roughness_base[r, c] = 1.1
                
    # Simulate speckle and multilooking
    I_oc_L = rng.exponential(scale=1.0, size=(H_hr, W_hr))
    I_sc_L = rng.exponential(scale=cpr_L_base, size=(H_hr, W_hr))
    I_oc_S = rng.exponential(scale=1.0, size=(H_hr, W_hr))
    I_sc_S = rng.exponential(scale=cpr_S_base, size=(H_hr, W_hr))
    
    I_oc_L_multilook = I_oc_L.reshape(H, 2, W, 2).mean(axis=(1, 3))
    I_sc_L_multilook = I_sc_L.reshape(H, 2, W, 2).mean(axis=(1, 3))
    I_oc_S_multilook = I_oc_S.reshape(H, 2, W, 2).mean(axis=(1, 3))
    I_sc_S_multilook = I_sc_S.reshape(H, 2, W, 2).mean(axis=(1, 3))
    
    cpr_L_noisy = I_sc_L_multilook / np.clip(I_oc_L_multilook, 1e-4, None)
    cpr_S_noisy = I_sc_S_multilook / np.clip(I_oc_S_multilook, 1e-4, None)
    
    dop_noisy = dop_base.reshape(H, 2, W, 2).mean(axis=(1, 3)) + rng.normal(0.0, 0.08, size=(H, W))
    dop_noisy = np.clip(dop_noisy, 0.0, 1.0)
    
    roughness_noisy = roughness_base.reshape(H, 2, W, 2).mean(axis=(1, 3)) + rng.normal(0.0, 0.05, size=(H, W))
    roughness_noisy = np.clip(roughness_noisy, 0.05, None)
    
    cpr_L_filtered = refined_lee_filter(cpr_L_noisy, window_size=7)
    cpr_S_filtered = refined_lee_filter(cpr_S_noisy, window_size=7)
    dop_filtered = refined_lee_filter(dop_noisy, window_size=7)
    
    # Circular PSR covering most of the crop
    psr_mask = np.zeros((H, W), dtype=bool)
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((Y - H//2)**2 + (X - W//2)**2)
    # Increase the PSR radius to cover more of the crop (e.g. 125 pixels)
    psr_mask[dist_from_center <= H//2 - 2] = True
    
    scene = PolarimetricScene(
        cpr_L=cpr_L_filtered,
        cpr_S=cpr_S_filtered,
        dop=dop_filtered,
        roughness=roughness_noisy,
        slope=rng.uniform(0, 10, size=(H, W)),
        illumination=np.where(psr_mask, 0.0, 0.90),
        psr_mask=psr_mask,
        pixel_area_m2=625.0 # 25m x 25m
    )
    
    # The evaluation ground truth must be masked by the PSR as well
    eval_gt = gt_ice_mask & psr_mask
    
    return scene, eval_gt

def evaluate_baseline(scene, gt_mask):
    """Global-threshold baseline: CPR_L > 1.0 AND DOP < 0.13 inside PSR."""
    y_true = gt_mask.flatten()
    y_pred = (scene.psr_mask & (scene.cpr_L > 1.0) & (scene.dop < 0.13)).flatten()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    sverdrup_mask = os.path.join(results_dir, "sverdrup_active_ice_mask_crop.npy")
    faustini_mask = os.path.join(results_dir, "faustini_active_ice_mask_crop.npy")
    
    out_csv = os.path.join(results_dir, "real_mask_validation.csv")
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_name", "method", "precision", "recall", "f1"])
        
        for name, mask_path in [("Sverdrup", sverdrup_mask), ("Faustini", faustini_mask)]:
            if not os.path.exists(mask_path):
                print(f"Skipping {name} (crop not found)")
                continue
                
            scene, gt_mask = generate_semi_synthetic_scene(mask_path)
            
            print(f"Evaluating {name} Scene:")
            print(f"  Total pixels in mask: {np.load(mask_path).sum()}")
            print(f"  Total pixels in eval_gt (inside PSR): {gt_mask.sum()}")
            
            if gt_mask.sum() == 0:
                print("  Warning: eval_gt is empty!")
            
            # 1. Run Baseline
            bp, br, bf = evaluate_baseline(scene, gt_mask)
            writer.writerow([name, "global_threshold_baseline", bp, br, bf])
            print(f"  {name} Baseline: P={bp:.4f}, R={br:.4f}, F1={bf:.4f}")
            
            # 2. Run Cascade
            ref = build_reference_population(scene)
            # Use standard parameters, but we can set sigma_thresh lower if needed (e.g. 1.5) to balance noise
            output = run_detector(
                scene, ref,
                sigma_thresh=1.5,
                roughness_sigma=2.5,
                dbscan_eps=60.0,
                dbscan_minpts=4,
                min_cluster_area_m2=100.0
            )
            
            y_true = gt_mask.flatten()
            y_pred = output.ice_candidates.flatten()
            cp, cr, cf, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            writer.writerow([name, "proposed_cascade", cp, cr, cf])
            print(f"  {name} Cascade : P={cp:.4f}, R={cr:.4f}, F1={cf:.4f}")
            
    print(f"Semi-synthetic validation complete. Results written to {out_csv}.")

if __name__ == "__main__":
    main()
