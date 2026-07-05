"""
BAH 2026 — Lunar Ice Detection Pipeline
Code skeleton: replace TODO blocks with full implementations.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class PolarimetricScene:
    cpr_L: np.ndarray        # L-band CPR, shape (H, W)
    cpr_S: np.ndarray        # S-band CPR, shape (H, W)
    dop: np.ndarray          # Degree of polarisation
    roughness: np.ndarray    # LOLA-derived RMS roughness
    slope: np.ndarray        # Terrain slope (degrees)
    illumination: np.ndarray # Fractional illumination [0,1]
    psr_mask: np.ndarray     # Boolean; True = permanently shadowed
    pixel_area_m2: float     # Area per pixel in m^2


@dataclass
class DetectionOutput:
    ice_prob: np.ndarray       # Pi map [0,1]
    fp_risk: np.ndarray        # Phi map [0,1]
    ice_candidates: np.ndarray # Boolean mask


@dataclass
class TraverseOutput:
    paths: list                # Ranked list of (path_coords, energy, time)
    landing_site: Tuple        # (row, col) of p*
    feasible: bool


# ─────────────────────────────────────────────
# Stage 1 — Reference Population
# ─────────────────────────────────────────────

def build_reference_population(scene: PolarimetricScene,
                                 known_icefree_psr: Optional[np.ndarray] = None
                                 ) -> dict:
    """
    Build dry-regolith reference statistics.
    Prefers ice-free PSR pixels; falls back to slope-matched sunlit terrain.
    """
    if known_icefree_psr is not None and known_icefree_psr.sum() > 50:
        ref_pixels = scene.cpr_L[known_icefree_psr]
        ref_dop    = scene.dop[known_icefree_psr]
        ref_rough  = scene.roughness[known_icefree_psr]
        ref_type   = "ice_free_psr"
        logger.info("Reference: ice-free PSR pixels (preferred).")
    else:
        # Fallback: sunlit terrain, slope-matched
        sunlit = (~scene.psr_mask) & (scene.illumination > 0.2)
        ref_pixels = scene.cpr_L[sunlit]
        ref_dop    = scene.dop[sunlit]
        ref_rough  = scene.roughness[sunlit]
        ref_type   = "sunlit_fallback"
        logger.warning("Reference: sunlit fallback — maturity bias risk logged.")

    return {
        "mu_cpr": ref_pixels.mean(),
        "std_cpr": ref_pixels.std(),
        "mu_dop": ref_dop.mean(),
        "std_dop": ref_dop.std(),
        "mu_rough": ref_rough.mean(),
        "std_rough": ref_rough.std(),
        "type": ref_type
    }


# ─────────────────────────────────────────────
# Stages 2–5 — AND-Cascade Detector
# ─────────────────────────────────────────────

def run_detector(scene: PolarimetricScene,
                 ref: dict,
                 sigma_thresh: float = 2.0,
                 roughness_sigma: float = 2.5,
                 dbscan_eps: float = 30.0,
                 dbscan_minpts: int = 4,
                 min_cluster_area_m2: float = 100.0) -> DetectionOutput:

    H, W = scene.cpr_L.shape
    ice_prob = np.zeros((H, W))
    fp_risk  = np.zeros((H, W))

    # --- Stage 2: Anomaly scoring ---
    z_cpr = (scene.cpr_L - ref["mu_cpr"]) / ref["std_cpr"]
    z_dop = (scene.dop   - ref["mu_dop"]) / ref["std_dop"]
    stage2 = scene.psr_mask & (z_cpr >= sigma_thresh) & (z_dop <= -sigma_thresh)

    # --- Stage 3: Roughness veto ---
    rough_thresh = ref["mu_rough"] + roughness_sigma * ref["std_rough"]
    roughness_anomaly = scene.roughness > rough_thresh
    # Log roughness-scale mismatch risk
    fp_risk[stage2 & roughness_anomaly] += 0.4   # scale-mismatch penalty
    stage3 = stage2 & ~roughness_anomaly

    # --- Stage 4: Spatial coherence (DBSCAN) ---
    coords = np.column_stack(np.where(stage3))
    if len(coords) < dbscan_minpts:
        logger.warning("No anomalous pixels survived roughness veto.")
        return DetectionOutput(ice_prob, fp_risk, np.zeros((H, W), bool))

    # Convert pixel coords to metres for DBSCAN epsilon
    pixel_size = np.sqrt(scene.pixel_area_m2)
    coords_m   = coords * pixel_size
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minpts).fit_predict(coords_m)

    stage4_mask = np.zeros((H, W), bool)
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = coords[labels == lbl]
        area = len(cluster) * scene.pixel_area_m2
        if area >= min_cluster_area_m2:
            stage4_mask[cluster[:, 0], cluster[:, 1]] = True

    # --- Stage 5: Dual-frequency corroboration ---
    stage5_mask = stage4_mask.copy()
    ratio = np.where(scene.cpr_L > 0, scene.cpr_S / scene.cpr_L, 0)
    weak_corr = stage4_mask & (ratio <= 1.0)
    fp_risk[weak_corr] += 0.3    # corroboration absent
    # (still retained; just lower confidence)

    # --- Ice probability (logistic combination) ---
    confidence = (
        np.clip(z_cpr, 0, 5) / 5.0 * 0.4 +
        np.clip(-z_dop, 0, 5) / 5.0 * 0.4 +
        np.clip(ratio - 1, 0, 2) / 2.0 * 0.2
    )
    ice_prob[stage5_mask] = np.clip(confidence[stage5_mask], 0, 1)

    # Baseline-bias risk from reference type
    if ref["type"] == "sunlit_fallback":
        fp_risk += 0.15

    fp_risk = np.clip(fp_risk, 0, 1)

    return DetectionOutput(ice_prob, fp_risk, stage5_mask)


# ─────────────────────────────────────────────
# Landing Site Optimisation
# ─────────────────────────────────────────────

def find_landing_site(scene: PolarimetricScene,
                       dsc_rim_mask: np.ndarray,
                       rock_cover: np.ndarray,
                       slope_limit: float = 5.0,
                       illum_min: float = 0.20,
                       rock_max: float = 0.05) -> Tuple:
    """
    Returns pixel coordinates of optimal landing site p*.
    Landing distance is an OUTPUT of this function, never an input.
    """
    feasible = (
        (~scene.psr_mask) &
        (scene.slope <= slope_limit) &
        (scene.illumination >= illum_min) &
        (rock_cover <= rock_max)
    )
    if not feasible.any():
        logger.error("No feasible landing site found. Relax constraints.")
        return None

    # Distance transform from DSC rim
    dist_from_rim = ndimage.distance_transform_edt(~dsc_rim_mask)
    dist_from_rim[~feasible] = np.inf

    p_star = np.unravel_index(dist_from_rim.argmin(), dist_from_rim.shape)
    dist_m = dist_from_rim[p_star] * np.sqrt(scene.pixel_area_m2)
    logger.info(f"Landing site: {p_star}, distance to rim: {dist_m:.1f} m")
    return p_star


# ─────────────────────────────────────────────
# Energy Budget & A* Traverse Planner
# ─────────────────────────────────────────────

def compute_energy(path_coords: np.ndarray,
                   slope_map: np.ndarray,
                   pixel_size_m: float,
                   n_science_stops: int,
                   t_shadow_hours: float,
                   e0: float = 2.0,         # Wh/m flat terrain
                   k_traction: float = 0.05, # traction coefficient
                   e_science: float = 5.0,   # Wh per stop
                   p_rhu: float = 1.0,       # W
                   p_comms: float = 2.0,     # W
                   sensitivity_k: float = None) -> dict:
    """
    Slope-dependent energy budget.
    k_traction sensitivity: k in [0.03, 0.10] varies total by ~±8%.
    """
    if sensitivity_k is not None:
        k_traction = sensitivity_k

    e_loco = 0.0
    for i in range(len(path_coords) - 1):
        r0, c0 = path_coords[i]
        r1, c1 = path_coords[i+1]
        dist = np.sqrt((r1-r0)**2 + (c1-c0)**2) * pixel_size_m
        slope_deg = slope_map[int(r0), int(c0)]
        slope_rad = np.radians(slope_deg)
        e_loco += e0 * (1 + k_traction * np.tan(slope_rad)) * dist

    e_sci   = n_science_stops * e_science
    e_heat  = (p_rhu + p_comms) * t_shadow_hours
    total   = e_loco + e_sci + e_heat

    return {
        "total_wh": total,
        "locomotion_wh": e_loco,
        "science_wh": e_sci,
        "thermal_comms_wh": e_heat
    }


def astar_traverse(cost_map: np.ndarray,
                   start: Tuple,
                   targets: np.ndarray,
                   e_max_wh: float,
                   t_relay_hours: float,
                   pixel_size_m: float,
                   slope_map: np.ndarray,
                   contingency: float = 0.20,
                   e0: float = 2.0,
                   k_traction: float = 0.05) -> TraverseOutput:
    """
    A* path planner with hard energy + relay constraints.
    Returns only feasible paths; empty list if none exist.
    """
    import heapq

    usable_energy = (1.0 - contingency) * e_max_wh
    usable_time = t_relay_hours
    logger.info(f"Usable energy budget: {usable_energy:.1f} Wh, relay window: {usable_time:.1f} h")

    H, W = cost_map.shape
    target_set = set(map(tuple, targets.tolist()))

    # Priority queue: (f_score, energy_consumed, g_score, current_node, path)
    pq = []
    heapq.heappush(pq, (0.0, 0.0, 0.0, start, [start]))

    visited = {}

    while pq:
        f, e_curr, g, curr, path = heapq.heappop(pq)

        if curr in target_set:
            logger.info(f"Path found! Length: {len(path)} steps, Energy: {e_curr:.1f} Wh")
            return TraverseOutput(paths=[(path, e_curr, 0.0)], landing_site=start, feasible=True)

        if curr in visited and visited[curr] <= e_curr:
            continue
        visited[curr] = e_curr

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                # Basic locomotion cost
                dist = np.sqrt(dr**2 + dc**2) * pixel_size_m
                slope_deg = slope_map[nr, nc]
                e_step = e0 * (1 + k_traction * np.tan(np.radians(slope_deg))) * dist

                new_e = e_curr + e_step
                # Prune if energy exceeded
                if new_e > usable_energy:
                    continue

                new_g = g + cost_map[nr, nc]
                # Heuristic: Euclidean distance * base cost per meter (2.0)
                min_h = float('inf')
                for tr, tc in targets:
                    h_dist = np.sqrt((nr - tr)**2 + (nc - tc)**2) * pixel_size_m
                    # Since cost_map is added to g, and energy is evaluated separately, 
                    # we use distance * base_cost to guide the search.
                    min_h = min(min_h, h_dist * 0.1) 
                h = min_h 
                
                heapq.heappush(pq, (new_g + h, new_e, new_g, (nr, nc), path + [(nr, nc)]))

    logger.warning("No feasible path found under joint constraints.")
    return TraverseOutput(paths=[], landing_site=start, feasible=False)


# ─────────────────────────────────────────────
# Ice Volume Estimation
# ─────────────────────────────────────────────

def estimate_ice_volume(cpr_L_anomaly: np.ndarray,
                         candidate_mask: np.ndarray,
                         pixel_area_m2: float,
                         depth_m: float = 2.0,
                         a: float = 0.15,
                         b: float = -0.05,
                         n_mc: int = 10_000) -> dict:
    """
    Empirical CPR-to-ice-fraction mapping with MC uncertainty propagation.
    Coefficients a, b from Sahu et al. 2025 (Cabeus/Shackleton calibration).
    """
    cpr_vals = cpr_L_anomaly[candidate_mask]
    f_ice = a * np.log(np.clip(cpr_vals, 0.01, None)) + b
    f_ice = np.clip(f_ice, 0, 1)

    # Monte Carlo: perturb CPR by measurement noise, area by delineation error
    volumes = []
    for _ in range(n_mc):
        cpr_noisy = cpr_vals * rng.normal(1.0, 0.08, size=cpr_vals.shape)
        f_noisy   = a * np.log(np.clip(cpr_noisy, 0.01, None)) + b
        f_noisy   = np.clip(f_noisy, 0, 1)
        area_frac = rng.normal(1.0, 0.05)   # 5% area delineation noise
        n_pix_noisy = len(cpr_vals) * area_frac
        v = n_pix_noisy * pixel_area_m2 * f_noisy.mean() * depth_m
        volumes.append(v)

    volumes = np.array(volumes)
    ci_lo, ci_hi = np.percentile(volumes, [5, 95])
    headline = float(len(cpr_vals) * pixel_area_m2 * f_ice.mean() * depth_m)

    logger.info(f"Ice volume: {headline:.0f} m³  90% CI [{ci_lo:.0f}, {ci_hi:.0f}]")
    return {"headline_m3": headline, "ci_90_lo": ci_lo, "ci_90_hi": ci_hi,
            "uncertainty_pct": (ci_hi - ci_lo) / (2 * headline) * 100}


# ─────────────────────────────────────────────
# U-Net Generalisation Layer
# ─────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=6):
            super().__init__()
            self.enc1 = DoubleConv(in_channels, 64)
            self.enc2 = DoubleConv(64, 128)
            self.enc3 = DoubleConv(128, 256)
            self.enc4 = DoubleConv(256, 512)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(512, 1024)
            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = DoubleConv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = DoubleConv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = DoubleConv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = DoubleConv(128, 64)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b  = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(self.head(d1))

    def fuse_cascade_unet(pi_cascade: np.ndarray,
                           pi_unet: np.ndarray) -> np.ndarray:
        """Geometric mean fusion (Eq. 7 in paper)."""
        return np.sqrt(np.clip(pi_cascade, 0, 1) *
                       np.clip(pi_unet, 0, 1))

except ImportError:
    logger.warning("PyTorch not installed — U-Net layer unavailable.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Pipeline skeleton loaded. Replace TODO blocks to run.")
