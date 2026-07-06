"""Clusters the water ice pixels in the official ISRO DFSAR mask to find major ice deposits."""
import rasterio
import numpy as np
import os
from sklearn.cluster import DBSCAN

def proj_to_latlon(x, y, R=1737400.0):
    rho = np.sqrt(x**2 + y**2)
    if rho == 0:
        return -90.0, 0.0
    c = 2 * np.arctan(rho / (2.0 * R))
    lat = -90.0 + np.degrees(c)
    lon = np.degrees(np.arctan2(x, -y))
    return lat, lon

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        rows, cols = np.where(mask == 1.0)
        num_ice_pixels = len(rows)
        if num_ice_pixels == 0:
            print("No ice pixels found.")
            return

        print(f"Total ice pixels: {num_ice_pixels}")
        
        # Take a 2% sample to make clustering fast
        rng = np.random.default_rng(42)
        sample_size = min(10000, num_ice_pixels)
        indices = rng.choice(num_ice_pixels, size=sample_size, replace=False)
        sample_rows = rows[indices]
        sample_cols = cols[indices]
        
        # Convert to projection coordinates (x, y) in kilometers
        pts_km = []
        for r, c in zip(sample_rows, sample_cols):
            x, y = src.xy(r, c)
            pts_km.append((x / 1000.0, y / 1000.0))
        pts_km = np.array(pts_km)
        
        # Run DBSCAN in kilometers (epsilon = 5 km, min_samples = 20)
        print("Clustering ice deposits (DBSCAN)...")
        labels = DBSCAN(eps=5.0, min_samples=20).fit_predict(pts_km)
        
        unique_labels = set(labels)
        print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} major ice deposits.")
        
        cluster_info = []
        for lbl in unique_labels:
            if lbl == -1:
                continue
            cluster_pts = pts_km[labels == lbl]
            centroid_x = cluster_pts[:, 0].mean() * 1000.0
            centroid_y = cluster_pts[:, 1].mean() * 1000.0
            lat, lon = proj_to_latlon(centroid_x, centroid_y)
            
            # Estimate total pixels in this cluster (re-scale from 2% sample)
            est_pixels = int(len(cluster_pts) * (num_ice_pixels / sample_size))
            est_area_km2 = est_pixels * (25.0 * 25.0) / 1e6 # 25m resolution
            
            cluster_info.append((lbl, lat, lon, est_pixels, est_area_km2))
            
        # Sort by size
        cluster_info.sort(key=lambda x: x[3], reverse=True)
        
        print("\nMajor Ice Deposits Identified:")
        for idx, (lbl, lat, lon, size, area) in enumerate(cluster_info[:15]):
            print(f"  Deposit {idx+1:2d} -> Centroid: ({lat:6.2f}°S, {lon:7.2f}°E) | Est. Size: {size:6d} pixels | Est. Area: {area:5.1f} km²")

if __name__ == "__main__":
    main()
