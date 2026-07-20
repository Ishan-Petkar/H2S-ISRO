"""Crops the official DFSAR mask around a known active ice region near Sverdrup crater."""
import rasterio
import numpy as np
import os

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return

    with rasterio.open(mask_path) as src:
        # Known active point at row=3113, col=5394
        row_center, col_center = 3113, 5394
        
        # Crop 256x256 patch
        row_start = row_center - 128
        col_start = col_center - 128
        
        window = rasterio.windows.Window(col_start, row_start, 256, 256)
        crop = src.read(1, window=window)
        
        ice_pixels = (crop == 1.0).sum()
        total_pixels = crop.size
        pct = (ice_pixels / total_pixels) * 100
        print(f"=== Active Sverdrup Region Mask Crop ===")
        print(f"Crop shape: {crop.shape}")
        print(f"Ice pixels: {ice_pixels} / {total_pixels} ({pct:.4f}%)")
        
        if ice_pixels > 0:
            print("Found ice pixels inside active Sverdrup crop!")
            out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "sverdrup_active_ice_mask_crop.npy")
            np.save(out_path, crop)
            print(f"Saved cropped mask to {out_path}")

if __name__ == "__main__":
    main()
