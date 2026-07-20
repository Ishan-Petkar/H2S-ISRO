"""Crops the official DFSAR water ice mask around Shackleton crater (0,0) projection center."""
import rasterio
import numpy as np
import os

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return

    with rasterio.open(mask_path) as src:
        # The center of projection (0,0) is at pixel coords (6100, 6100)
        # Let's crop a 400x400 patch (10km x 10km at 25m spacing) centered around (6100, 6100)
        row_start = 6100 - 200
        col_start = 6100 - 200
        
        window = rasterio.windows.Window(col_start, row_start, 400, 400)
        crop = src.read(1, window=window)
        
        # Check if there are water ice pixels in the crop
        ice_pixels = (crop == 1.0).sum()
        total_pixels = crop.size
        pct = (ice_pixels / total_pixels) * 100
        print(f"=== Shackleton Crater Mask Crop ===")
        print(f"Crop shape: {crop.shape}")
        print(f"Ice pixels: {ice_pixels} / {total_pixels} ({pct:.4f}%)")
        
        # Save the crop
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "shackleton_official_ice_mask_crop.npy")
        np.save(out_path, crop)
        print(f"Saved cropped mask to {out_path}")

if __name__ == "__main__":
    main()
