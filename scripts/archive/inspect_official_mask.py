"""Inspects the official ISRO DFSAR polar water ice binary mask GeoTIFF."""
import rasterio
import numpy as np
import os

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return

    with rasterio.open(mask_path) as src:
        print("=== Official DFSAR Polar Water Ice Mask (South Pole) ===")
        print(f"Driver: {src.driver}")
        print(f"Width: {src.width}, Height: {src.height}")
        print(f"Bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform:\n{src.transform}")
        print(f"Bounding Box: {src.bounds}")
        
        # Read the mask
        mask = src.read(1)
        unique, counts = np.unique(mask, return_counts=True)
        stats = dict(zip(unique, counts))
        print("Pixel counts:")
        for val, count in stats.items():
            percentage = (count / mask.size) * 100
            print(f"  Value {val}: {count} pixels ({percentage:.4f}%)")

if __name__ == "__main__":
    main()
