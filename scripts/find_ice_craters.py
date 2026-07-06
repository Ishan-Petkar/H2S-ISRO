"""Finds the locations of detected water ice pixels in the official ISRO DFSAR mask."""
import rasterio
import numpy as np
import os

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        # Find indices of all pixels with value 1.0
        rows, cols = np.where(mask == 1.0)
        num_ice_pixels = len(rows)
        print(f"Total water ice pixels found: {num_ice_pixels}")
        
        if num_ice_pixels == 0:
            return
            
        # Get projection coordinates for the first 10 pixels
        print("\nFirst 20 ice-bearing pixel coordinates (pixel row/col -> projection x/y):")
        for i in range(min(20, num_ice_pixels)):
            r, c = rows[i], cols[i]
            x, y = src.xy(r, c)
            # Standard conversion: polar stereographic coordinates x,y can be converted to lat,lon
            # using standard equations, or we can just show x,y.
            print(f"  Pixel ({r:5d}, {c:5d}) -> Proj (x={x:10.1f}, y={y:10.1f} meters)")

        # Let's find the bounding box of ice pixels
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        print(f"\nIce pixels bounding box in pixel coordinates:")
        print(f"  Row range: {min_r} to {max_r}")
        print(f"  Col range: {min_c} to {max_c}")
        
        # Convert bounding box to projection coordinates
        x_min, y_max = src.xy(min_r, min_c)
        x_max, y_min = src.xy(max_r, max_c)
        print(f"Ice pixels bounding box in projection coordinates:")
        print(f"  X range: {x_min:.1f} to {x_max:.1f} meters")
        print(f"  Y range: {y_min:.1f} to {y_max:.1f} meters")

if __name__ == "__main__":
    main()
