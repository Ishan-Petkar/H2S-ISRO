"""Prints lat/lon of some ice pixels to inspect coordinate conversion."""
import rasterio
import numpy as np
import os

def proj_to_latlon(x, y, R=1737400.0):
    rho = np.sqrt(x**2 + y**2)
    c = 2 * np.arctan(rho / (2.0 * R))
    lat = -90.0 + np.degrees(c)
    lon = np.degrees(np.arctan2(x, -y))
    return lat, lon

def main():
    mask_path = os.path.join(os.path.dirname(__file__), "..", "data", "dfsar", "DFSAR_Water_Ice", "Polar_Water_Ice_Products", "ICY_CRATERS_SP.tif")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        rows, cols = np.where(mask == 1.0)
        num_ice_pixels = len(rows)
        print(f"Total ice pixels: {num_ice_pixels}")
        
        # Select 20 spread-out points
        step = max(1, num_ice_pixels // 20)
        for i in range(0, num_ice_pixels, step):
            r, c = rows[i], cols[i]
            x, y = src.xy(r, c)
            lat, lon = proj_to_latlon(x, y)
            print(f"Index {i:5d} -> Pixel ({r:5d}, {c:5d}) -> Proj ({x:9.1f}, {y:9.1f}) -> Lat/Lon ({lat:6.2f} S, {lon:7.2f} E)")

if __name__ == "__main__":
    main()
