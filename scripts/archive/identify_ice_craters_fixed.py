"""Identifies which lunar south pole craters contain ice pixels in the official DFSAR mask (using fixed polar distances)."""
import rasterio
import numpy as np
import os

# Coordinates of known South Pole craters (latitude, longitude)
CRATERS = {
    "Cabeus": (-84.9, -35.5),
    "Faustini": (-87.3, 77.0),
    "Shoemaker": (-88.1, 44.9),
    "Haworth": (-87.4, -4.0),
    "Amundsen": (-84.5, 82.8),
    "de Gerlache": (-88.5, -87.1),
    "Sverdrup": (-88.5, -152.0),
    "Nobile": (-85.3, 53.5),
    "Shackleton": (-89.9, 0.0),
}

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
        
        # Take a 10% random sample for classification
        rng = np.random.default_rng(42)
        sample_size = min(40000, num_ice_pixels)
        indices = rng.choice(num_ice_pixels, size=sample_size, replace=False)
        sample_rows = rows[indices]
        sample_cols = cols[indices]
        
        # Convert sample pixels to lat/lon
        latlons = []
        for r, c in zip(sample_rows, sample_cols):
            x, y = src.xy(r, c)
            lat, lon = proj_to_latlon(x, y)
            latlons.append((lat, lon))
            
        latlons = np.array(latlons)
        
        # Classify each point using great-circle distance approximation
        crater_hits = {name: 0 for name in CRATERS}
        unclassified = 0
        
        # Craters have radii around 15-40 km, which is about 0.5 to 1.3 degrees at lunar scale (30.3 km per degree)
        # We use 1.0 degree of great-circle distance as the classification radius
        class_radius_deg = 1.0
        
        for lat, lon in latlons:
            found = False
            for name, coords in CRATERS.items():
                c_lat, c_lon = coords
                d_lat = lat - c_lat
                d_lon = (lon - c_lon + 180) % 360 - 180
                
                # Scale longitude difference by cosine of latitude to get physical/great-circle distance
                d_lon_scaled = d_lon * np.cos(np.radians(lat))
                dist = np.sqrt(d_lat**2 + d_lon_scaled**2)
                
                if dist <= class_radius_deg:
                    crater_hits[name] += 1
                    found = True
                    break
            if not found:
                unclassified += 1
                
        print("\n=== Fixed Ice Pixels Distribution per Crater (based on 10% sample) ===")
        scale = num_ice_pixels / len(indices)
        for name, hits in sorted(crater_hits.items(), key=lambda item: item[1], reverse=True):
            est_pixels = int(hits * scale)
            pct = (hits / len(indices)) * 100
            coords = CRATERS[name]
            print(f"  {name:15s} (at {coords[0]:.1f}°S, {coords[1]:.1f}°E): {est_pixels:6d} pixels ({pct:5.2f}%)")
        
        est_unclassified = int(unclassified * scale)
        print(f"  Other/Unclassified                       : {est_unclassified:6d} pixels ({unclassified/len(indices)*100:5.2f}%)")

if __name__ == "__main__":
    main()
