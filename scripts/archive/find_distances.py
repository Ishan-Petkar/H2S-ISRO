"""Computes distance of sample points to known craters to see why they are unclassified."""
import numpy as np

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

samples = [
    (-84.07, -130.51),
    (-87.47, -166.71),
    (-87.41, -133.92),
    (-88.09, 91.30),
    (-83.93, 67.94)
]

for lat, lon in samples:
    print(f"\nPoint ({lat} S, {lon} E):")
    for name, coords in CRATERS.items():
        c_lat, c_lon = coords
        d_lat = lat - c_lat
        d_lon = (lon - c_lon + 180) % 360 - 180
        dist = np.sqrt(d_lat**2 + d_lon**2)
        print(f"  to {name:12s}: dist = {dist:5.2f} degrees")
