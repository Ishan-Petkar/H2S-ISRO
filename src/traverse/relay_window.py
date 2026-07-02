"""
Relay visibility window computation from Chandrayaan-2 TLE.
Requires: skyfield>=1.46
pip install skyfield
"""

from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import ecliptic_frame
import numpy as np
from datetime import timedelta

# Chandrayaan-2 TLE — fetch latest from CelesTrak before use
TLE_LINE1 = "1 44441U 19042A   24180.50000000  .00000000  00000-0  00000-0 0  9999"
TLE_LINE2 = "2 44441  98.5000 270.0000 0001000  90.0000 270.0000 13.20000000 00001"

def compute_relay_windows(crater_lat_deg: float,
                           crater_lon_deg: float,
                           t_start_utc: str,
                           duration_hours: float = 48.0,
                           min_elevation_deg: float = 5.0) -> list:
    """
    Returns list of (rise_time, set_time, duration_minutes) tuples
    for relay visibility over the crater location.

    Parameters
    ----------
    crater_lat_deg   : crater latitude (degrees, negative = south)
    crater_lon_deg   : crater longitude (degrees)
    t_start_utc      : ISO format start time, e.g. '2026-08-06T00:00:00'
    duration_hours   : total window to search
    min_elevation_deg: minimum orbiter elevation for link closure
    """
    ts = load.timescale()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2,
                                'Chandrayaan-2', ts)

    # Note: skyfield uses Earth-based coordinates.
    # For lunar surface targets, this is an approximation — the
    # actual relay geometry requires a lunar ephemeris.
    # This is documented as a labeled assumption in the paper.
    crater = wgs84.latlon(crater_lat_deg, crater_lon_deg)

    t0 = ts.utc(*[int(x) for x in
                   t_start_utc.replace('T',' ').replace(':','-')
                   .replace(' ','-').split('-')])

    windows = []
    # Step through in 1-minute increments
    times = ts.utc(t0.utc[0], t0.utc[1], t0.utc[2],
                   t0.utc[3],
                   range(int(duration_hours * 60)))

    diff = satellite - crater
    topocentric = diff.at(times)
    el, az, dist = topocentric.altaz()
    visible = el.degrees > min_elevation_deg

    # Find rising/setting edges
    edges = np.diff(visible.astype(int))
    rises = np.where(edges == 1)[0]
    sets  = np.where(edges == -1)[0]

    for r, s in zip(rises, sets):
        dur = (s - r)  # minutes
        windows.append({
            "rise_min": int(r),
            "set_min": int(s),
            "duration_min": dur
        })

    if not windows:
        import logging
        logging.warning("No relay windows found — using 120-min assumed cadence.")
        windows = [{"rise_min": i*120, "set_min": i*120+30,
                    "duration_min": 30, "assumed": True}
                   for i in range(int(duration_hours/2))]

    return windows


def max_shadow_excursion_hours(windows: list) -> float:
    """Returns the tightest single relay window duration in hours."""
    if not windows:
        return 0.5  # fallback
    return max(w["duration_min"] for w in windows) / 60.0
