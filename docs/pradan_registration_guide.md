# Registration Guide for ISRO PRADAN L2 DFSAR Products

This guide outlines the manual and programmatic steps required to ingest and co-register Chandrayaan-2 DFSAR L2 data (downloaded from the ISRO PRADAN portal) with NASA LOLA DEMs. This ensures that the L-band CPR/DOP features align perfectly with the high-resolution topographic constraints required by the stackable-veto cascade and the traverse planner.

## 1. Data Procurement
1. Navigate to the [ISRO PRADAN / ISSDC portal](https://pradan.issdc.gov.in/).
2. Query for **Chandrayaan-2 Dual Frequency Synthetic Aperture Radar (DFSAR)** products over your target South Pole site (e.g., Shackleton or Cabeus).
3. Download the **Level-2 (L2) Hybrid-Polarimetric** products. These are provided as four standard Stokes parameter rasters ($S_1, S_2, S_3, S_4$).

## 2. Radiometric Feature Extraction
Use the provided `src/pradan_ingestion.py` script. The script calculates:
*   **CPR (Circular Polarization Ratio)**: $CPR = \frac{S_1 - S_4}{S_1 + S_4}$ (assuming standard LHCP transmit)
*   **DOP (Degree of Polarization)**: $DOP = \frac{\sqrt{S_2^2 + S_3^2 + S_4^2}}{S_1}$

*Note: Check the PDS label for the specific transmit polarization. If RHCP is used instead of LHCP, the CPR sign convention (numerator) flips to $S_1 + S_4$.*

## 3. Co-Registration with LOLA DEM
Because DFSAR and LOLA operate on distinct coordinate reference systems (or suffer from raw orbital ephemeris offsets), a rigid spatial shift is strictly required before applying the roughness veto.

1. **Load DEM**: Download the corresponding $5$\,m/px LOLA GeoTIFF from the PDS Planetary Geodynamics Data Archive (PGDA).
2. **Feature Matching**: Open both the DFSAR $S_1$ (total power/intensity) image and the LOLA DEM in a GIS platform (e.g., QGIS or ArcGIS).
3. **Crater Rim Alignment**: Identify 3 to 5 prominent, sharp crater rims visible in both the radar backscatter ($S_1$) and the altimetry DEM. Radar shadowing generally points away from the sensor, so align the leading illuminated edges.
4. **Warping**: Apply a linear translation (or an affine transformation if scale/rotation errors are present) to the DFSAR raster to align it precisely over the LOLA grid.
5. **Resampling**: Resample the DFSAR data using a nearest-neighbor filter (to strictly preserve the raw radiometric statistical distributions, avoiding bilinear smoothing) to match the exact pixel grid of the LOLA DEM ($5$\,m/px).

## 4. Pipeline Execution
Once the derived `cpr.tif`, `dop.tif`, and the LOLA `roughness.tif` and `slope.tif` are perfectly aligned, stacked, and cropped to identical dimensions, they can be directly loaded into the `PolarimetricScene` data structure inside `src/pipeline_skeleton.py` to run the anomaly detector.
