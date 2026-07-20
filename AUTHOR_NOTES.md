# Author Notes — BAH 2026 Paper

## Assumptions Made

1. DFSAR Level-2 fully polarimetric products are available on PRADAN
   in standard SLC or GRD format compatible with MIDAS/SNAP.
   If only compact-polarimetric (CP) products are available, the m-chi
   decomposition path is unaffected but full Stokes DOP computation
   must be replaced with CP-approximate DOP.

2. An ice-free PSR reference region is assumed to exist within the
   same scene. If not, the sunlit fallback is used with explicit bias
   risk logged — this is disclosed in Known Unknowns item 1.

3. Relay orbital elements are assumed available in published form
   (Chandrayaan-2 orbiter TLE). If unavailable for the target DSC,
   a representative 2-hour visibility cadence for a polar orbiter
   is used as a stated assumption.

4. The A* planner is implemented in `pipeline_skeleton.py` (and executed via `src/planner_execution.py`) using a standard binary heap. This is fully functional and tested on real LOLA DEM crops.

## Open Design Decisions

- **U-Net layer**: The framework is purely physics-based. Adding a U-Net
  trained on anomaly-pipeline pseudo-labels would improve spatial
  generalisation across craters. Recommend implementing this after
  the proposal stage if shortlisted for the Grand Finale.

- **DBSCAN parameters**: eps=30m and minPts=4 are reasonable defaults
  but should be tuned via ablation (experiments/ablation_dbscan.py).

- **Energy budget constants**: e_loco=2 Wh/m, e_science=5 Wh/stop are
  illustrative. Replace with VIPER-class or Chandrayaan-3 rover specs
  if available.

## Recommended Next Steps

1. Register on PRADAN and confirm data product level available.
2. Run pipeline_skeleton.py Stages 1-4 on a small (~100x100 px) test patch.
3. Add your names, PRN, and college details to the LaTeX author block.
4. Check institutional formatting requirements for BAH submission template
   (the paper may need to match a specific Hack2Skill template rather
   than full IEEEtran --- use this LaTeX as the technical depth document
   and adapt to their template structure).
5. Run: pdflatex main.tex && bibtex main && pdflatex main.tex x2
   to compile. Requires texlive-full or MikTeX with IEEEtran class.

## Humanizing Resources Applied

- github.com/humanize/humanize — text humanization library; principles of
  making technical numbers readable in prose extended to uncertainty ranges.
- General academic writing principles: opening with a concrete narrative
  (the robot crossing into darkness), varied sentence length, explicit
  section signposting, empathetic framing of limitations in Known Unknowns.
