"""Unit tests — H2S-ISRO pipeline results verification."""
import csv, math, sys, os

ROUGHNESS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "roughness_ablation.csv")
DBSCAN_CSV    = os.path.join(os.path.dirname(__file__), "..", "results", "dbscan_ablation.csv")

def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def test_roughness_peak_f1():
    rows = _load(ROUGHNESS_CSV)
    f1_at_2_5 = float([r for r in rows
                        if float(r["threshold_sigma"]) == 2.50][0]["f1"])
    assert f1_at_2_5 >= 0.560, f"F1 at 2.5σ = {f1_at_2_5}, expected ≥ 0.560"

def test_dbscan_best_config():
    rows = _load(DBSCAN_CSV)
    best = max(rows, key=lambda r: float(r["f1"]))
    assert float(best["epsilon_m"]) in [30.0, 60.0], \
        f"Best ε unexpected: {best}"
    assert float(best["f1"]) >= 0.560

def test_precision_floor():
    """Precision must always exceed 0.95 for the default ε=30, minPts=4."""
    rows = _load(DBSCAN_CSV)
    row = [r for r in rows
           if float(r["epsilon_m"]) == 30.0
           and int(r["min_pts"]) == 4][0]
    assert float(row["precision"]) >= 0.95

if __name__ == "__main__":
    tests = [test_roughness_peak_f1, test_dbscan_best_config,
             test_precision_floor]
    failed = []
    for t in tests:
        try: 
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e: 
            failed.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {e}")
    if failed: 
        sys.exit(1)
    print("All tests passed.")
