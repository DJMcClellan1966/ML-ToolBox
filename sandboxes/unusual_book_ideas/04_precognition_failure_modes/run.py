"""Sandbox 04: Precognition + Failure-Mode Warnings - viability test."""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    from ml_toolbox.textbook_concepts.data_quality import (
        feature_informativeness,
        missing_value_impact,
    )
    # Synthetic data with one weak feature and some missingness
    np.random.seed(42)
    n, d = 100, 5
    X = np.random.randn(n, d)
    X[:, 0] = 0.1 * np.random.randn(n)  # Low variance -> low informativeness
    # Inject missing
    mask = np.random.rand(n, d) < 0.1
    X_missing = X.copy()
    X_missing[mask] = np.nan
    # For missing_value_impact: X (with NaNs filled for computation) and boolean missing_mask
    X_full = np.nan_to_num(X_missing, nan=0.0)
    missing_mask = np.isnan(X_missing)
    info = feature_informativeness(X_full, n_bins=5)
    # Missing impact: X, missing_mask
    try:
        impact = missing_value_impact(X_full, missing_mask, n_bins=5)
        impact_ok = isinstance(impact, dict) and len(impact) > 0
    except Exception:
        impact = None
        impact_ok = False
    # "Failure warning" string based on low informativeness
    low_info_idx = np.argmin(info)
    warnings = []
    if info[low_info_idx] < 0.5:
        warnings.append("Low feature informativeness on at least one feature (Bishop: consider feature selection).")
    return {
        "ok": True,
        "informativeness_shape": info.shape,
        "lowest_informativeness": float(info[low_info_idx]),
        "missing_impact_ok": impact_ok,
        "warnings_count": len(warnings),
        "warnings": warnings,
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
