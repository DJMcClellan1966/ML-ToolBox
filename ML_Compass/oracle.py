"""
Algorithm Oracle: problem profile -> pattern, suggestion, why.
"""
from typing import Dict, Any, List

# Rule table: problem profile -> (pattern, suggestion, why)
ORACLE_RULES: List[Dict[str, Any]] = [
    {
        "profile": {"tabular": True, "classification": True, "n_samples": "small"},
        "pattern": "Classification (small n)",
        "suggestion": "LogisticRegression or RandomForest with default regularization.",
        "why": "Small n favors regularized models; avoid overfitting.",
    },
    {
        "profile": {"tabular": True, "classification": True, "n_samples": "medium"},
        "pattern": "Classification (medium n)",
        "suggestion": "RandomForest or SVC; consider ensemble of 3-5 models with majority vote.",
        "why": "Ensemble reduces variance; theory-as-channel suggests 3-5 models with majority vote.",
    },
    {
        "profile": {"tabular": True, "classification": True, "n_samples": "large"},
        "pattern": "Classification (large n)",
        "suggestion": "Gradient boosting or deep model; ensemble still helps for robustness.",
        "why": "Enough data for more capacity; ensemble corrects errors (Shannon).",
    },
    {
        "profile": {"high_dim": True, "classification": True},
        "pattern": "High-dim classification",
        "suggestion": "PCA or mutual-information feature selection + LogisticRegression or RandomForest.",
        "why": "Bishop: bias-variance; reduce dimensions or regularize.",
    },
    {
        "profile": {"text": True, "need_safety": True, "high_volume": True},
        "pattern": "Preprocess-then-classify (Skiena: reduce then solve)",
        "suggestion": "Use toolbox.data.preprocess(..., advanced=True) with safety filter; then classifier.",
        "why": "Text + safety implies preprocessing compartment; high volume benefits from dedup.",
    },
    {
        "profile": {"unsupervised": True, "cluster_shape": "unknown"},
        "pattern": "Density-based or hierarchical",
        "suggestion": "Try DBSCAN or hierarchical clustering; use silhouette to compare.",
        "why": "Unknown k and shape suggest DBSCAN or dendrogram inspection.",
    },
]

DEFAULT_SUGGESTION = "No match; use default: RandomForest or LogisticRegression."
DEFAULT_WHY = "Fallback when profile does not match rules."


def suggest(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a problem profile, return pattern, suggestion, and why.

    Args:
        profile: e.g. {"tabular": True, "classification": True, "n_samples": "medium"}

    Returns:
        {"ok": bool, "pattern": str, "suggestion": str, "why": str, "profile_used": dict}
    """
    for r in ORACLE_RULES:
        if r["profile"] == profile:
            return {
                "ok": True,
                "pattern": r["pattern"],
                "suggestion": r["suggestion"],
                "why": r["why"],
                "profile_used": profile,
            }
    return {
        "ok": False,
        "pattern": None,
        "suggestion": DEFAULT_SUGGESTION,
        "why": DEFAULT_WHY,
        "profile_used": profile,
    }
