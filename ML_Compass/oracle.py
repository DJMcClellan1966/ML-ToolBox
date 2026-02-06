"""
Algorithm Oracle: problem profile -> pattern, suggestion, why.
Supports profile-based rules and NL description via pattern_guidance + quantum_enhancements.
"""
from typing import Dict, Any, List, Optional

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


def suggest_from_description(description: str) -> Dict[str, Any]:
    """
    Oracle path from natural language: match description to ML pattern guidance.
    Uses quantum_enhancements.oracle_suggest_nl when kernel available; else keyword fallback.

    Args:
        description: e.g. "My model memorizes the training set" or "I have 1000 rows and a big neural net"

    Returns:
        {"ok": bool, "pattern": str, "suggestion": str, "why": str, "source": str}
    """
    from .pattern_guidance import get_rules_for_nl_oracle
    rules = get_rules_for_nl_oracle()
    try:
        from .quantum_enhancements import oracle_suggest_nl
        out = oracle_suggest_nl(description, rules)
        if out is not None:
            return {**out, "source": out.get("source", "nl_match")}
    except Exception:
        pass
    # Keyword fallback: find first rule whose description words appear in description
    desc_lower = description.lower()
    for r in rules:
        words = set(r["description"].lower().split())
        if any(w in desc_lower for w in words if len(w) > 4):
            return {
                "ok": True,
                "pattern": r["pattern"],
                "suggestion": r["suggestion"],
                "why": r["why"],
                "source": "keyword_fallback",
            }
    return {
        "ok": False,
        "pattern": None,
        "suggestion": DEFAULT_SUGGESTION,
        "why": DEFAULT_WHY,
        "source": "no_match",
    }
