"""
ML-focused pattern guidance (avoid / encourage). Inspired by CodeLearn llm_suggestions.
Used by oracle_suggest_nl and /guidance API.
"""
from typing import Dict, Any, List

# Pattern -> description, suggestion, why (for NL oracle and API)
PATTERN_GUIDANCE: Dict[str, Dict[str, str]] = {
    "overfitting": {
        "description": "Model fits training data too closely; poor generalization to new data.",
        "suggestion": "Use stronger regularization (L1/L2), early stopping, more data, or reduce model capacity.",
        "why": "Bias-variance tradeoff: high variance from fitting noise.",
    },
    "underfitting": {
        "description": "Model too simple; high bias, cannot capture signal in data.",
        "suggestion": "Increase model capacity, add features, or reduce regularization.",
        "why": "Bishop: bias dominates; model is not expressive enough.",
    },
    "data_leakage": {
        "description": "Information from test or future leaks into training (e.g. scaling on full dataset).",
        "suggestion": "Fit scalers/transformers on training fold only; use Pipeline and cross_validate.",
        "why": "Leakage inflates metrics and causes production failures.",
    },
    "unbalanced_classes": {
        "description": "Classification with highly imbalanced class frequencies.",
        "suggestion": "Use class_weight, oversample minority (SMOTE), or adjust decision threshold; consider precision-recall.",
        "why": "Accuracy is misleading; minority class matters for detection.",
    },
    "high_capacity_low_data": {
        "description": "Using a high-capacity model (e.g. deep net) with small sample size.",
        "suggestion": "Prefer regularized linear models or small ensembles; use cross-validation and early stopping.",
        "why": "Capacity without data leads to overfitting (Shannon: limited data limits reliable complexity).",
    },
    "no_validation_set": {
        "description": "Evaluating or tuning on the same data used for training.",
        "suggestion": "Hold out a validation set or use cross-validation; never tune on test.",
        "why": "Theory-as-channel: need independent signal to estimate generalization.",
    },
    "ignoring_baseline": {
        "description": "Not comparing against a simple baseline (e.g. mean, majority class).",
        "suggestion": "Always report baseline accuracy or RMSE; ensure the model beats it.",
        "why": "Skiena: reduce then solve; baseline is the simplest reducer.",
    },
    "ensemble_small_n": {
        "description": "Using many models in an ensemble when data is limited.",
        "suggestion": "Use 3–5 diverse models with majority vote; avoid 10+ models on small n.",
        "why": "Channel capacity: redundancy helps but too many weak learners add noise.",
    },
    "tabular_default": {
        "description": "Tabular classification or regression without problem-specific tuning.",
        "suggestion": "Start with RandomForest or LogisticRegression; try gradient boosting for larger n.",
        "why": "Robust defaults reduce variance; iterate based on validation.",
    },
    "text_safety": {
        "description": "Text input with safety or compliance requirements.",
        "suggestion": "Preprocess with toolbox.data.preprocess(..., advanced=True); apply safety filter before modeling.",
        "why": "Skiena: reduce then solve; compartmentalize safety.",
    },
}


def get_rules_for_nl_oracle() -> List[Dict[str, Any]]:
    """Return list of {description, pattern, suggestion, why} for oracle_suggest_nl."""
    return [
        {
            "description": v["description"],
            "pattern": k,
            "suggestion": v["suggestion"],
            "why": v["why"],
        }
        for k, v in PATTERN_GUIDANCE.items()
    ]


def get_avoid_encourage(minimal: bool = False) -> Dict[str, Any]:
    """
    Return avoid / encourage lists for /guidance API.
    avoid: patterns to avoid with short reason; encourage: patterns to prefer.
    """
    avoid = [
        {"pattern": k.replace("_", " "), "reason": v["suggestion"]}
        for k, v in PATTERN_GUIDANCE.items()
        if k in ("overfitting", "data_leakage", "no_validation_set", "high_capacity_low_data", "ignoring_baseline")
    ]
    encourage = [
        {"pattern": k.replace("_", " "), "reason": v["suggestion"]}
        for k, v in PATTERN_GUIDANCE.items()
        if k in ("tabular_default", "ensemble_small_n", "unbalanced_classes")
    ]
    out = {"avoid": avoid, "encourage": encourage}
    if not minimal:
        out["all_patterns"] = list(PATTERN_GUIDANCE.keys())
        out["risky_patterns"] = [k for k in PATTERN_GUIDANCE if k in ("overfitting", "data_leakage", "no_validation_set")]
    return out
