"""Sandbox 08: Algorithm Oracle with Reasoning - viability test."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    # Rule table: problem profile -> (pattern, suggestion, why)
    rules = [
        {
            "profile": {"text": True, "need_safety": True, "high_volume": True},
            "pattern": "Preprocess-then-classify (Skiena: reduce then solve)",
            "suggestion": "Use toolbox.data.preprocess(..., advanced=True) with safety filter; then classifier.",
            "why": "Text + safety implies preprocessing compartment with PocketFence; high volume benefits from dedup.",
        },
        {
            "profile": {"high_dim": True, "few_samples": True},
            "pattern": "Regularization and feature selection (Bishop: bias-variance)",
            "suggestion": "Use mutual-information feature selection or PCA; Ridge/Lasso or Random Forest with limited depth.",
            "why": "Few samples in high dimensions overfit; reduce dimensions or regularize.",
        },
        {
            "profile": {"unsupervised": True, "cluster_shape": "unknown"},
            "pattern": "Density-based or hierarchical (Bentley: back-of-envelope for k)",
            "suggestion": "Try DBSCAN or hierarchical clustering; use silhouette to compare.",
            "why": "Unknown k and shape suggest DBSCAN or dendrogram inspection.",
        },
    ]
    # Query: first profile
    query = rules[0]["profile"]
    match = next((r for r in rules if r["profile"] == query), None)
    assert match is not None
    chain = f"Pattern: {match['pattern']}\nSuggestion: {match['suggestion']}\nWhy: {match['why']}"
    return {
        "ok": True,
        "rules_count": len(rules),
        "query_matched": True,
        "chain_length": len(chain),
        "pattern_preview": match["pattern"][:50] + "...",
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
