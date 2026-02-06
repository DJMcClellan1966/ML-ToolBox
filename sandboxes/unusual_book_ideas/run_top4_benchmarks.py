"""
Run Top 4 pipeline against multiple benchmark datasets (Kaggle-style / UCI-style).
Uses only sklearn built-in datasets; no Kaggle API required.
Datasets: Iris, Digits, Wine, Breast Cancer.
Output: JSON and brief summary for CI or local validation.
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = Path(__file__).resolve().parent
for p in [REPO_ROOT, SANDBOX_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def run_top4_on_dataset(name, X, y, test_size=0.25, random_state=42):
    """Run Oracle + Explainer + Socratic (stub) + 3-model ensemble; return metrics."""
    from combined_top4.run import oracle_suggest, explainer_entropy, theory_as_channel_ensemble
    rng = np.random.RandomState(random_state)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rng)
    models = [
        LogisticRegression(random_state=rng, max_iter=500),
        RandomForestClassifier(n_estimators=50, random_state=rng),
        SVC(kernel="rbf", random_state=rng),
    ]
    for m in models:
        m.fit(X_train, y_train)
    tc = theory_as_channel_ensemble(models, X_test, y_test)
    profile = {"tabular": True, "classification": True, "n_samples": "medium"}
    oracle = oracle_suggest(profile)
    explainer = explainer_entropy()
    return {
        "dataset": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(np.unique(y)),
        "oracle_ok": oracle["ok"],
        "explainer_ok": explainer["ok"],
        "single_accs": tc["single_accs"],
        "corrected_acc": tc["corrected_acc"],
        "best_single": tc["best_single"],
        "improvement": tc["improvement"],
        "ensemble_helps_or_equal": tc["corrected_acc"] >= tc["best_single"],
    }


def main():
    if not SKLEARN_AVAILABLE:
        print("SKIP: sklearn not available")
        return {"ok": False, "reason": "sklearn not available"}

    results = {}
    rng = np.random.RandomState(42)

    # Iris (binary subset for consistency with sandbox)
    X, y = load_iris(return_X_y=True)
    y_bin = (y == 0).astype(int)
    results["iris_binary"] = run_top4_on_dataset("iris_binary", X, y_bin, random_state=42)

    # Digits
    X, y = load_digits(return_X_y=True)
    results["digits"] = run_top4_on_dataset("digits", X, y, random_state=42)

    # Wine
    X, y = load_wine(return_X_y=True)
    results["wine"] = run_top4_on_dataset("wine", X, y, random_state=42)

    # Breast Cancer
    X, y = load_breast_cancer(return_X_y=True)
    results["breast_cancer"] = run_top4_on_dataset("breast_cancer", X, y, random_state=42)

    # Summary: pass if oracle + explainer ok on all; ensemble can be <= best_single on some datasets (known for majority vote)
    components_ok = all(
        r.get("oracle_ok") and r.get("explainer_ok")
        for r in results.values()
    )
    ensemble_helps_count = sum(1 for r in results.values() if r.get("ensemble_helps_or_equal"))
    results["_summary"] = {
        "all_pass": components_ok,
        "datasets": list(k for k in results if not k.startswith("_")),
        "ensemble_helps_or_equal_count": ensemble_helps_count,
        "total_datasets": len([k for k in results if not k.startswith("_")]),
    }

    out_path = SANDBOX_DIR / "top4_benchmark_results.json"
    # Convert numpy for JSON
    def to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print("Top 4 benchmark results:", json.dumps(results["_summary"], indent=2))
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"  {k}: corrected_acc={v['corrected_acc']:.4f}, best_single={v['best_single']:.4f}, ensemble_helps_or_equal={v['ensemble_helps_or_equal']}")
    print("Wrote", out_path)
    return results


if __name__ == "__main__":
    main()
    sys.exit(0)
