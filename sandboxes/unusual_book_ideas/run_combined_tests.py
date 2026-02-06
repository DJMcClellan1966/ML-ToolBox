"""
Rigorous test runner for combined Top 4 and Top 8 sandboxes.
Uses real sklearn datasets (Iris, Digits) and models; records metrics and pass/fail.
"""
import sys
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = Path(__file__).resolve().parent
for p in [REPO_ROOT, SANDBOX_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from sklearn.datasets import load_digits
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def run_top4():
    from combined_top4.run import run_top4 as _run
    return _run()


def run_top8():
    from combined_top8.run import run_top8 as _run
    return _run()


def main():
    results = {"top4": None, "top8": None, "summary": {}}

    # Run Top 4
    try:
        results["top4"] = run_top4()
        r4 = results["top4"]
        rw4 = r4.get("real_world", {})
        results["summary"]["top4"] = {
            "oracle_ok": r4.get("oracle", {}).get("ok", False),
            "explainer_ok": r4.get("explainer", {}).get("ok", False),
            "socratic_ok": r4.get("socratic", {}).get("ok", False),
            "real_world_ok": rw4.get("ok", False),
            "corrected_acc": rw4.get("corrected_acc"),
            "best_single_acc": rw4.get("best_single"),
            "ensemble_helps_or_equal": rw4.get("corrected_acc", 0) >= rw4.get("best_single", 0) if rw4.get("ok") else None,
            "pass": (
                r4.get("oracle", {}).get("ok")
                and r4.get("explainer", {}).get("ok")
                and r4.get("socratic", {}).get("ok")
                and (rw4.get("ok") and (rw4.get("corrected_acc", 0) >= rw4.get("best_single", 0)))
            ),
        }
    except Exception as e:
        results["summary"]["top4"] = {"pass": False, "error": str(e)}

    # Run Top 8
    try:
        results["top8"] = run_top8()
        r8 = results["top8"]
        rw8 = r8.get("real_world", {})
        results["summary"]["top8"] = {
            "oracle_ok": r8.get("oracle", {}).get("ok", False),
            "explainer_ok": r8.get("explainer", {}).get("ok", False),
            "socratic_ok": r8.get("socratic", {}).get("ok", False),
            "warn_ok": r8.get("warn", {}).get("ok", False),
            "curriculum_ok": r8.get("curriculum", {}).get("ok", False),
            "linguistics_ok": r8.get("linguistics", {}).get("ok", False),
            "dissipative_ok": r8.get("dissipative", {}).get("ok", False),
            "real_world_ok": rw8.get("ok", False),
            "corrected_acc": rw8.get("corrected_acc"),
            "best_single_acc": rw8.get("best_single"),
            "ensemble_helps_or_equal": rw8.get("corrected_acc", 0) >= rw8.get("best_single", 0) if rw8.get("ok") else None,
            "warnings_count": len(r8.get("warn", {}).get("warnings", [])),
            "pass": (
                r8.get("oracle", {}).get("ok")
                and r8.get("explainer", {}).get("ok")
                and r8.get("socratic", {}).get("ok")
                and r8.get("curriculum", {}).get("ok")
                and r8.get("linguistics", {}).get("ok")
                and r8.get("dissipative", {}).get("ok")
                and (r8.get("warn", {}).get("ok") or not rw8.get("ok"))  # warn optional if no sklearn
                and (rw8.get("ok") and (rw8.get("corrected_acc", 0) >= rw8.get("best_single", 0)))
            ),
        }
    except Exception as e:
        results["summary"]["top8"] = {"pass": False, "error": str(e)}

    # Rigorous: also run on Digits (harder dataset) for ensemble comparison
    if SKLEARN_AVAILABLE and results.get("top4") and results["top4"].get("real_world", {}).get("ok"):
        try:
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
            X, y = load_digits(return_X_y=True)
            X = StandardScaler().fit_transform(X)
            rng = np.random.RandomState(42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng)
            models = [
                LogisticRegression(random_state=rng, max_iter=500),
                RandomForestClassifier(n_estimators=50, random_state=rng),
                SVC(kernel="rbf", random_state=rng),
            ]
            for m in models:
                m.fit(X_train, y_train)
            preds = np.column_stack([m.predict(X_test) for m in models])
            ec = ErrorCorrectingPredictions(redundancy_factor=3)
            corrected = ec.correct_predictions(preds, method="majority_vote")
            single_accs = [np.mean(m.predict(X_test) == y_test) for m in models]
            corrected_acc = np.mean(corrected == y_test)
            results["digits"] = {
                "single_accs": [float(a) for a in single_accs],
                "corrected_acc": float(corrected_acc),
                "best_single": float(max(single_accs)),
                "improvement": float(corrected_acc - max(single_accs)),
                "ensemble_helps": bool(corrected_acc >= max(single_accs)),
            }
        except Exception as e:
            results["digits"] = {"error": str(e)}
    else:
        results["digits"] = {"skipped": True}

    return results


if __name__ == "__main__":
    out = main()
    print("SUMMARY TOP4:", out["summary"].get("top4"))
    print("SUMMARY TOP8:", out["summary"].get("top8"))
    top4_pass = out["summary"].get("top4", {}).get("pass", False)
    top8_pass = out["summary"].get("top8", {}).get("pass", False)
    print("TOP4 PASS:", top4_pass)
    print("TOP8 PASS:", top8_pass)
    # Write machine-readable results for assessment doc
    out_path = Path(__file__).parent / "combined_test_results.json"
    export = {"summary": out["summary"], "top4_real_world": out.get("top4", {}).get("real_world"), "top8_real_world": out.get("top8", {}).get("real_world"), "top8_warn": out.get("top8", {}).get("warn"), "digits": out.get("digits")}
    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(type(obj))
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2, default=_json_default)
    print("Wrote", out_path)
    sys.exit(0 if (top4_pass and top8_pass) else 1)
