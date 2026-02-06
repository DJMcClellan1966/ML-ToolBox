"""
Tests for Combined Top 4 sandbox (Oracle + Explainers + Theory-as-Channel + Socratic).
Runs against real sklearn datasets. No Kaggle API required.
"""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = REPO_ROOT / "sandboxes" / "unusual_book_ideas"
for p in [REPO_ROOT, SANDBOX_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Only run if sandbox exists
COMBINED_TOP4_EXISTS = (SANDBOX_DIR / "combined_top4" / "run.py").exists()


@pytest.mark.skipif(not COMBINED_TOP4_EXISTS, reason="combined_top4 sandbox not found")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestCombinedTop4:
    """Run Top 4 sandbox and assert component and real-world results."""

    def test_top4_sandbox_full_run(self):
        """Top 4 run completes and all components pass."""
        from combined_top4.run import run_top4
        r = run_top4()
        assert r["oracle"]["ok"], "oracle should return ok"
        assert r["explainer"]["ok"], "explainer should return ok"
        assert r["socratic"]["ok"], "socratic should return ok"
        assert r.get("real_world", {}).get("ok"), "real_world (Iris) should run"
        assert r["real_world"]["corrected_acc"] >= r["real_world"]["best_single"], "ensemble should be >= best single"

    def test_top4_oracle_returns_suggestion(self):
        from combined_top4.run import oracle_suggest
        out = oracle_suggest({"tabular": True, "classification": True, "n_samples": "medium"})
        assert out["ok"]
        assert out.get("pattern") and out.get("suggestion") and out.get("why")

    def test_top4_explainer_three_views(self):
        from combined_top4.run import explainer_entropy
        out = explainer_entropy()
        assert out["ok"]
        assert "information_theory" in out and "thermodynamics" in out and "ml" in out

    def test_top4_ensemble_on_digits(self):
        """Theory-as-channel: ensemble improves or equals best single on Digits."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
        import numpy as np
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
        assert corrected_acc >= max(single_accs), "Digits: ensemble should be >= best single"

    def test_top4_ensemble_on_wine(self):
        """Top 4 style ensemble on Wine (multiclass)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
        import numpy as np
        X, y = load_wine(return_X_y=True)
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
        corrected_acc = np.mean(corrected == y_test)
        best_single = max(np.mean(m.predict(X_test) == y_test) for m in models)
        assert corrected_acc >= best_single, "Wine: ensemble should be >= best single"

    def test_top4_ensemble_on_breast_cancer(self):
        """Top 4 style ensemble on Breast Cancer (binary). Majority vote can be marginally below best single."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
        import numpy as np
        X, y = load_breast_cancer(return_X_y=True)
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
        corrected_acc = np.mean(corrected == y_test)
        best_single = max(np.mean(m.predict(X_test) == y_test) for m in models)
        # Allow small gap: majority vote can be marginally worse than best single on some splits
        assert corrected_acc >= best_single - 0.02, "Breast Cancer: ensemble should be within 2% of best single"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
