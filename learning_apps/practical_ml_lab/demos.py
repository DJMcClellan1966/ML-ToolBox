"""Demos for Practical ML Lab. Uses ml_toolbox.textbook_concepts.practical_ml."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def practical_poly_features():
    """Polynomial features demo."""
    try:
        from ml_toolbox.textbook_concepts.practical_ml import FeatureEngineering
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_poly = FeatureEngineering.polynomial_features(X, degree=2)
        out = "Polynomial Features (degree=2)\n"
        out += f"Input shape: {X.shape}\n"
        out += f"Output shape: {X_poly.shape}\n"
        out += f"Original: {X[0].tolist()}\n"
        out += f"Expanded: {X_poly[0].round(2).tolist()[:10]}..."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def practical_model_select():
    """Model selection demo."""
    try:
        from ml_toolbox.textbook_concepts.practical_ml import ModelSelection
        result = ModelSelection.select_classifier(n_samples=1000, n_features=20)
        out = "Model Selection for Classification\n"
        out += f"Given: 1000 samples, 20 features\n"
        out += f"Recommended: {result.get('model', result)}\n"
        
        result2 = ModelSelection.select_regressor(n_samples=500, n_features=10)
        out += f"\nFor regression (500 samples, 10 features):\n"
        out += f"Recommended: {result2.get('model', result2)}"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def practical_cross_val():
    """Cross-validation demo."""
    try:
        from ml_toolbox.textbook_concepts.practical_ml import CrossValidation
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        result = CrossValidation.cross_validate_stratified(X, y, n_splits=5)
        out = "Stratified K-Fold Cross-Validation\n"
        out += f"Data: 100 samples, 5 features, binary labels\n"
        out += f"Splits: 5\n"
        out += f"Result: {result}\n"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def practical_ensemble():
    """Ensemble methods demo."""
    try:
        from ml_toolbox.textbook_concepts.practical_ml import EnsembleMethods
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        models = [DecisionTreeClassifier(max_depth=3), LogisticRegression()]
        for m in models:
            m.fit(X[:150], y[:150])
        
        ensemble = EnsembleMethods.voting_classifier(models, X[150:], voting='hard')
        out = "Voting Classifier Ensemble\n"
        out += "Models: DecisionTree + LogisticRegression\n"
        out += f"Ensemble predictions shape: {len(ensemble)}\n"
        out += f"Sample predictions: {ensemble[:10].tolist()}"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def practical_binning():
    """Binning features demo."""
    try:
        from ml_toolbox.textbook_concepts.practical_ml import FeatureEngineering
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X_binned = FeatureEngineering.binning(X, n_bins=5)
        out = "Feature Binning\n"
        out += f"Input: 50 samples, 3 features (continuous)\n"
        out += f"Bins: 5\n"
        out += f"Output shape: {X_binned.shape}\n"
        out += f"Unique bin values: {np.unique(X_binned[:, 0]).tolist()}"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


DEMO_HANDLERS = {
    "practical_poly": practical_poly_features,
    "practical_model_select": practical_model_select,
    "practical_cross_val": practical_cross_val,
    "practical_ensemble": practical_ensemble,
    "practical_binning": practical_binning,
}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
