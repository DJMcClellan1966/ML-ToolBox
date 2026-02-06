"""
Curriculum: Hands-On Machine Learning (Géron) — practical ML workflows, pipelines, production.
From ml_toolbox.textbook_concepts.practical_ml: FeatureEngineering, ModelSelection, HyperparameterTuning, EnsembleMethods, CrossValidation, ProductionML.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "feature_eng", "name": "Feature Engineering", "short": "Features", "color": "#2563eb"},
    {"id": "model_selection", "name": "Model Selection", "short": "Model Select", "color": "#059669"},
    {"id": "tuning", "name": "Hyperparameter Tuning", "short": "Tuning", "color": "#7c3aed"},
    {"id": "ensembles", "name": "Ensemble Methods", "short": "Ensembles", "color": "#dc2626"},
    {"id": "cross_val", "name": "Cross-Validation", "short": "CV", "color": "#d97706"},
    {"id": "production", "name": "Production ML", "short": "Production", "color": "#0d9488"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "poly_features", "book_id": "feature_eng", "level": "basics", "title": "Polynomial & Interaction Features",
     "learn": "Create polynomial and interaction features for linear models. FeatureEngineering.polynomial_features, interaction_features.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import FeatureEngineering\nimport numpy as np\nX = np.array([[1,2],[2,3],[3,4]])\nprint(FeatureEngineering.polynomial_features(X, degree=2).shape)",
     "try_demo": None},
    {"id": "binning", "book_id": "feature_eng", "level": "intermediate", "title": "Binning & Scaling",
     "learn": "Bin continuous features; scale for models. FeatureEngineering.binning, scaling helpers.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import FeatureEngineering\nimport numpy as np\nX = np.random.randn(100, 3)\nprint(FeatureEngineering.binning(X, n_bins=5).shape)",
     "try_demo": None},
    {"id": "model_select", "book_id": "model_selection", "level": "basics", "title": "Model Selection",
     "learn": "Choose model by task and data size. ModelSelection: select_classifier, select_regressor, select_by_criteria.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import ModelSelection\nprint(ModelSelection.select_classifier(n_samples=1000, n_features=20))",
     "try_demo": None},
    {"id": "hyperparam", "book_id": "tuning", "level": "intermediate", "title": "Hyperparameter Tuning",
     "learn": "Grid search, random search, Bayesian-style tuning. HyperparameterTuning from practical_ml.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import HyperparameterTuning",
     "try_demo": None},
    {"id": "ensemble", "book_id": "ensembles", "level": "intermediate", "title": "Ensemble Methods",
     "learn": "Voting, stacking, bagging. EnsembleMethods: voting classifier/regressor, stack_models.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import EnsembleMethods",
     "try_demo": None},
    {"id": "cross_validation", "book_id": "cross_val", "level": "basics", "title": "Cross-Validation",
     "learn": "K-fold, stratified K-fold, time-series splits. CrossValidation from practical_ml.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import CrossValidation\nimport numpy as np\nX,y=np.random.randn(100,5),np.random.randint(0,2,100)\nprint(CrossValidation.cross_validate_stratified(X,y,n_splits=5))",
     "try_demo": None},
    {"id": "production_ml", "book_id": "production", "level": "advanced", "title": "Production ML",
     "learn": "Deployment patterns, versioning, monitoring. ProductionML from practical_ml.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import ProductionML",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
