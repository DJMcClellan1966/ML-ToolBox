"""
Test Ensemble Learning with AdvancedDataPreprocessor
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_learning import EnsembleLearner, PreprocessorEnsemble
from data_preprocessor import AdvancedDataPreprocessor

# Try to import sklearn models
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.datasets import make_classification, make_regression
    SKLEARN_MODELS_AVAILABLE = True
except ImportError:
    SKLEARN_MODELS_AVAILABLE = False
    print("Warning: sklearn models not available. Install with: pip install scikit-learn")


def test_voting_ensemble():
    """Test voting ensemble"""
    print("="*80)
    print("VOTING ENSEMBLE TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    # Create data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=42)
    
    # Create base models
    models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    # Create voting ensemble
    ensemble_learner = EnsembleLearner(random_state=42)
    ensemble = ensemble_learner.create_voting_ensemble(models, task_type='classification', voting='soft')
    
    # Evaluate
    results = ensemble_learner.evaluate_ensemble(
        ensemble, X, y,
        task_type='classification',
        cv_folds=5,
        verbose=True
    )
    
    # Compare with individual models
    print("\n[Individual Model Comparison]")
    print("-" * 80)
    for name, model in models:
        individual_results = ensemble_learner.evaluate_ensemble(
            model, X, y,
            task_type='classification',
            cv_folds=5,
            verbose=False
        )
        print(f"{name}: Test Score = {individual_results['test_score']:.4f}")
    
    print(f"\nEnsemble: Test Score = {results['test_score']:.4f}")
    print(f"Improvement: {results['test_score'] - max([ensemble_learner.evaluate_ensemble(m, X, y, task_type='classification', cv_folds=5, verbose=False)['test_score'] for _, m in models]):.4f}")


def test_bagging_ensemble():
    """Test bagging ensemble"""
    print("\n" + "="*80)
    print("BAGGING ENSEMBLE TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    # Create data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=42)
    
    # Create base model
    base_model = DecisionTreeClassifier(random_state=42)
    
    # Create bagging ensemble
    ensemble_learner = EnsembleLearner(random_state=42)
    ensemble = ensemble_learner.create_bagging_ensemble(
        base_model, n_estimators=10, task_type='classification'
    )
    
    # Evaluate
    results = ensemble_learner.evaluate_ensemble(
        ensemble, X, y,
        task_type='classification',
        cv_folds=5,
        verbose=True
    )
    
    # Compare with base model
    print("\n[Base Model Comparison]")
    print("-" * 80)
    base_results = ensemble_learner.evaluate_ensemble(
        base_model, X, y,
        task_type='classification',
        cv_folds=5,
        verbose=False
    )
    print(f"Base Model: Test Score = {base_results['test_score']:.4f}")
    print(f"Bagging Ensemble: Test Score = {results['test_score']:.4f}")
    print(f"Improvement: {results['test_score'] - base_results['test_score']:.4f}")


def test_boosting_ensemble():
    """Test boosting ensemble"""
    print("\n" + "="*80)
    print("BOOSTING ENSEMBLE TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    # Create data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=42)
    
    # Create base model
    base_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    # Create boosting ensemble
    ensemble_learner = EnsembleLearner(random_state=42)
    ensemble = ensemble_learner.create_boosting_ensemble(
        base_model, n_estimators=50, learning_rate=0.1, task_type='classification'
    )
    
    # Evaluate
    results = ensemble_learner.evaluate_ensemble(
        ensemble, X, y,
        task_type='classification',
        cv_folds=5,
        verbose=True
    )


def test_stacking_ensemble():
    """Test stacking ensemble"""
    print("\n" + "="*80)
    print("STACKING ENSEMBLE TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    # Create data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=42)
    
    # Create base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    # Create meta-model
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create stacking ensemble
    ensemble_learner = EnsembleLearner(random_state=42)
    ensemble = ensemble_learner.create_stacking_ensemble(
        base_models, meta_model, task_type='classification', cv_folds=5
    )
    
    # Evaluate
    results = ensemble_learner.evaluate_ensemble(
        ensemble, X, y,
        task_type='classification',
        cv_folds=5,
        verbose=True
    )


def test_preprocessor_ensemble():
    """Test preprocessor ensemble"""
    print("\n" + "="*80)
    print("PREPROCESSOR ENSEMBLE TEST")
    print("="*80)
    
    # Create sample data
    raw_data = [
        "Python is great for data science",
        "Python is excellent for data science",  # Semantic duplicate
        "Machine learning uses algorithms",
        "ML uses algorithms",  # Semantic duplicate
        "I need help with programming",
        "I require assistance with code",  # Semantic duplicate
        "Business revenue increased",
        "Sales profits grew",
        "Learn Python programming",
        "Study Python coding"  # Semantic duplicate
    ]
    
    # Create multiple preprocessors with different settings
    preprocessor1 = AdvancedDataPreprocessor(
        dedup_threshold=0.8,
        compression_ratio=0.5,
        compression_method='pca'
    )
    
    preprocessor2 = AdvancedDataPreprocessor(
        dedup_threshold=0.9,
        compression_ratio=0.7,
        compression_method='pca'
    )
    
    preprocessor3 = AdvancedDataPreprocessor(
        dedup_threshold=0.85,
        compression_ratio=0.6,
        compression_method='pca'
    )
    
    # Create preprocessor ensemble
    preprocessor_ensemble = PreprocessorEnsemble(random_state=42)
    preprocessor_ensemble.add_preprocessor('preprocessor1', preprocessor1)
    preprocessor_ensemble.add_preprocessor('preprocessor2', preprocessor2)
    preprocessor_ensemble.add_preprocessor('preprocessor3', preprocessor3)
    
    # Preprocess with ensemble
    results = preprocessor_ensemble.preprocess_ensemble(raw_data, verbose=True)
    
    print(f"\n[Ensemble Summary]")
    print(f"  Combined embeddings shape: {results['combined_embeddings'].shape}")
    print(f"  Consensus categories: {len(results['consensus_categories'])}")
    print(f"  Average quality: {results['avg_quality']:.4f}")


def main():
    """Run all tests"""
    try:
        test_voting_ensemble()
        test_bagging_ensemble()
        test_boosting_ensemble()
        test_stacking_ensemble()
        test_preprocessor_ensemble()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
