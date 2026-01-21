"""
Test ML Evaluation and Hyperparameter Tuning
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_evaluation import MLEvaluator, HyperparameterTuner, PreprocessorOptimizer
from data_preprocessor import AdvancedDataPreprocessor

# Try to import sklearn models
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    SKLEARN_MODELS_AVAILABLE = True
except ImportError:
    SKLEARN_MODELS_AVAILABLE = False
    print("Warning: sklearn models not available. Install with: pip install scikit-learn")


def test_model_evaluation():
    """Test model evaluation"""
    print("="*80)
    print("MODEL EVALUATION TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    # Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    # Classification test
    print("\n[TEST 1] Classification Evaluation")
    print("-" * 80)
    X_clf, y_clf = make_classification(n_samples=200, n_features=20, n_classes=3, n_informative=10, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    evaluator = MLEvaluator(random_state=42)
    
    results = evaluator.evaluate_model(
        model, X_clf, y_clf,
        task_type='classification',
        cv_folds=5,
        verbose=True
    )
    
    # Regression test
    print("\n[TEST 2] Regression Evaluation")
    print("-" * 80)
    X_reg, y_reg = make_regression(n_samples=200, n_features=20, random_state=42)
    
    model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
    
    results_reg = evaluator.evaluate_model(
        model_reg, X_reg, y_reg,
        task_type='regression',
        cv_folds=5,
        verbose=True
    )


def test_hyperparameter_tuning():
    """Test hyperparameter tuning"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING TEST")
    print("="*80)
    
    if not SKLEARN_MODELS_AVAILABLE:
        print("\nSkipping test - sklearn models not available")
        return
    
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=200, n_features=20, n_classes=3, n_informative=10, random_state=42)
    
    # Grid search
    print("\n[TEST 1] Grid Search")
    print("-" * 80)
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 10]
    }
    
    tuner = HyperparameterTuner(random_state=42)
    results = tuner.tune_hyperparameters(
        model, param_grid, X, y,
        method='grid',
        cv_folds=3,
        verbose=True
    )
    
    # Random search
    print("\n[TEST 2] Random Search")
    print("-" * 80)
    results_random = tuner.tune_hyperparameters(
        model, param_grid, X, y,
        method='random',
        n_iter=10,
        cv_folds=3,
        verbose=True
    )


def test_preprocessor_optimization():
    """Test preprocessor optimization"""
    print("\n" + "="*80)
    print("PREPROCESSOR OPTIMIZATION TEST")
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
        "Sales profits grew",  # Similar
        "Learn Python programming",
        "Study Python coding"  # Semantic duplicate
    ]
    
    labels = ['technical', 'technical', 'technical', 'technical', 
              'support', 'support', 'business', 'business', 
              'education', 'education']
    
    print("\n[TEST] Preprocessor Optimization")
    print("-" * 80)
    
    optimizer = PreprocessorOptimizer(random_state=42)
    
    # Smaller grid for testing
    param_grid = {
        'dedup_threshold': [0.8, 0.9],
        'compression_ratio': [0.5, 0.7],
        'compression_method': ['pca']
    }
    
    results = optimizer.optimize_preprocessor(
        raw_data,
        labels=labels,
        task_type='classification',
        param_grid=param_grid,
        cv_folds=3,
        verbose=True
    )


def main():
    """Run all tests"""
    try:
        test_model_evaluation()
        test_hyperparameter_tuning()
        test_preprocessor_optimization()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
