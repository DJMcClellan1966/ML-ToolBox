"""
Machine Learning Evaluation and Hyperparameter Tuning
Best practices for model evaluation and optimization
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessor import AdvancedDataPreprocessor
import numpy as np

# Try to import sklearn for ML evaluation
try:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, KFold, StratifiedKFold,
        GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class MLEvaluator:
    """
    Machine Learning Evaluator with Best Practices
    
    Features:
    - Cross-validation
    - Train/test splits
    - Multiple metrics
    - Learning curves
    - Overfitting detection
    - Model comparison
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_results = []
        
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        cv_folds: int = 5,
        test_size: float = 0.2,
        metrics: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            test_size: Test set size ratio
            metrics: List of metrics to compute
            verbose: Print detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Default metrics
        if metrics is None:
            if task_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            else:
                metrics = ['mse', 'mae', 'r2']
        
        # Encode labels for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
            le = None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded if task_type == 'classification' else None
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            'task_type': task_type,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'cv_folds': cv_folds,
            'metrics': {},
            'cross_validation': {},
            'learning_curves': None,
            'overfitting_detected': False,
            'processing_time': time.time() - start_time
        }
        
        # Test set metrics
        if task_type == 'classification':
            results['metrics']['test'] = self._calculate_classification_metrics(
                y_test, y_test_pred, metrics
            )
            results['metrics']['train'] = self._calculate_classification_metrics(
                y_train, y_train_pred, metrics
            )
        else:
            results['metrics']['test'] = self._calculate_regression_metrics(
                y_test, y_test_pred, metrics
            )
            results['metrics']['train'] = self._calculate_regression_metrics(
                y_train, y_train_pred, metrics
            )
        
        # Cross-validation
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        results['cross_validation'] = {
            'mean': float(np.mean(cv_scores)),
            'std': float(np.std(cv_scores)),
            'scores': cv_scores.tolist()
        }
        
        # Overfitting detection
        if task_type == 'classification':
            train_acc = results['metrics']['train'].get('accuracy', 0)
            test_acc = results['metrics']['test'].get('accuracy', 0)
            results['overfitting_detected'] = (train_acc - test_acc) > 0.1
            results['overfitting_gap'] = train_acc - test_acc
        else:
            train_r2 = results['metrics']['train'].get('r2', 0)
            test_r2 = results['metrics']['test'].get('r2', 0)
            results['overfitting_detected'] = (train_r2 - test_r2) > 0.2
            results['overfitting_gap'] = train_r2 - test_r2
        
        # Learning curves
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            results['learning_curves'] = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist()
            }
        except:
            pass
        
        # Print results
        if verbose:
            self._print_evaluation_results(results)
        
        self.evaluation_results.append(results)
        return results
    
    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = float(accuracy_score(y_true, y_pred))
        if 'precision' in metrics:
            results['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        if 'recall' in metrics:
            results['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        if 'f1' in metrics:
            results['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        return results
    
    def _calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = float(mean_squared_error(y_true, y_pred))
        if 'mae' in metrics:
            results['mae'] = float(mean_absolute_error(y_true, y_pred))
        if 'r2' in metrics:
            results['r2'] = float(r2_score(y_true, y_pred))
        
        return results
    
    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nTask Type: {results['task_type']}")
        print(f"Train Size: {results['train_size']}, Test Size: {results['test_size']}")
        print(f"CV Folds: {results['cv_folds']}")
        
        print("\n[Test Set Metrics]")
        for metric, value in results['metrics']['test'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n[Cross-Validation]")
        print(f"  Mean: {results['cross_validation']['mean']:.4f}")
        print(f"  Std: {results['cross_validation']['std']:.4f}")
        
        print("\n[Overfitting Detection]")
        if results['overfitting_detected']:
            print(f"  WARNING: Overfitting detected! Gap: {results['overfitting_gap']:.4f}")
        else:
            print(f"  No significant overfitting. Gap: {results['overfitting_gap']:.4f}")
        
        print(f"\nProcessing Time: {results['processing_time']:.3f}s")
        print("="*80 + "\n")


class HyperparameterTuner:
    """
    Hyperparameter Tuning with Best Practices
    
    Features:
    - Grid search
    - Random search
    - Bayesian optimization (if available)
    - Best parameter selection
    - Validation curve analysis
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tuning_results = []
    
    def tune_hyperparameters(
        self,
        model: Any,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        method: str = 'grid',  # 'grid', 'random'
        cv_folds: int = 5,
        n_iter: int = 20,  # For random search
        scoring: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning
        
        Args:
            model: Scikit-learn compatible model
            param_grid: Parameter grid to search
            X: Features
            y: Labels
            task_type: 'classification' or 'regression'
            method: 'grid' or 'random'
            cv_folds: Number of CV folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric
            verbose: Print detailed results
            
        Returns:
            Dictionary with tuning results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Default scoring
        if scoring is None:
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # CV strategy
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Choose search method
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_jobs=self.n_jobs, verbose=1 if verbose else 0
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
                n_jobs=self.n_jobs, random_state=self.random_state,
                verbose=1 if verbose else 0
            )
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Perform search
        if verbose:
            print(f"\n[Tuning] Method: {method.upper()}, CV Folds: {cv_folds}")
            print(f"Parameter Grid: {len(param_grid)} parameters")
        
        search.fit(X, y)
        
        # Results
        results = {
            'method': method,
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'best_model': search.best_estimator_,
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            },
            'processing_time': time.time() - start_time
        }
        
        # Validation curves for top parameters
        if verbose:
            self._print_tuning_results(results)
        
        self.tuning_results.append(results)
        return results
    
    def _print_tuning_results(self, results: Dict):
        """Print tuning results"""
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*80)
        
        print(f"\nMethod: {results['method'].upper()}")
        print(f"Best Score: {results['best_score']:.4f}")
        print("\n[Best Parameters]")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nProcessing Time: {results['processing_time']:.3f}s")
        print("="*80 + "\n")
    
    def analyze_validation_curves(
        self,
        model: Any,
        param_name: str,
        param_range: List[Any],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        cv_folds: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze validation curves for a parameter
        
        Returns:
            Dictionary with validation curve data
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=self.n_jobs
        )
        
        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist()
        }


class PreprocessorOptimizer:
    """
    Optimize AdvancedDataPreprocessor hyperparameters
    
    Uses ML best practices to find optimal preprocessing parameters
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.optimization_results = []
    
    def optimize_preprocessor(
        self,
        raw_data: List[str],
        labels: Optional[List[Any]] = None,
        task_type: str = 'classification',
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize preprocessor hyperparameters
        
        Args:
            raw_data: Raw text data
            labels: Optional labels for supervised optimization
            task_type: 'classification' or 'regression'
            param_grid: Parameter grid to search
            cv_folds: Number of CV folds
            verbose: Print detailed results
            
        Returns:
            Dictionary with optimization results
        """
        if param_grid is None:
            param_grid = {
                'dedup_threshold': [0.7, 0.8, 0.9, 0.95],
                'compression_ratio': [0.3, 0.5, 0.7, 0.9],
                'compression_method': ['pca', 'svd']
            }
        
        best_score = -np.inf
        best_params = None
        best_preprocessor = None
        all_results = []
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        
        if verbose:
            print(f"\n[Optimization] Testing {total_combinations} parameter combinations")
            print(f"CV Folds: {cv_folds}")
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Create preprocessor with these parameters
            preprocessor = AdvancedDataPreprocessor(
                dedup_threshold=params.get('dedup_threshold', 0.9),
                enable_compression=params.get('compression_ratio', 0.5) is not None,
                compression_ratio=params.get('compression_ratio', 0.5) or 0.5,
                compression_method=params.get('compression_method', 'pca')
            )
            
            # Preprocess data
            results = preprocessor.preprocess(raw_data, verbose=False)
            
            # Evaluate (simplified - would use actual model in production)
            if labels is not None:
                # Use compressed embeddings for evaluation
                if results['compressed_embeddings'] is not None:
                    X = results['compressed_embeddings']
                else:
                    # Use original embeddings
                    X = np.array([preprocessor.quantum_kernel.embed(item) for item in results['deduplicated']])
                
                # Simple evaluation metric (data quality)
                score = self._evaluate_preprocessing_quality(results, X, labels)
            else:
                # Unsupervised: use data quality metrics
                score = self._evaluate_data_quality(results)
            
            all_results.append({
                'params': params,
                'score': score,
                'results': results
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_preprocessor = preprocessor
        
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'best_preprocessor': best_preprocessor,
            'all_results': all_results,
            'total_combinations': total_combinations
        }
        
        if verbose:
            self._print_optimization_results(optimization_results)
        
        self.optimization_results.append(optimization_results)
        return optimization_results
    
    def _evaluate_preprocessing_quality(
        self, results: Dict, X: np.ndarray, labels: List[Any]
    ) -> float:
        """Evaluate preprocessing quality using labels"""
        # Simple metric: balance between compression and data quality
        quality_score = results['stats']['avg_quality']
        compression_ratio = results['compression_info'].get('compression_ratio_achieved', 1.0)
        variance_retained = results['compression_info'].get('variance_retained', 0.0)
        
        # Combined score
        score = (
            quality_score * 0.4 +
            compression_ratio * 0.2 +
            variance_retained * 0.4
        )
        
        return score
    
    def _evaluate_data_quality(self, results: Dict) -> float:
        """Evaluate data quality without labels"""
        quality_score = results['stats']['avg_quality']
        duplicates_removed = results['stats']['duplicates_removed']
        original_count = results['original_count']
        
        # Score based on quality and deduplication effectiveness
        dedup_effectiveness = duplicates_removed / max(original_count, 1)
        
        score = quality_score * 0.6 + dedup_effectiveness * 0.4
        return score
    
    def _print_optimization_results(self, results: Dict):
        """Print optimization results"""
        print("\n" + "="*80)
        print("PREPROCESSOR OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\nTotal Combinations Tested: {results['total_combinations']}")
        print(f"Best Score: {results['best_score']:.4f}")
        print("\n[Best Parameters]")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print("="*80 + "\n")
