"""
Kuhn/Johnson Advanced Resampling Methods
Advanced cross-validation and resampling techniques from Applied Predictive Modeling

Methods:
- Repeated K-Fold CV (reduce variance in estimates)
- Bootstrap resampling (with confidence intervals)
- Leave-One-Out CV (for small datasets)
- Group K-Fold (for grouped data)
- Time Series CV (blocking, forward chaining)
- Nested CV (for hyperparameter tuning + evaluation)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import (
        KFold, StratifiedKFold, LeaveOneOut, GroupKFold,
        TimeSeriesSplit, cross_val_score, cross_validate
    )
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class RepeatedKFold:
    """
    Repeated K-Fold Cross-Validation
    
    Repeat k-fold CV multiple times with different random splits
    to reduce variance in performance estimates.
    """
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 10, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds in each CV
            n_repeats: Number of times to repeat k-fold CV
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.random_states = [random_state + i for i in range(n_repeats)]
    
    def split(self, X, y=None, groups=None):
        """Generate train/test splits"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RepeatedKFold")
        
        np.random.seed(self.random_state)
        
        for repeat_idx in range(self.n_repeats):
            if y is not None and len(np.unique(y)) < 10:  # Likely classification
                kf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_states[repeat_idx]
                )
                splits = kf.split(X, y, groups)
            else:
                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_states[repeat_idx]
                )
                splits = kf.split(X, y, groups)
            
            for train_idx, test_idx in splits:
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        return self.n_splits * self.n_repeats


class BootstrapResampler:
    """
    Bootstrap Resampling
    
    Create multiple bootstrap samples for robust performance estimation
    """
    
    def __init__(self, n_bootstraps: int = 100, random_state: int = 42):
        """
        Args:
            n_bootstraps: Number of bootstrap samples
            random_state: Random seed
        """
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        """Generate bootstrap train/test splits"""
        np.random.seed(self.random_state)
        n_samples = len(X)
        
        for i in range(self.n_bootstraps):
            # Bootstrap sample (with replacement)
            train_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Out-of-bag samples (not in bootstrap)
            all_idx = np.arange(n_samples)
            test_idx = np.setdiff1d(all_idx, train_idx)
            
            # If no OOB samples, use all samples
            if len(test_idx) == 0:
                test_idx = all_idx
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        return self.n_bootstraps
    
    def bootstrap_ci(self, scores: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval
        
        Args:
            scores: Array of scores from bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with mean, lower bound, upper bound
        """
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'lower': float(np.percentile(scores, lower_percentile)),
            'upper': float(np.percentile(scores, upper_percentile)),
            'confidence': confidence
        }


class TimeSeriesCV:
    """
    Time Series Cross-Validation
    
    Forward chaining CV for time series data
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        """
        Args:
            n_splits: Number of splits
            test_size: Size of test set (default: auto)
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None):
        """Generate time series splits"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for TimeSeriesCV")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        return tscv.split(X, y, groups)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        return self.n_splits


class GroupKFoldCV:
    """
    Group K-Fold Cross-Validation
    
    Ensures that the same group is not in both train and test sets
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Args:
            n_splits: Number of folds
        """
        self.n_splits = n_splits
    
    def split(self, X, y=None, groups=None):
        """Generate group-based splits"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for GroupKFoldCV")
        
        if groups is None:
            raise ValueError("groups must be provided for GroupKFoldCV")
        
        gkf = GroupKFold(n_splits=self.n_splits)
        return gkf.split(X, y, groups)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        return self.n_splits


class NestedCV:
    """
    Nested Cross-Validation
    
    Outer loop: Model evaluation
    Inner loop: Hyperparameter tuning
    
    Prevents overfitting in hyperparameter selection
    """
    
    def __init__(self, outer_cv: Any, inner_cv: Any):
        """
        Args:
            outer_cv: Outer CV splitter (for evaluation)
            inner_cv: Inner CV splitter (for hyperparameter tuning)
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
    
    def split(self, X, y=None, groups=None):
        """Generate nested CV splits"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for NestedCV")
        
        # Outer CV splits
        for outer_train_idx, outer_test_idx in self.outer_cv.split(X, y, groups):
            # Inner CV on outer training set
            X_outer_train = X[outer_train_idx]
            y_outer_train = y[outer_train_idx] if y is not None else None
            groups_outer_train = groups[outer_train_idx] if groups is not None else None
            
            inner_splits = list(self.inner_cv.split(
                X_outer_train, y_outer_train, groups_outer_train
            ))
            
            yield {
                'outer_train': outer_train_idx,
                'outer_test': outer_test_idx,
                'inner_splits': inner_splits
            }
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of outer splits"""
        return self.outer_cv.get_n_splits(X, y, groups)


class AdvancedResampler:
    """
    Main class for advanced resampling methods
    
    Implements Kuhn/Johnson resampling techniques
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def repeated_kfold_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 10,
        scoring: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Repeated K-Fold Cross-Validation
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            n_splits: Number of folds
            n_repeats: Number of repeats
            scoring: Scoring metric
            verbose: Print progress
            
        Returns:
            Dictionary with CV scores and statistics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=rkf,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        if verbose:
            print(f"Repeated K-Fold CV ({n_repeats} repeats x {n_splits} folds):")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Min: {np.min(cv_scores):.4f}, Max: {np.max(cv_scores):.4f}")
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'min': float(np.min(cv_scores)),
            'max': float(np.max(cv_scores)),
            'n_splits': n_splits * n_repeats,
            'n_repeats': n_repeats,
            'n_folds': n_splits
        }
    
    def bootstrap_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstraps: int = 100,
        scoring: Optional[str] = None,
        confidence: float = 0.95,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Bootstrap Cross-Validation with Confidence Intervals
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            n_bootstraps: Number of bootstrap samples
            scoring: Scoring metric
            confidence: Confidence level for CI
            verbose: Print progress
            
        Returns:
            Dictionary with bootstrap scores and confidence intervals
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        bootstrapper = BootstrapResampler(n_bootstraps=n_bootstraps, random_state=self.random_state)
        
        scores = []
        for train_idx, test_idx in bootstrapper.split(X, y):
            if len(test_idx) == 0:
                continue
            
            X_train_boot = X[train_idx]
            y_train_boot = y[train_idx]
            X_test_boot = X[test_idx]
            y_test_boot = y[test_idx]
            
            # Fit and evaluate
            model_copy = self._clone_model(model)
            model_copy.fit(X_train_boot, y_train_boot)
            
            # Score
            if scoring == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test_boot, model_copy.predict(X_test_boot))
            elif scoring == 'neg_mean_squared_error':
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_test_boot, model_copy.predict(X_test_boot))
            else:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(scoring)
                score = scorer(model_copy, X_test_boot, y_test_boot)
            
            scores.append(score)
        
        scores = np.array(scores)
        
        # Bootstrap CI
        ci = BootstrapResampler().bootstrap_ci(scores, confidence)
        
        if verbose:
            print(f"Bootstrap CV ({n_bootstraps} bootstraps):")
            print(f"  Mean Score: {ci['mean']:.4f} ± {ci['std']:.4f}")
            print(f"  {confidence*100:.0f}% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        return {
            'bootstrap_scores': scores.tolist(),
            'mean': ci['mean'],
            'std': ci['std'],
            'confidence_interval': {
                'lower': ci['lower'],
                'upper': ci['upper'],
                'confidence': confidence
            },
            'n_bootstraps': n_bootstraps
        }
    
    def loocv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Leave-One-Out Cross-Validation
        
        Best for small datasets
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            scoring: Scoring metric
            verbose: Print progress
            
        Returns:
            Dictionary with LOOCV scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        loo = LeaveOneOut()
        cv_scores = cross_val_score(model, X, y, cv=loo, scoring=scoring, n_jobs=-1)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        if verbose:
            print(f"Leave-One-Out CV ({len(X)} samples):")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'n_splits': len(X)
        }
    
    def group_kfold_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        scoring: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Group K-Fold Cross-Validation
        
        Ensures same group not in both train/test
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            groups: Group labels
            n_splits: Number of folds
            scoring: Scoring metric
            verbose: Print progress
            
        Returns:
            Dictionary with group CV scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        gkf = GroupKFoldCV(n_splits=n_splits)
        cv_scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring=scoring, n_jobs=-1)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        if verbose:
            print(f"Group K-Fold CV ({n_splits} folds):")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'n_splits': n_splits,
            'n_groups': len(np.unique(groups))
        }
    
    def time_series_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        scoring: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Time Series Cross-Validation
        
        Forward chaining for time series data
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            n_splits: Number of splits
            scoring: Scoring metric
            verbose: Print progress
            
        Returns:
            Dictionary with time series CV scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'neg_mean_squared_error'
        
        tscv = TimeSeriesCV(n_splits=n_splits)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        if verbose:
            print(f"Time Series CV ({n_splits} splits):")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'n_splits': n_splits
        }
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model to avoid modifying the original"""
        if hasattr(model, 'get_params'):
            from sklearn.base import clone
            return clone(model)
        else:
            # Simple copy for non-sklearn models
            import copy
            return copy.deepcopy(model)
