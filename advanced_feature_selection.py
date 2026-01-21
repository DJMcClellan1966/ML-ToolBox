"""
Kuhn/Johnson Advanced Feature Selection
Wrapper and embedded methods for feature selection

Methods:
- Forward selection
- Backward elimination
- Recursive Feature Elimination (RFE)
- Embedded methods (L1 regularization)
- Stability selection
- Feature selection within CV
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.feature_selection import (
        RFE, RFECV, SelectFromModel,
        SelectKBest, f_classif, f_regression,
        mutual_info_classif, mutual_info_regression
    )
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
    from sklearn.metrics import make_scorer, accuracy_score, r2_score
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class ForwardSelection:
    """
    Forward Feature Selection
    
    Start with no features, add one at a time based on performance
    """
    
    def __init__(
        self,
        estimator: Any,
        n_features_to_select: Optional[int] = None,
        scoring: Optional[Union[str, Callable]] = None,
        cv: int = 5,
        verbose: int = 0
    ):
        """
        Args:
            estimator: Base estimator
            n_features_to_select: Number of features to select (None for auto)
            scoring: Scoring metric
            cv: Cross-validation folds
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.selected_features_ = None
        self.feature_ranking_ = None
        self.cv_scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit forward selection"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for ForwardSelection")
        
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        # Determine number of features to select
        n_select = self.n_features_to_select if self.n_features_to_select is not None else n_features
        
        # Scoring
        if self.scoring is None:
            self.scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        # CV splitter
        if len(np.unique(y)) < 10:  # Classification
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        selected = []
        remaining = list(range(n_features))
        cv_scores = []
        
        for step in range(min(n_select, n_features)):
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining:
                # Try adding this feature
                candidate_features = selected + [feature]
                X_candidate = X[:, candidate_features]
                
                # Cross-validation score
                scores = cross_val_score(
                    self.estimator, X_candidate, y,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = feature
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                cv_scores.append(best_score)
                
                if self.verbose > 0:
                    print(f"Step {step+1}: Added feature {best_feature}, CV score: {best_score:.4f}")
        
        self.selected_features_ = np.array(selected)
        self.feature_ranking_ = np.array(selected)  # Ranking is order of selection
        self.cv_scores_ = np.array(cv_scores)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to selected features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class BackwardElimination:
    """
    Backward Feature Elimination
    
    Start with all features, remove one at a time
    """
    
    def __init__(
        self,
        estimator: Any,
        n_features_to_select: Optional[int] = None,
        scoring: Optional[Union[str, Callable]] = None,
        cv: int = 5,
        verbose: int = 0
    ):
        """
        Args:
            estimator: Base estimator
            n_features_to_select: Number of features to keep (None for auto)
            scoring: Scoring metric
            cv: Cross-validation folds
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.selected_features_ = None
        self.feature_ranking_ = None
        self.cv_scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit backward elimination"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for BackwardElimination")
        
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        # Determine number of features to keep
        n_keep = self.n_features_to_select if self.n_features_to_select is not None else 1
        
        # Scoring
        if self.scoring is None:
            self.scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        # CV splitter
        if len(np.unique(y)) < 10:  # Classification
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Start with all features
        selected = list(range(n_features))
        cv_scores = []
        eliminated = []
        
        # Baseline score with all features
        baseline_scores = cross_val_score(
            self.estimator, X, y,
            cv=cv_splitter,
            scoring=self.scoring,
            n_jobs=-1
        )
        cv_scores.append(np.mean(baseline_scores))
        
        # Eliminate features one by one
        while len(selected) > n_keep:
            best_score = -np.inf
            best_feature_to_remove = None
            
            for feature in selected:
                # Try removing this feature
                candidate_features = [f for f in selected if f != feature]
                X_candidate = X[:, candidate_features]
                
                # Cross-validation score
                scores = cross_val_score(
                    self.estimator, X_candidate, y,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature_to_remove = feature
            
            if best_feature_to_remove is not None:
                selected.remove(best_feature_to_remove)
                eliminated.append(best_feature_to_remove)
                cv_scores.append(best_score)
                
                if self.verbose > 0:
                    print(f"Step {len(eliminated)}: Removed feature {best_feature_to_remove}, "
                          f"CV score: {best_score:.4f}, Features remaining: {len(selected)}")
        
        self.selected_features_ = np.array(selected)
        self.feature_ranking_ = np.array(eliminated[::-1])  # Reverse order (first eliminated = least important)
        self.cv_scores_ = np.array(cv_scores)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to selected features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class StabilitySelection:
    """
    Stability Selection
    
    Select features that are consistently selected across CV folds
    """
    
    def __init__(
        self,
        estimator: Any,
        threshold: float = 0.6,
        cv: int = 5,
        n_bootstrap: int = 100
    ):
        """
        Args:
            estimator: Base estimator (should support feature selection)
            threshold: Minimum selection frequency to include feature
            cv: Cross-validation folds
            n_bootstrap: Number of bootstrap samples
        """
        self.estimator = estimator
        self.threshold = threshold
        self.cv = cv
        self.n_bootstrap = n_bootstrap
        self.selected_features_ = None
        self.selection_frequencies_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stability selection"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for StabilitySelection")
        
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        # Track feature selection across bootstrap samples
        feature_selections = np.zeros((self.n_bootstrap, n_features))
        
        np.random.seed(42)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Use RFE or embedded method
            if hasattr(self.estimator, 'feature_importances_'):
                # Tree-based: use feature importances
                model = type(self.estimator)(**self.estimator.get_params())
                model.fit(X_boot, y_boot)
                
                # Select top 50% of features
                importances = model.feature_importances_
                n_select = max(1, n_features // 2)
                top_features = np.argsort(importances)[-n_select:]
                feature_selections[i, top_features] = 1
            
            elif hasattr(self.estimator, 'coef_'):
                # Linear model: use coefficients
                model = type(self.estimator)(**self.estimator.get_params())
                model.fit(X_boot, y_boot)
                
                # Select features with non-zero coefficients
                if len(model.coef_.shape) > 1:
                    coef = np.abs(model.coef_).mean(axis=0)
                else:
                    coef = np.abs(model.coef_)
                
                n_select = max(1, n_features // 2)
                top_features = np.argsort(coef)[-n_select:]
                feature_selections[i, top_features] = 1
        
        # Calculate selection frequencies
        selection_frequencies = np.mean(feature_selections, axis=0)
        
        # Select features above threshold
        selected = np.where(selection_frequencies >= self.threshold)[0]
        
        self.selected_features_ = selected
        self.selection_frequencies_ = selection_frequencies
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to selected features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class AdvancedFeatureSelector:
    """
    Main class for advanced feature selection methods
    """
    
    def __init__(self, method: str = 'rfe', **kwargs):
        """
        Args:
            method: Selection method ('forward', 'backward', 'rfe', 'embedded', 'stability')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.selector_ = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: Optional[Any] = None,
        n_features_to_select: Optional[int] = None
    ):
        """
        Fit feature selector
        
        Args:
            X: Features
            y: Labels
            estimator: Base estimator (required for wrapper methods)
            n_features_to_select: Number of features to select
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for AdvancedFeatureSelector")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if estimator is None:
            # Default estimator
            if len(np.unique(y)) < 10:  # Classification
                from sklearn.linear_model import LogisticRegression
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                from sklearn.linear_model import LinearRegression
                estimator = LinearRegression()
        
        # Select method
        if self.method == 'forward':
            self.selector_ = ForwardSelection(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                **self.kwargs
            )
        
        elif self.method == 'backward':
            self.selector_ = BackwardElimination(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                **self.kwargs
            )
        
        elif self.method == 'rfe':
            n_features_to_select = n_features_to_select or (X.shape[1] // 2)
            self.selector_ = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                **self.kwargs
            )
        
        elif self.method == 'rfe_cv':
            self.selector_ = RFECV(
                estimator=estimator,
                **self.kwargs
            )
        
        elif self.method == 'embedded':
            # L1 regularization for feature selection
            if len(np.unique(y)) < 10:  # Classification
                estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
            else:
                estimator = LassoCV(random_state=42, cv=5)
            
            self.selector_ = SelectFromModel(estimator, **self.kwargs)
        
        elif self.method == 'stability':
            self.selector_ = StabilitySelection(
                estimator=estimator,
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit selector
        self.selector_.fit(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to selected features"""
        if self.selector_ is None:
            raise ValueError("Must fit before transform")
        return self.selector_.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, estimator: Optional[Any] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y, estimator).transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features"""
        if self.selector_ is None:
            raise ValueError("Must fit before get_support")
        
        if hasattr(self.selector_, 'get_support'):
            return self.selector_.get_support(indices=indices)
        elif hasattr(self.selector_, 'selected_features_'):
            if indices:
                return self.selector_.selected_features_
            else:
                mask = np.zeros(self.selector_.selected_features_.max() + 1, dtype=bool)
                mask[self.selector_.selected_features_] = True
                return mask
        else:
            raise AttributeError("Selector does not support get_support")
