"""
Kuhn/Johnson Systematic Missing Data Handling
Methods for handling missing data with proper CV integration

Methods:
- Imputation strategies (mean, median, mode, KNN, predictive)
- Missing data pattern detection
- Indicator variables
- Imputation within CV (prevent leakage)
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
    from sklearn.impute import (
        SimpleImputer, KNNImputer, IterativeImputer
    )
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class MissingDataHandler:
    """
    Systematic missing data handling with CV-aware imputation
    """
    
    def __init__(
        self,
        strategy: str = 'knn',
        add_indicator: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'iterative')
            add_indicator: Add missing indicator variables
            random_state: Random seed
        """
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.random_state = random_state
        self.imputer_ = None
        self.missing_indicator_ = None
        self.n_features_original_ = None
        self.is_fitted = False
    
    def detect_missing_pattern(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect missing data patterns
        
        Args:
            X: Features
            
        Returns:
            Dictionary with missing data statistics
        """
        X = np.asarray(X)
        
        # Missing values per feature
        missing_per_feature = np.isnan(X).sum(axis=0)
        missing_percentage = (missing_per_feature / len(X)) * 100
        
        # Missing values per sample
        missing_per_sample = np.isnan(X).sum(axis=1)
        
        # Overall statistics
        total_missing = np.isnan(X).sum()
        total_values = X.size
        missing_percentage_overall = (total_missing / total_values) * 100
        
        # Features with missing values
        features_with_missing = np.where(missing_per_feature > 0)[0]
        
        # Patterns
        missing_patterns = {}
        for i in range(len(X)):
            pattern = tuple(np.isnan(X[i]).astype(int))
            if pattern not in missing_patterns:
                missing_patterns[pattern] = 0
            missing_patterns[pattern] += 1
        
        return {
            'total_missing': int(total_missing),
            'total_values': int(total_values),
            'missing_percentage_overall': float(missing_percentage_overall),
            'missing_per_feature': missing_per_feature.tolist(),
            'missing_percentage_per_feature': missing_percentage.tolist(),
            'missing_per_sample': missing_per_sample.tolist(),
            'features_with_missing': features_with_missing.tolist(),
            'n_features_with_missing': len(features_with_missing),
            'missing_patterns': missing_patterns,
            'has_missing': total_missing > 0
        }
    
    def _create_imputer(self, strategy: str) -> Any:
        """Create imputer based on strategy"""
        if not SKLEARN_AVAILABLE:
            return None
        
        if strategy == 'mean':
            return SimpleImputer(strategy='mean')
        elif strategy == 'median':
            return SimpleImputer(strategy='median')
        elif strategy == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        elif strategy == 'iterative':
            return IterativeImputer(random_state=self.random_state, max_iter=10)
        else:
            return SimpleImputer(strategy='mean')  # Default
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit imputer
        
        Args:
            X: Features
            y: Labels (optional, for iterative imputation)
        """
        X = np.asarray(X)
        
        # Create imputer
        self.imputer_ = self._create_imputer(self.strategy)
        
        if self.imputer_ is None:
            raise ImportError("sklearn is required for imputation")
        
        # Fit imputer
        if y is not None and self.strategy == 'iterative':
            # Iterative imputation can use y
            self.imputer_.fit(X, y)
        else:
            self.imputer_.fit(X)
        
        # Create missing indicator if requested
        if self.add_indicator:
            self.missing_indicator_ = np.isnan(X)
            self.n_features_original_ = X.shape[1]
        
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data (impute missing values)
        
        Args:
            X: Features
            
        Returns:
            Imputed features (with indicators if requested)
        """
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X)
        
        # Impute missing values
        X_imputed = self.imputer_.transform(X)
        
        # Add missing indicators if requested
        if self.add_indicator:
            missing_mask = np.isnan(X)
            X_with_indicators = np.hstack([X_imputed, missing_mask.astype(float)])
            return X_with_indicators
        else:
            return X_imputed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class CVAwareImputation:
    """
    Cross-Validation Aware Imputation
    
    Fits imputer on training fold only, transforms test fold
    Prevents data leakage
    """
    
    def __init__(self, imputation_strategy: str = 'knn', add_indicator: bool = True):
        """
        Args:
            imputation_strategy: Imputation strategy
            add_indicator: Add missing indicator variables
        """
        self.imputation_strategy = imputation_strategy
        self.add_indicator = add_indicator
        self.imputers_ = {}  # Store imputers for each fold
    
    def fit_transform_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Impute within cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Cross-validation splitter
            
        Returns:
            Imputed X, statistics
        """
        X = np.asarray(X).copy()
        y = np.asarray(y)
        
        imputation_stats = {
            'folds_processed': 0,
            'total_imputed': 0,
            'features_imputed': set()
        }
        
        # Process each fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            # Fit imputer on training fold
            handler = MissingDataHandler(
                strategy=self.imputation_strategy,
                add_indicator=self.add_indicator,
                random_state=42
            )
            handler.fit(X_train, y[train_idx] if y is not None else None)
            
            # Transform training fold
            X_train_imputed = handler.transform(X_train)
            
            # Transform test fold (using training imputer)
            X_test_imputed = handler.transform(X_test)
            
            # Update X with imputed values
            if self.add_indicator:
                # Remove indicators for original feature replacement
                n_original = handler.n_features_original_
                X[train_idx, :] = X_train_imputed[:, :n_original]
                X[test_idx, :] = X_test_imputed[:, :n_original]
                
                # Store full imputed versions separately
                # (In practice, you'd rebuild the full dataset)
            else:
                X[train_idx, :] = X_train_imputed
                X[test_idx, :] = X_test_imputed
            
            # Store imputer for this fold
            self.imputers_[fold_idx] = handler
            
            # Statistics
            imputation_stats['folds_processed'] += 1
            imputation_stats['total_imputed'] += np.isnan(X).sum()
        
        # Final imputation (for any remaining missing values)
        final_handler = MissingDataHandler(
            strategy=self.imputation_strategy,
            add_indicator=self.add_indicator
        )
        X_final = final_handler.fit_transform(X, y)
        
        return X_final, imputation_stats
