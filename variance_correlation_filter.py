"""
Kuhn/Johnson Near-Zero Variance & Correlation Filtering
Remove uninformative and highly correlated features

Methods:
- Near-zero variance detection
- Correlation filtering
- Percent unique values
- Frequency ratio
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import Counter
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")


class VarianceCorrelationFilter:
    """
    Filter features based on variance and correlation
    """
    
    def __init__(
        self,
        remove_nzv: bool = True,
        nzv_threshold: float = 0.01,
        remove_high_correlation: bool = True,
        corr_threshold: float = 0.95,
        percent_unique_threshold: float = 0.01
    ):
        """
        Args:
            remove_nzv: Remove near-zero variance features
            nzv_threshold: Variance threshold (features with variance < threshold are removed)
            remove_high_correlation: Remove highly correlated features
            corr_threshold: Correlation threshold (features with correlation > threshold)
            percent_unique_threshold: Minimum percent unique values
        """
        self.remove_nzv = remove_nzv
        self.nzv_threshold = nzv_threshold
        self.remove_high_correlation = remove_high_correlation
        self.corr_threshold = corr_threshold
        self.percent_unique_threshold = percent_unique_threshold
        
        self.nzv_features_ = None
        self.correlated_features_ = None
        self.selected_features_ = None
        self.is_fitted = False
    
    def _detect_nzv(self, X: np.ndarray) -> np.ndarray:
        """Detect near-zero variance features"""
        # Calculate variance per feature
        variances = np.var(X, axis=0)
        
        # Calculate percent unique values
        n_samples = X.shape[0]
        percent_unique = np.array([
            len(np.unique(X[:, i])) / n_samples for i in range(X.shape[1])
        ])
        
        # Calculate frequency ratio (most common / second most common)
        frequency_ratios = []
        for i in range(X.shape[1]):
            counter = Counter(X[:, i])
            if len(counter) >= 2:
                sorted_counts = sorted(counter.values(), reverse=True)
                ratio = sorted_counts[0] / sorted_counts[1]
            else:
                ratio = np.inf  # Only one unique value
            frequency_ratios.append(ratio)
        
        frequency_ratios = np.array(frequency_ratios)
        
        # Identify NZV features
        nzv_mask = (
            (variances < self.nzv_threshold) |
            (percent_unique < self.percent_unique_threshold) |
            (frequency_ratios > 95)  # One value dominates
        )
        
        return np.where(nzv_mask)[0]
    
    def _detect_high_correlation(self, X: np.ndarray) -> np.ndarray:
        """Detect highly correlated features"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        highly_correlated_pairs = []
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > self.corr_threshold:
                    highly_correlated_pairs.append((i, j, corr_matrix[i, j]))
        
        # Remove one feature from each pair
        # Strategy: keep feature with lower index
        features_to_remove = set()
        
        for i, j, corr in highly_correlated_pairs:
            # Keep lower index, remove higher index
            features_to_remove.add(j)
        
        return np.array(list(features_to_remove))
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit filter
        
        Args:
            X: Features
            y: Labels (not used, for sklearn compatibility)
        """
        X = np.asarray(X)
        
        # Detect NZV features
        if self.remove_nzv:
            self.nzv_features_ = self._detect_nzv(X)
        else:
            self.nzv_features_ = np.array([])
        
        # Detect highly correlated features
        if self.remove_high_correlation:
            self.correlated_features_ = self._detect_high_correlation(X)
        else:
            self.correlated_features_ = np.array([])
        
        # Combine features to remove
        all_features_to_remove = np.unique(
            np.concatenate([self.nzv_features_, self.correlated_features_])
        )
        
        # Selected features
        all_features = np.arange(X.shape[1])
        self.selected_features_ = np.setdiff1d(all_features, all_features_to_remove)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to selected features"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X)
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get mask or indices of selected features
        
        Args:
            indices: If True, return indices; if False, return boolean mask
        """
        if not self.is_fitted:
            raise ValueError("Must fit before get_support")
        
        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(self.selected_features_.max() + 1, dtype=bool)
            mask[self.selected_features_] = True
            return mask
    
    def get_removed_features_info(self) -> Dict[str, Any]:
        """Get information about removed features"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_removed_features_info")
        
        return {
            'nzv_features': self.nzv_features_.tolist() if self.nzv_features_ is not None else [],
            'correlated_features': self.correlated_features_.tolist() if self.correlated_features_ is not None else [],
            'selected_features': self.selected_features_.tolist(),
            'n_removed': len(self.nzv_features_) + len(self.correlated_features_),
            'n_selected': len(self.selected_features_)
        }
