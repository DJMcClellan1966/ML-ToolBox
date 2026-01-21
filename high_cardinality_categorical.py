"""
Kuhn/Johnson High-Cardinality Categorical Handling
Methods for handling categorical variables with many levels

Methods:
- Target encoding (mean encoding, leave-one-out)
- Feature hashing (hashing trick)
- Frequency encoding
- Rare category grouping
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import Counter, defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.feature_extraction import FeatureHasher
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class TargetEncoder:
    """
    Target Encoding (Mean Encoding)
    
    Encode categorical variables with mean of target variable
    """
    
    def __init__(self, smoothing: float = 1.0, leave_one_out: bool = False):
        """
        Args:
            smoothing: Smoothing parameter (higher = more global mean influence)
            leave_one_out: Use leave-one-out encoding (prevent overfitting)
        """
        self.smoothing = smoothing
        self.leave_one_out = leave_one_out
        self.encodings_ = {}
        self.global_mean_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit target encoder
        
        Args:
            X: Categorical features (1D array)
            y: Target variable
        """
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()
        
        # Calculate global mean
        self.global_mean_ = np.mean(y)
        
        # Calculate mean per category
        category_means = defaultdict(lambda: {'sum': 0, 'count': 0})
        
        for category, target in zip(X, y):
            category_means[category]['sum'] += target
            category_means[category]['count'] += 1
        
        # Calculate encodings
        for category, stats in category_means.items():
            category_mean = stats['sum'] / stats['count']
            
            # Smoothing: blend category mean with global mean
            n = stats['count']
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            
            self.encodings_[category] = smoothed_mean
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform categorical to target-encoded values
        
        Args:
            X: Categorical features
            y: Target variable (optional, for leave-one-out)
            
        Returns:
            Target-encoded values
        """
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X).ravel()
        
        if self.leave_one_out and y is not None:
            # Leave-one-out encoding
            y = np.asarray(y).ravel()
            encoded = np.zeros(len(X))
            
            for i, (category, target) in enumerate(zip(X, y)):
                # Calculate mean excluding current sample
                category_samples = np.where(X == category)[0]
                category_targets = y[category_samples]
                if len(category_targets) > 1:
                    category_mean = (np.sum(category_targets) - target) / (len(category_targets) - 1)
                else:
                    category_mean = self.global_mean_
                
                # Smoothing
                n = len(category_targets) - 1
                encoded[i] = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            
            return encoded
        else:
            # Regular encoding
            encoded = np.array([self.encodings_.get(cat, self.global_mean_) for cat in X])
            return encoded
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X, y if self.leave_one_out else None)


class FrequencyEncoder:
    """
    Frequency Encoding
    
    Encode categories by their frequency in the dataset
    """
    
    def __init__(self):
        self.frequencies_ = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit frequency encoder"""
        X = np.asarray(X).ravel()
        
        # Calculate frequencies
        counter = Counter(X)
        total = len(X)
        
        for category, count in counter.items():
            self.frequencies_[category] = count / total
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to frequencies"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X).ravel()
        encoded = np.array([self.frequencies_.get(cat, 0.0) for cat in X])
        return encoded
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class HighCardinalityHandler:
    """
    Handle high-cardinality categorical variables
    """
    
    def __init__(
        self,
        method: str = 'target_encoding',
        min_frequency: float = 0.01,
        n_hash_features: int = 16
    ):
        """
        Args:
            method: Encoding method ('target_encoding', 'hashing', 'frequency', 'one_hot')
            min_frequency: Minimum frequency to keep category separate (others grouped)
            n_hash_features: Number of hash features (for hashing method)
        """
        self.method = method
        self.min_frequency = min_frequency
        self.n_hash_features = n_hash_features
        self.encoders_ = {}
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[int]] = None
    ):
        """
        Fit encoders for categorical columns
        
        Args:
            X: Features (can have multiple categorical columns)
            y: Target variable (required for target encoding)
            columns: Column indices for categorical features (None = all columns)
        """
        X = np.asarray(X)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if columns is None:
            columns = list(range(X.shape[1]))
        
        for col_idx in columns:
            X_col = X[:, col_idx]
            
            # Group rare categories
            if self.min_frequency > 0:
                X_col = self._group_rare_categories(X_col)
            
            # Create encoder based on method
            if self.method == 'target_encoding':
                if y is None:
                    raise ValueError("y is required for target encoding")
                encoder = TargetEncoder(smoothing=1.0, leave_one_out=False)
                encoder.fit(X_col, y)
            
            elif self.method == 'frequency':
                encoder = FrequencyEncoder()
                encoder.fit(X_col)
            
            elif self.method == 'hashing':
                # Hashing doesn't need fitting
                encoder = None
            
            elif self.method == 'one_hot':
                # One-hot encoding (only for low cardinality)
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(X_col.reshape(-1, 1))
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.encoders_[col_idx] = {
                'encoder': encoder,
                'method': self.method
            }
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transform categorical features
        
        Args:
            X: Features
            y: Target variable (for leave-one-out target encoding)
            
        Returns:
            Encoded features
        """
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        encoded_features = []
        
        for col_idx, encoder_info in self.encoders_.items():
            X_col = X[:, col_idx]
            method = encoder_info['method']
            encoder = encoder_info['encoder']
            
            if method == 'target_encoding':
                encoded = encoder.transform(X_col, y)
                encoded_features.append(encoded.reshape(-1, 1))
            
            elif method == 'frequency':
                encoded = encoder.transform(X_col)
                encoded_features.append(encoded.reshape(-1, 1))
            
            elif method == 'hashing':
                # Feature hashing
                hasher = FeatureHasher(n_features=self.n_hash_features, input_type='string')
                # Convert to strings for hashing
                X_col_str = [str(x) for x in X_col]
                encoded = hasher.transform(X_col_str).toarray()
                encoded_features.append(encoded)
            
            elif method == 'one_hot':
                encoded = encoder.transform(X_col.reshape(-1, 1))
                encoded_features.append(encoded)
        
        # Combine all encoded features
        result = np.hstack(encoded_features)
        return result
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[int]] = None
    ) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y, columns).transform(X, y)
    
    def _group_rare_categories(self, X: np.ndarray) -> np.ndarray:
        """Group rare categories together"""
        X = np.asarray(X).ravel()
        
        # Calculate frequencies
        counter = Counter(X)
        total = len(X)
        min_count = int(self.min_frequency * total)
        
        # Find rare categories
        rare_categories = {cat for cat, count in counter.items() if count < min_count}
        
        if len(rare_categories) > 0:
            # Group rare categories
            X_grouped = X.copy()
            for rare_cat in rare_categories:
                X_grouped[X == rare_cat] = '__RARE__'
        
        return X
