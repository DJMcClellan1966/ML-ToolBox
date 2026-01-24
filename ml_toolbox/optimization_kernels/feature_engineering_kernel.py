"""
Feature Engineering Kernel - Unified Feature Transformation Pipeline

Provides unified interface for all feature engineering operations with
automatic optimization and parallel processing.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor

from .kernel_optimizations import should_use_kernel, optimize_for_size

logger = logging.getLogger(__name__)


class FeatureEngineeringKernel:
    """
    Unified kernel for feature engineering
    
    Provides:
    - Unified feature transformation pipeline
    - Automatic feature selection
    - Parallel feature computation
    """
    
    def __init__(self, toolbox=None, parallel: bool = True):
        """
        Initialize feature engineering kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        parallel : bool, default=True
            Enable parallel processing
        """
        self.toolbox = toolbox
        self.parallel = parallel
        self._transformers = {}
        self._fitted = False
        
    def transform(self, X: np.ndarray, operations: List[str] = None, **kwargs) -> np.ndarray:
        """
        Transform features using specified operations
        
        Parameters
        ----------
        X : array-like
            Input features
        operations : list of str, optional
            Operations to apply (default: ['standardize', 'normalize'])
        **kwargs
            Additional transformation parameters
            
        Returns
        -------
        X_transformed : array-like
            Transformed features
        """
        X = np.asarray(X)
        
        # Check if kernel should be used (avoid overhead for small operations)
        if not should_use_kernel(X):
            # Use direct NumPy for small operations (faster due to no overhead)
            return self._direct_transform(X, operations)
        
        if operations is None:
            operations = ['standardize', 'normalize']
        
        result = X.copy()
        
        # Apply operations sequentially
        for op in operations:
            if op == 'standardize':
                result = self._standardize(result)
            elif op == 'normalize':
                result = self._normalize(result)
            elif op == 'select':
                result = self._select_features(result, **kwargs)
            elif op == 'polynomial':
                result = self._polynomial_features(result, **kwargs)
            else:
                logger.warning(f"Unknown operation: {op}")
        
        return result
    
    def _direct_transform(self, X: np.ndarray, operations: Optional[List[str]] = None) -> np.ndarray:
        """Direct NumPy transformation for small operations (no kernel overhead)"""
        if operations is None:
            operations = ['standardize']
        
        result = X.copy()
        for op in operations:
            if op == 'standardize':
                mean = np.mean(result, axis=0, keepdims=True)
                std = np.std(result, axis=0, keepdims=True)
                std = np.where(std < 1e-10, 1.0, std)
                result = (result - mean) / std
            elif op == 'normalize':
                min_val = np.min(result, axis=0, keepdims=True)
                max_val = np.max(result, axis=0, keepdims=True)
                range_val = max_val - min_val
                range_val = np.where(range_val < 1e-10, 1.0, range_val)
                result = (result - min_val) / range_val
        
        return result
    
    def auto_engineer(self, X: np.ndarray, y: np.ndarray, max_features: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Automatic feature engineering
        
        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target labels
        max_features : int, optional
            Maximum number of features to keep
        **kwargs
            Additional parameters
            
        Returns
        -------
        X_engineered : array-like
            Engineered features
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Check if kernel should be used
        if not should_use_kernel(X):
            # Use direct NumPy for small operations
            return self._direct_transform(X, ['standardize'])
        
        logger.info(f"[FeatureEngineeringKernel] Auto-engineering features from {X.shape[1]} to {max_features or 'auto'}")
        
        # Standardize
        X_std = self._standardize(X)
        
        # Select features if needed
        if max_features and X_std.shape[1] > max_features:
            X_std = self._select_features(X_std, y, max_features=max_features)
        
        return X_std
    
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Standardize features"""
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return (X - mean) / std
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features"""
        min_val = np.min(X, axis=0, keepdims=True)
        max_val = np.max(X, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-10, 1.0, range_val)
        return (X - min_val) / range_val
    
    def _select_features(self, X: np.ndarray, y: Optional[np.ndarray] = None, max_features: Optional[int] = None, **kwargs) -> np.ndarray:
        """Select best features"""
        if max_features is None or X.shape[1] <= max_features:
            return X
        
        # Simple variance-based selection
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        return X[:, top_indices]
    
    def _polynomial_features(self, X: np.ndarray, degree: int = 2, **kwargs) -> np.ndarray:
        """Create polynomial features"""
        # Simple polynomial features (interaction terms)
        n_samples, n_features = X.shape
        if degree == 2 and n_features < 10:
            # Add interaction terms
            interactions = []
            for i in range(n_features):
                for j in range(i, n_features):
                    interactions.append(X[:, i] * X[:, j])
            if interactions:
                return np.column_stack([X] + interactions)
        return X
