"""
Kuhn/Johnson Model-Specific Preprocessing
Different preprocessing strategies for different model types

Model Types:
- Linear models (need centering/scaling)
- Tree models (don't need scaling)
- Distance-based (k-NN) (need spatial sign)
- Neural networks (need scaling)
- SVM (need scaling, optional normalization)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
        Normalizer, PowerTransformer, QuantileTransformer,
        PolynomialFeatures
    )
    from sklearn.decomposition import PCA
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class SpatialSignTransformer(BaseEstimator, TransformerMixin):
    """
    Spatial Sign Transformation
    
    Projects data onto unit sphere. Useful for distance-based models like k-NN.
    """
    
    def __init__(self):
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Transform data to spatial sign"""
        X = np.asarray(X)
        # Calculate L2 norm for each sample
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        # Project onto unit sphere
        X_transformed = X / norms
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class ModelSpecificPreprocessor:
    """
    Model-Specific Preprocessing
    
    Applies appropriate preprocessing based on model type
    """
    
    def __init__(self, model_type: str = 'auto', **kwargs):
        """
        Args:
            model_type: Type of model ('linear', 'tree', 'knn', 'svm', 'neural_net', 'auto')
            **kwargs: Additional preprocessing options
        """
        self.model_type = model_type
        self.preprocessor = None
        self._fit_preprocessor = None
        self.kwargs = kwargs
    
    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect model type from model object"""
        model_name = type(model).__name__.lower()
        
        # Linear models
        if any(x in model_name for x in ['linear', 'logistic', 'ridge', 'lasso', 'elastic', 'svd', 'pca']):
            return 'linear'
        
        # Tree models
        elif any(x in model_name for x in ['tree', 'forest', 'gradientboosting', 'xgboost', 'lightgbm', 'catboost']):
            return 'tree'
        
        # k-NN
        elif any(x in model_name for x in ['kneighbors', 'knn', 'neighbors']):
            return 'knn'
        
        # SVM
        elif any(x in model_name for x in ['svm', 'svc', 'svr', 'svm']):
            return 'svm'
        
        # Neural networks
        elif any(x in model_name for x in ['mlp', 'neural', 'perceptron', 'sequential']):
            return 'neural_net'
        
        # Default to linear (safest)
        else:
            return 'linear'
    
    def _create_preprocessor(self, model_type: str) -> Any:
        """Create appropriate preprocessor for model type"""
        if not SKLEARN_AVAILABLE:
            return None
        
        if model_type == 'linear':
            # Centering and scaling for linear models
            scaler_type = self.kwargs.get('scaler_type', 'standard')
            if scaler_type == 'standard':
                return StandardScaler()
            elif scaler_type == 'robust':
                return RobustScaler()
            elif scaler_type == 'minmax':
                return MinMaxScaler()
            else:
                return StandardScaler()
        
        elif model_type == 'tree':
            # Trees don't need scaling
            return None
        
        elif model_type == 'knn':
            # Spatial sign for distance-based models
            return SpatialSignTransformer()
        
        elif model_type == 'svm':
            # Scaling for SVM (robust to outliers)
            return RobustScaler()
        
        elif model_type == 'neural_net':
            # Scaling for neural networks
            scaler_type = self.kwargs.get('scaler_type', 'minmax')
            if scaler_type == 'standard':
                return StandardScaler()
            elif scaler_type == 'minmax':
                return MinMaxScaler()
            else:
                return MinMaxScaler()
        
        else:
            # Default to standard scaling
            return StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, model: Optional[Any] = None):
        """
        Fit preprocessor
        
        Args:
            X: Features
            y: Labels (optional)
            model: Model object (for auto-detection)
        """
        X = np.asarray(X)
        
        # Auto-detect model type if needed
        if self.model_type == 'auto' and model is not None:
            detected_type = self._detect_model_type(model)
            self.preprocessor = self._create_preprocessor(detected_type)
        elif self.model_type != 'auto':
            self.preprocessor = self._create_preprocessor(self.model_type)
        else:
            # Default to standard scaling
            self.preprocessor = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Fit preprocessor
        if self.preprocessor is not None:
            self._fit_preprocessor = self.preprocessor.fit(X)
        else:
            self._fit_preprocessor = None
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        X = np.asarray(X)
        
        if self._fit_preprocessor is not None:
            return self._fit_preprocessor.transform(X)
        else:
            # No preprocessing needed (e.g., trees)
            return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, model: Optional[Any] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y, model).transform(X)


class BoxCoxTransformer:
    """
    Box-Cox Transformation for Skewed Data
    
    Normalizes skewed distributions. Only works for positive values.
    """
    
    def __init__(self, standardize: bool = True):
        """
        Args:
            standardize: Whether to standardize after transformation
        """
        self.standardize = standardize
        self.lambdas_ = None
        self.scaler_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit transformer"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for BoxCoxTransformer")
        
        X = np.asarray(X)
        
        # Check for non-positive values
        if np.any(X <= 0):
            raise ValueError("Box-Cox transformation requires positive values")
        
        # Use sklearn's PowerTransformer with method='box-cox'
        from sklearn.preprocessing import PowerTransformer
        self.transformer_ = PowerTransformer(method='box-cox', standardize=self.standardize)
        self.transformer_.fit(X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        X = np.asarray(X)
        
        if np.any(X <= 0):
            # Use Yeo-Johnson instead (handles negative values)
            from sklearn.preprocessing import PowerTransformer
            transformer = PowerTransformer(method='yeo-johnson', standardize=self.standardize)
            transformer.fit(X)
            return transformer.transform(X)
        
        return self.transformer_.transform(X)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class YeoJohnsonTransformer:
    """
    Yeo-Johnson Transformation
    
    Extension of Box-Cox that handles negative values
    """
    
    def __init__(self, standardize: bool = True):
        """
        Args:
            standardize: Whether to standardize after transformation
        """
        self.standardize = standardize
        self.transformer_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit transformer"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for YeoJohnsonTransformer")
        
        from sklearn.preprocessing import PowerTransformer
        self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=self.standardize)
        self.transformer_.fit(X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.transformer_.transform(X)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class PreprocessingPipeline:
    """
    Complete Preprocessing Pipeline
    
    Applies multiple preprocessing steps in sequence
    """
    
    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps
        self.fitted_steps = []
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all preprocessing steps"""
        X = np.asarray(X)
        self.fitted_steps = []
        
        for name, transformer in self.steps:
            X = transformer.fit_transform(X, y)
            self.fitted_steps.append((name, transformer))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through pipeline"""
        X = np.asarray(X)
        
        for name, transformer in self.fitted_steps:
            X = transformer.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline(
    model_type: str = 'auto',
    handle_skewness: bool = False,
    apply_pca: bool = False,
    pca_components: Optional[int] = None,
    model: Optional[Any] = None,
    **kwargs
) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline for a specific model type
    
    Args:
        model_type: Type of model ('linear', 'tree', 'knn', 'svm', 'neural_net', 'auto')
        handle_skewness: Apply Box-Cox/Yeo-Johnson transformation
        apply_pca: Apply PCA dimensionality reduction
        pca_components: Number of PCA components (None for auto)
        model: Model object (for auto-detection)
        **kwargs: Additional preprocessing options
    
    Returns:
        PreprocessingPipeline object
    """
    steps = []
    
    # Handle skewness first
    if handle_skewness:
        use_yeo_johnson = kwargs.get('use_yeo_johnson', True)
        if use_yeo_johnson:
            steps.append(('yeo_johnson', YeoJohnsonTransformer()))
        else:
            steps.append(('box_cox', BoxCoxTransformer()))
    
    # Model-specific preprocessing
    model_preprocessor = ModelSpecificPreprocessor(model_type=model_type, **kwargs)
    if model_preprocessor._create_preprocessor(model_type) is not None:
        steps.append(('model_specific', model_preprocessor))
    
    # PCA (optional)
    if apply_pca and SKLEARN_AVAILABLE:
        pca = PCA(n_components=pca_components)
        steps.append(('pca', pca))
    
    return PreprocessingPipeline(steps)
