"""
Algorithm Kernel - Unified Interface for All ML Algorithms

Provides a single unified interface for all ML algorithms with automatic
selection, batch processing, and optimized execution.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class AlgorithmKernel:
    """
    Unified kernel for all ML algorithms
    
    Provides:
    - Single fit()/predict() interface for all algorithms
    - Automatic algorithm selection
    - Batch processing support
    - Parallel operations
    """
    
    def __init__(self, toolbox=None, parallel: bool = True):
        """
        Initialize algorithm kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        parallel : bool, default=True
            Enable parallel processing
        """
        self.toolbox = toolbox
        self.parallel = parallel
        self._models = {}
        self._fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, algorithm: str = 'auto', **kwargs) -> 'AlgorithmKernel':
        """
        Fit model using specified algorithm
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        algorithm : str, default='auto'
            Algorithm to use ('auto', 'rf', 'svm', 'lr', 'gb', etc.)
        **kwargs
            Additional algorithm parameters
            
        Returns
        -------
        self : AlgorithmKernel
            Fitted kernel
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Auto-select algorithm
        if algorithm == 'auto':
            algorithm = self._auto_select_algorithm(X, y)
        
        logger.info(f"[AlgorithmKernel] Fitting {algorithm} algorithm")
        
        # Fit model
        if self.toolbox:
            result = self.toolbox.fit(X, y, task_type='auto', model_type=algorithm, **kwargs)
            if isinstance(result, dict) and 'model' in result:
                self._models[algorithm] = result['model']
            else:
                self._models[algorithm] = result
        else:
            # Fallback to simple training
            self._models[algorithm] = self._simple_fit(X, y, algorithm, **kwargs)
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray, algorithm: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using fitted model
        
        Parameters
        ----------
        X : array-like
            Input features
        algorithm : str, optional
            Algorithm to use (uses first fitted if None)
        **kwargs
            Additional prediction parameters
            
        Returns
        -------
        predictions : array-like
            Predictions
        """
        if not self._fitted:
            raise ValueError("Kernel must be fitted before prediction. Call fit() first.")
        
        X = np.asarray(X)
        
        # Select algorithm
        if algorithm is None:
            algorithm = list(self._models.keys())[0] if self._models else 'auto'
        
        if algorithm not in self._models:
            raise ValueError(f"Algorithm {algorithm} not fitted. Available: {list(self._models.keys())}")
        
        model = self._models[algorithm]
        
        # Make prediction
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif isinstance(model, dict) and 'model' in model:
            return model['model'].predict(X)
        else:
            raise ValueError(f"Model {algorithm} does not support prediction")
    
    def batch_predict(self, X_batch: List[np.ndarray], algorithm: Optional[str] = None, **kwargs) -> List[np.ndarray]:
        """
        Batch prediction with parallel processing
        
        Parameters
        ----------
        X_batch : list of array-like
            Batch of input features
        algorithm : str, optional
            Algorithm to use
        **kwargs
            Additional prediction parameters
            
        Returns
        -------
        predictions : list of array-like
            Batch of predictions
        """
        if self.parallel and len(X_batch) > 1:
            # Parallel batch prediction
            with ThreadPoolExecutor(max_workers=min(4, len(X_batch))) as executor:
                futures = [executor.submit(self.predict, X, algorithm, **kwargs) for X in X_batch]
                return [f.result() for f in as_completed(futures)]
        else:
            # Sequential prediction
            return [self.predict(X, algorithm, **kwargs) for X in X_batch]
    
    def _auto_select_algorithm(self, X: np.ndarray, y: np.ndarray) -> str:
        """Auto-select best algorithm based on data characteristics"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Simple heuristic selection
        if n_samples < 1000:
            return 'rf'  # Random Forest for small datasets
        elif n_features > 100:
            return 'rf'  # Random Forest for high-dimensional
        elif n_classes == 2:
            return 'lr'  # Logistic Regression for binary
        else:
            return 'rf'  # Default to Random Forest
    
    def _simple_fit(self, X: np.ndarray, y: np.ndarray, algorithm: str, **kwargs):
        """Simple fallback training"""
        try:
            from simple_ml_tasks import SimpleMLTasks
            tasks = SimpleMLTasks()
            return tasks.quick_train(X, y)
        except:
            # Very simple fallback
            class SimpleModel:
                def __init__(self, X, y):
                    self.X = X
                    self.y = y
                def predict(self, X_new):
                    # Simple nearest neighbor
                    from scipy.spatial.distance import cdist
                    distances = cdist(X_new, self.X)
                    nearest = np.argmin(distances, axis=1)
                    return self.y[nearest]
            return SimpleModel(X, y)
    
    def get_model(self, algorithm: Optional[str] = None):
        """Get fitted model"""
        if algorithm is None:
            algorithm = list(self._models.keys())[0] if self._models else None
        return self._models.get(algorithm)
