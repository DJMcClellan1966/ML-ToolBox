"""
Serving Kernel - Unified Model Serving

Provides unified interface for model serving with batch inference and
parallel prediction.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor

from .kernel_optimizations import should_parallelize, optimize_for_size

logger = logging.getLogger(__name__)


class ServingKernel:
    """
    Unified kernel for model serving
    
    Provides:
    - Unified serving interface
    - Batch inference
    - Parallel prediction
    """
    
    def __init__(self, parallel: bool = True):
        """
        Initialize serving kernel
        
        Parameters
        ----------
        parallel : bool, default=True
            Enable parallel serving
        """
        self.parallel = parallel
        self._models = {}
        
    def serve(self, model: Any, X: np.ndarray, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Serve predictions from model
        
        Parameters
        ----------
        model : Any
            Trained model
        X : array-like
            Input features
        batch_size : int, optional
            Batch size for processing (auto-determined if None)
        **kwargs
            Additional parameters
            
        Returns
        -------
        predictions : array-like
            Predictions
        """
        X = np.asarray(X)
        
        # Auto-determine batch size if not provided
        if batch_size is None:
            config = optimize_for_size(X, operation_type='inference')
            batch_size = config.get('batch_size')
        
        # For small batches, use direct prediction (no batching overhead)
        if batch_size is None or len(X) <= batch_size:
            return self._predict(model, X, **kwargs)
        
        # Batch processing for larger datasets
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            pred = self._predict(model, batch, **kwargs)
            predictions.append(pred)
        return np.concatenate(predictions, axis=0)
    
    def batch_serve(self, models: List[Any], X: np.ndarray, **kwargs) -> List[np.ndarray]:
        """
        Serve predictions from multiple models in parallel
        
        Parameters
        ----------
        models : list
            List of trained models
        X : array-like
            Input features
        **kwargs
            Additional parameters
            
        Returns
        -------
        predictions : list of array-like
            Predictions from each model
        """
        X = np.asarray(X)
        
        # Only parallelize if batch size is large enough
        if self.parallel and should_parallelize(len(models)) and len(models) > 1:
            # Parallel serving
            with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
                futures = [executor.submit(self._predict, model, X, **kwargs) for model in models]
                return [f.result() for f in futures]
        else:
            # Sequential serving (faster for small batches)
            return [self._predict(model, X, **kwargs) for model in models]
    
    def _predict(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make prediction from model"""
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif isinstance(model, dict) and 'model' in model:
            return model['model'].predict(X)
        else:
            raise ValueError("Model does not support prediction")
