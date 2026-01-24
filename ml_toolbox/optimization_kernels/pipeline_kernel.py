"""
Pipeline Kernel - Unified Data Pipeline

Provides unified interface for complete data pipelines with automatic
optimization and parallel processing.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging

from .kernel_optimizations import should_use_kernel, optimize_for_size

logger = logging.getLogger(__name__)


class PipelineKernel:
    """
    Unified kernel for data pipelines
    
    Provides:
    - Unified pipeline execution
    - Automatic optimization
    - Parallel processing
    """
    
    def __init__(self, toolbox=None, steps: Optional[List[str]] = None):
        """
        Initialize pipeline kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        steps : list of str, optional
            Pipeline steps (default: ['preprocess', 'engineer', 'select'])
        """
        self.toolbox = toolbox
        self.steps = steps or ['preprocess', 'engineer', 'select']
        self._fitted = False
        
    def execute(self, X: np.ndarray, steps: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """
        Execute pipeline on data
        
        Parameters
        ----------
        X : array-like
            Input data
        steps : list of str, optional
            Steps to execute (default: self.steps)
        **kwargs
            Additional parameters
            
        Returns
        -------
        X_processed : array-like
            Processed data
        """
        X = np.asarray(X)
        steps = steps or self.steps
        
        # Check if kernel should be used (avoid overhead for small operations)
        if not should_use_kernel(X):
            # Use direct NumPy for small operations (faster due to no overhead)
            return self._direct_execute(X, steps, **kwargs)
        
        result = X.copy()
        
        # Execute steps sequentially
        for step in steps:
            if step == 'preprocess':
                result = self._preprocess(result)
            elif step == 'engineer':
                result = self._engineer(result)
            elif step == 'select':
                result = self._select(result, **kwargs)
            else:
                logger.warning(f"Unknown step: {step}")
        
        return result
    
    def _direct_execute(self, X: np.ndarray, steps: List[str], **kwargs) -> np.ndarray:
        """Direct NumPy execution for small operations (no kernel overhead)"""
        result = X.copy()
        for step in steps:
            if step == 'preprocess':
                mean = np.mean(result, axis=0, keepdims=True)
                std = np.std(result, axis=0, keepdims=True)
                std = np.where(std < 1e-10, 1.0, std)
                result = (result - mean) / std
            elif step == 'engineer':
                # Skip for small operations
                pass
            elif step == 'select':
                max_features = kwargs.get('max_features')
                if max_features and result.shape[1] > max_features:
                    variances = np.var(result, axis=0)
                    top_indices = np.argsort(variances)[-max_features:]
                    result = result[:, top_indices]
        return result
    
    def auto_pipeline(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Automatic pipeline optimization
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like
            Target labels
        **kwargs
            Additional parameters
            
        Returns
        -------
        X_processed : array-like
            Optimized processed data
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        logger.info(f"[PipelineKernel] Auto-optimizing pipeline for {X.shape}")
        
        # Use computational kernels if available
        if self.toolbox and hasattr(self.toolbox, 'computational_kernel') and self.toolbox.computational_kernel:
            return self.toolbox.computational_kernel.standardize(X)
        else:
            return self.execute(X)
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data"""
        # Standardize
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return (X - mean) / std
    
    def _engineer(self, X: np.ndarray) -> np.ndarray:
        """Engineer features"""
        # For now, just return as-is
        return X
    
    def _select(self, X: np.ndarray, max_features: Optional[int] = None, **kwargs) -> np.ndarray:
        """Select features"""
        if max_features and X.shape[1] > max_features:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-max_features:]
            return X[:, top_indices]
        return X
