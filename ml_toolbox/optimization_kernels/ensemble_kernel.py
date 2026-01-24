"""
Ensemble Kernel - Unified Ensemble Methods

Provides unified interface for ensemble methods with parallel training
and smart model selection.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .kernel_optimizations import should_parallelize

logger = logging.getLogger(__name__)


class EnsembleKernel:
    """
    Unified kernel for ensemble methods
    
    Provides:
    - Unified ensemble interface
    - Parallel model training
    - Smart model selection
    """
    
    def __init__(self, toolbox=None, parallel: bool = True):
        """
        Initialize ensemble kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        parallel : bool, default=True
            Enable parallel training
        """
        self.toolbox = toolbox
        self.parallel = parallel
        self._models = {}
        self._fitted = False
        
    def create_ensemble(self, X: np.ndarray, y: np.ndarray, models: List[str] = None, method: str = 'voting', **kwargs) -> 'EnsembleKernel':
        """
        Create ensemble of models
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        models : list of str, optional
            Models to include (default: ['rf', 'svm', 'lr'])
        method : str, default='voting'
            Ensemble method ('voting', 'stacking', 'blending')
        **kwargs
            Additional parameters
            
        Returns
        -------
        self : EnsembleKernel
            Fitted ensemble
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if models is None:
            models = ['rf', 'svm', 'lr']
        
        logger.info(f"[EnsembleKernel] Creating {method} ensemble with {models}")
        
        # Only parallelize if batch size is large enough
        should_parallel = (self.parallel and 
                          should_parallelize(len(models)) and 
                          len(models) > 1)
        
        # Train models
        if should_parallel:
            # Parallel training
            with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
                futures = {executor.submit(self._train_model, X, y, model, **kwargs): model for model in models}
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        self._models[model_name] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to train {model_name}: {e}")
        else:
            # Sequential training (faster for small ensembles)
            for model in models:
                self._models[model] = self._train_model(X, y, model, **kwargs)
        
        self.method = method
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make ensemble predictions
        
        Parameters
        ----------
        X : array-like
            Input features
        **kwargs
            Additional parameters
            
        Returns
        -------
        predictions : array-like
            Ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = np.asarray(X)
        
        # Get predictions from all models
        predictions = []
        for model_name, model in self._models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif isinstance(model, dict) and 'model' in model:
                pred = model['model'].predict(X)
            else:
                continue
            predictions.append(pred)
        
        if not predictions:
            raise ValueError("No valid models in ensemble")
        
        # Combine predictions
        if self.method == 'voting':
            # Majority voting
            predictions = np.array(predictions)
            if len(predictions[0].shape) == 1:
                # Classification
                from scipy.stats import mode
                return mode(predictions, axis=0)[0].flatten()
            else:
                # Regression - average
                return np.mean(predictions, axis=0)
        else:
            # Default: average
            return np.mean(predictions, axis=0)
    
    def auto_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleKernel':
        """Automatic ensemble creation"""
        return self.create_ensemble(X, y, method='voting', **kwargs)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, model_type: str, **kwargs):
        """Train a single model"""
        if self.toolbox:
            result = self.toolbox.fit(X, y, task_type='auto', model_type=model_type, **kwargs)
            if isinstance(result, dict) and 'model' in result:
                return result['model']
            return result
        else:
            # Fallback
            from simple_ml_tasks import SimpleMLTasks
            tasks = SimpleMLTasks()
            return tasks.quick_train(X, y)
