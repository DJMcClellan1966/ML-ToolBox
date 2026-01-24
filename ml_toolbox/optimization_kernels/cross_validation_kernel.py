"""
Cross-Validation Kernel - Unified Cross-Validation

Provides unified interface for cross-validation with parallel fold processing
and smart fold allocation.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .kernel_optimizations import should_parallelize

logger = logging.getLogger(__name__)


class CrossValidationKernel:
    """
    Unified kernel for cross-validation
    
    Provides:
    - Unified CV interface
    - Parallel fold processing
    - Smart fold allocation
    """
    
    def __init__(self, toolbox=None, parallel: bool = True):
        """
        Initialize CV kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        parallel : bool, default=True
            Enable parallel fold processing
        """
        self.toolbox = toolbox
        self.parallel = parallel
        
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, **kwargs) -> Dict:
        """
        Perform cross-validation
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        cv : int, default=5
            Number of folds
        **kwargs
            Additional parameters
            
        Returns
        -------
        results : dict
            CV results with scores
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        logger.info(f"[CrossValidationKernel] Performing {cv}-fold CV")
        
        # Create folds
        folds = self._create_folds(X, y, cv)
        
        # Only parallelize if enough folds and data
        should_parallel = (self.parallel and 
                          should_parallelize(len(folds)) and 
                          len(folds) > 1 and
                          len(X) > 500)  # Only parallelize for larger datasets
        
        # Evaluate folds
        if should_parallel:
            # Parallel fold processing
            with ThreadPoolExecutor(max_workers=min(4, len(folds))) as executor:
                futures = {executor.submit(self._evaluate_fold, X, y, train_idx, val_idx, **kwargs): i 
                          for i, (train_idx, val_idx) in enumerate(folds)}
                scores = []
                for future in as_completed(futures):
                    fold_num = futures[future]
                    try:
                        score = future.result()
                        scores.append(score)
                    except Exception as e:
                        logger.error(f"Failed to evaluate fold {fold_num}: {e}")
        else:
            # Sequential fold processing (faster for small datasets)
            scores = [self._evaluate_fold(X, y, train_idx, val_idx, **kwargs) 
                     for train_idx, val_idx in folds]
        
        # Aggregate results
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_folds': len(scores)
        }
        
        return results
    
    def _create_folds(self, X: np.ndarray, y: np.ndarray, cv: int) -> List[tuple]:
        """Create CV folds"""
        n_samples = len(X)
        fold_size = n_samples // cv
        folds = []
        
        for i in range(cv):
            start = i * fold_size
            end = (i + 1) * fold_size if i < cv - 1 else n_samples
            val_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, n_samples)])
            folds.append((train_idx, val_idx))
        
        return folds
    
    def _evaluate_fold(self, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, **kwargs) -> float:
        """Evaluate a single fold"""
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        try:
            if self.toolbox:
                result = self.toolbox.fit(X_train, y_train, **kwargs)
                if isinstance(result, dict) and 'model' in result:
                    model = result['model']
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_val)
                    elif isinstance(model, dict) and 'model' in model:
                        pred = model['model'].predict(X_val)
                    else:
                        return 0.5
                    
                    # Calculate accuracy
                    if len(np.unique(y_val)) < 20:
                        # Classification
                        return np.mean(pred == y_val)
                    else:
                        # Regression
                        from sklearn.metrics import r2_score
                        return r2_score(y_val, pred)
            return 0.5
        except:
            return 0.0
