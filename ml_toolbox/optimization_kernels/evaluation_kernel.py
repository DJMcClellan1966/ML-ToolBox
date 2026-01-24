"""
Evaluation Kernel - Unified Model Evaluation

Provides unified interface for model evaluation with parallel metric computation.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class EvaluationKernel:
    """
    Unified kernel for model evaluation
    
    Provides:
    - Unified metrics interface
    - Parallel metric computation
    """
    
    def __init__(self, parallel: bool = True):
        """
        Initialize evaluation kernel
        
        Parameters
        ----------
        parallel : bool, default=True
            Enable parallel metric computation
        """
        self.parallel = parallel
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str] = None) -> Dict:
        """
        Evaluate predictions with multiple metrics
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        metrics : list of str, optional
            Metrics to compute (default: ['accuracy', 'precision', 'recall'])
            
        Returns
        -------
        results : dict
            Evaluation results
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if metrics is None:
            # Auto-detect metrics
            if len(np.unique(y_true)) < 20:
                metrics = ['accuracy', 'precision', 'recall']
            else:
                metrics = ['r2', 'mse', 'mae']
        
        results = {}
        
        # Compute metrics
        if self.parallel and len(metrics) > 1:
            # Parallel computation
            with ThreadPoolExecutor(max_workers=min(4, len(metrics))) as executor:
                futures = {executor.submit(self._compute_metric, y_true, y_pred, metric): metric 
                          for metric in metrics}
                for future in futures:
                    metric = futures[future]
                    try:
                        results[metric] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to compute {metric}: {e}")
        else:
            # Sequential computation
            for metric in metrics:
                results[metric] = self._compute_metric(y_true, y_pred, metric)
        
        return results
    
    def batch_evaluate(self, results_batch: List[tuple]) -> List[Dict]:
        """Batch evaluation of multiple results"""
        return [self.evaluate(y_true, y_pred) for y_true, y_pred in results_batch]
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Compute a single metric"""
        if metric == 'accuracy':
            return np.mean(y_true == y_pred)
        elif metric == 'precision':
            # Simple precision
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return 0.5
        elif metric == 'recall':
            # Simple recall
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 0.5
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        elif metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        else:
            return 0.0
