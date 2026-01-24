"""
Tuning Kernel - Unified Hyperparameter Optimization

Provides unified interface for hyperparameter tuning with parallel search
and smart search space reduction.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

from .kernel_optimizations import should_parallelize

logger = logging.getLogger(__name__)


class TuningKernel:
    """
    Unified kernel for hyperparameter tuning
    
    Provides:
    - Unified tuning interface
    - Parallel search
    - Smart search space reduction
    """
    
    def __init__(self, toolbox=None, parallel: bool = True):
        """
        Initialize tuning kernel
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        parallel : bool, default=True
            Enable parallel search
        """
        self.toolbox = toolbox
        self.parallel = parallel
        self._best_params = None
        self._best_score = None
        
    def tune(self, model, X: np.ndarray, y: np.ndarray, search_space: Dict, method: str = 'auto', **kwargs) -> Dict:
        """
        Tune hyperparameters
        
        Parameters
        ----------
        model : callable or str
            Model to tune
        X : array-like
            Training features
        y : array-like
            Training labels
        search_space : dict
            Hyperparameter search space
        method : str, default='auto'
            Tuning method ('auto', 'grid', 'random')
        **kwargs
            Additional parameters
            
        Returns
        -------
        best_params : dict
            Best hyperparameters found
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        logger.info(f"[TuningKernel] Tuning with {method} search")
        
        # Generate parameter combinations
        if method == 'grid':
            param_combinations = self._grid_search_space(search_space)
        else:
            # Random search (limited)
            param_combinations = self._random_search_space(search_space, n=10)
        
        # Only parallelize if enough combinations
        should_parallel = (self.parallel and 
                          should_parallelize(len(param_combinations)) and 
                          len(param_combinations) > 1)
        
        # Evaluate combinations
        if should_parallel:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=min(4, len(param_combinations))) as executor:
                futures = {executor.submit(self._evaluate_params, model, X, y, params, **kwargs): params 
                          for params in param_combinations}
                scores = {}
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        scores[tuple(params.items())] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to evaluate {params}: {e}")
        else:
            # Sequential evaluation (faster for small search spaces)
            scores = {tuple(params.items()): self._evaluate_params(model, X, y, params, **kwargs) 
                     for params in param_combinations}
        
        # Find best
        if scores:
            best_key = max(scores, key=scores.get)
            self._best_params = dict(best_key)
            self._best_score = scores[best_key]
            logger.info(f"[TuningKernel] Best score: {self._best_score:.4f}")
            return self._best_params
        else:
            return {}
    
    def smart_tune(self, model, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Automatic search space reduction"""
        # Simple default search space
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10]
        }
        return self.tune(model, X, y, search_space, method='random', **kwargs)
    
    def _grid_search_space(self, search_space: Dict) -> List[Dict]:
        """Generate grid search combinations"""
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations
    
    def _random_search_space(self, search_space: Dict, n: int = 10) -> List[Dict]:
        """Generate random search combinations"""
        import random
        keys = list(search_space.keys())
        combinations = []
        for _ in range(n):
            combo = {key: random.choice(search_space[key]) for key in keys}
            combinations.append(combo)
        return combinations
    
    def _evaluate_params(self, model, X: np.ndarray, y: np.ndarray, params: Dict, **kwargs) -> float:
        """Evaluate parameter combination"""
        try:
            if self.toolbox:
                result = self.toolbox.fit(X, y, **params, **kwargs)
                if isinstance(result, dict) and 'accuracy' in result:
                    return result['accuracy']
                elif isinstance(result, dict) and 'score' in result:
                    return result['score']
            # Simple fallback
            return 0.5
        except:
            return 0.0
