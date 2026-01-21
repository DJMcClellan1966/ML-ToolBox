"""
Kuhn/Johnson Variable Importance Analysis
Multiple methods for analyzing feature importance

Methods:
- Permutation importance (model-agnostic)
- Built-in importance (model-specific)
- SHAP values (if available)
- Stability analysis (across CV folds)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Don't print warning - SHAP is optional


class VariableImportanceAnalyzer:
    """
    Analyze variable importance using multiple methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None,
        n_repeats: int = 10,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Calculate permutation importance (model-agnostic)
        
        Args:
            model: Fitted model
            X: Features
            y: Labels
            scoring: Scoring metric
            n_repeats: Number of permutation repeats
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with importance scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=n_jobs
        )
        
        # Get importance scores
        importances_mean = perm_importance.importances_mean
        importances_std = perm_importance.importances_std
        
        # Create rankings
        feature_rankings = np.argsort(importances_mean)[::-1]
        
        return {
            'importances_mean': importances_mean.tolist(),
            'importances_std': importances_std.tolist(),
            'feature_rankings': feature_rankings.tolist(),
            'n_features': len(importances_mean),
            'method': 'permutation',
            'scoring': scoring
        }
    
    def builtin_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract built-in feature importance (model-specific)
        
        Works for tree-based models (Random Forest, Gradient Boosting, etc.)
        
        Args:
            model: Fitted model with feature_importances_ attribute
            feature_names: Optional feature names
            
        Returns:
            Dictionary with importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            return {
                'error': 'Model does not have feature_importances_ attribute',
                'method': 'builtin'
            }
        
        importances = model.feature_importances_
        
        # Create rankings
        feature_rankings = np.argsort(importances)[::-1]
        
        result = {
            'importances': importances.tolist(),
            'feature_rankings': feature_rankings.tolist(),
            'n_features': len(importances),
            'method': 'builtin'
        }
        
        if feature_names is not None:
            result['feature_names'] = feature_names
            result['importance_dict'] = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        
        return result
    
    def shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values (if available)
        
        Args:
            model: Fitted model
            X: Features
            feature_names: Optional feature names
            n_samples: Number of samples to use (None for all)
            
        Returns:
            Dictionary with SHAP importance scores
        """
        if not SHAP_AVAILABLE:
            return {
                'error': 'SHAP not available. Install with: pip install shap',
                'method': 'shap'
            }
        
        X = np.asarray(X)
        
        # Subsample if needed
        if n_samples is not None and len(X) > n_samples:
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(
                lambda x: model.predict_proba(x) if hasattr(model, 'predict_proba') else model.predict(x),
                X_sample[:100]  # Background data
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class (shap_values is list)
            if isinstance(shap_values, list):
                shap_values = np.abs(shap_values[0])  # Use first class
            
            # Calculate mean absolute SHAP values (importance)
            importances = np.mean(np.abs(shap_values), axis=0)
            
            # Create rankings
            feature_rankings = np.argsort(importances)[::-1]
            
            result = {
                'importances': importances.tolist(),
                'feature_rankings': feature_rankings.tolist(),
                'n_features': len(importances),
                'method': 'shap',
                'shap_values': shap_values.tolist() if len(shap_values) < 1000 else None
            }
            
            if feature_names is not None:
                result['feature_names'] = feature_names
                result['importance_dict'] = {
                    name: float(imp) for name, imp in zip(feature_names, importances)
                }
            
            return result
        
        except Exception as e:
            return {
                'error': f'SHAP calculation failed: {str(e)}',
                'method': 'shap'
            }
    
    def stability_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any = None,
        method: str = 'permutation',
        n_repeats: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze importance stability across CV folds
        
        Args:
            model: Model class (not fitted)
            X: Features
            y: Labels
            cv: Cross-validation splitter
            method: Importance method ('permutation' or 'builtin')
            n_repeats: Number of CV repeats
            
        Returns:
            Dictionary with stability statistics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if cv is None:
            if len(np.unique(y)) < 10:  # Classification
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        all_importances = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_test_cv = X[test_idx]
            y_test_cv = y[test_idx]
            
            # Fit model
            model_copy = self._clone_model(model)
            model_copy.fit(X_train_cv, y_train_cv)
            
            # Calculate importance
            if method == 'permutation':
                importance_result = self.permutation_importance(
                    model_copy, X_test_cv, y_test_cv, n_repeats=n_repeats
                )
                if 'error' not in importance_result:
                    all_importances.append(importance_result['importances_mean'])
            
            elif method == 'builtin':
                importance_result = self.builtin_importance(model_copy)
                if 'error' not in importance_result:
                    all_importances.append(importance_result['importances'])
        
        if len(all_importances) == 0:
            return {'error': 'Failed to calculate importance across folds'}
        
        all_importances = np.array(all_importances)
        
        # Calculate stability metrics
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        coefficient_of_variation = std_importance / (np.abs(mean_importance) + 1e-10)
        
        # Stable features (low CV)
        stable_threshold = np.percentile(coefficient_of_variation, 25)
        stable_features = np.where(coefficient_of_variation < stable_threshold)[0]
        
        return {
            'mean_importance': mean_importance.tolist(),
            'std_importance': std_importance.tolist(),
            'coefficient_of_variation': coefficient_of_variation.tolist(),
            'stable_features': stable_features.tolist(),
            'n_folds': len(all_importances),
            'method': method,
            'stability_threshold': float(stable_threshold)
        }
    
    def analyze(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        methods: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        include_shap: bool = False,
        include_stability: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive importance analysis using multiple methods
        
        Args:
            model: Fitted model
            X: Features
            y: Labels
            methods: List of methods to use (None for all available)
            feature_names: Optional feature names
            include_shap: Include SHAP values (slower)
            include_stability: Include stability analysis (slower)
            
        Returns:
            Dictionary with all importance analyses
        """
        if methods is None:
            methods = ['permutation', 'builtin']
            if include_shap:
                methods.append('shap')
        
        results = {
            'methods': methods,
            'n_features': X.shape[1],
            'feature_names': feature_names
        }
        
        # Permutation importance (always available)
        if 'permutation' in methods:
            perm_result = self.permutation_importance(model, X, y)
            if 'error' not in perm_result:
                results['permutation'] = perm_result
        
        # Built-in importance (if available)
        if 'builtin' in methods:
            builtin_result = self.builtin_importance(model, feature_names)
            if 'error' not in builtin_result:
                results['builtin'] = builtin_result
        
        # SHAP importance (if requested and available)
        if 'shap' in methods and include_shap:
            shap_result = self.shap_importance(model, X, feature_names)
            if 'error' not in shap_result:
                results['shap'] = shap_result
        
        # Stability analysis (if requested)
        if include_stability:
            stability_result = self.stability_analysis(model, X, y)
            if 'error' not in stability_result:
                results['stability'] = stability_result
        
        # Create combined ranking
        results['combined_ranking'] = self._combine_rankings(results)
        
        return results
    
    def _combine_rankings(self, results: Dict[str, Any]) -> List[int]:
        """Combine rankings from multiple methods"""
        rankings = []
        
        for method in ['permutation', 'builtin', 'shap']:
            if method in results and 'feature_rankings' in results[method]:
                rankings.append(results[method]['feature_rankings'])
        
        if len(rankings) == 0:
            return []
        
        # Average rankings
        rankings_array = np.array(rankings)
        mean_ranks = np.mean(rankings_array, axis=0)
        combined_ranking = np.argsort(mean_ranks).tolist()
        
        return combined_ranking
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model"""
        if hasattr(model, 'get_params'):
            from sklearn.base import clone
            return clone(model)
        else:
            import copy
            return copy.deepcopy(model)
