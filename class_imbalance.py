"""
Kuhn/Johnson Class Imbalance Handling
Methods for handling imbalanced datasets

Methods:
- SMOTE (Synthetic Minority Oversampling)
- ROSE (Random Over-Sampling Examples)
- Cost-sensitive learning
- Downsampling
- Threshold tuning
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import Counter
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Try to import sklearn
try:
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    from sklearn.metrics import roc_curve, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class ClassImbalanceHandler:
    """
    Handle class imbalance using multiple strategies
    """
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Args:
            method: Balancing method ('smote', 'adasyn', 'borderline_smote', 
                    'random_undersample', 'smote_tomek', 'smote_enn', 
                    'cost_sensitive', 'none')
            random_state: Random seed
        """
        self.method = method
        self.random_state = random_state
        self.resampler = None
        self.class_weights_ = None
        self.is_fitted = False
    
    def detect_imbalance(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect class imbalance
        
        Args:
            y: Labels
            
        Returns:
            Dictionary with imbalance statistics
        """
        y = np.asarray(y)
        class_counts = Counter(y)
        total = len(y)
        
        # Calculate class ratios
        ratios = {cls: count / total for cls, count in class_counts.items()}
        
        # Identify majority and minority classes
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        # Imbalance ratio
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
        
        # Determine if imbalanced
        is_imbalanced = imbalance_ratio > 2.0  # Threshold
        
        return {
            'is_imbalanced': is_imbalanced,
            'imbalance_ratio': imbalance_ratio,
            'class_counts': dict(class_counts),
            'class_ratios': ratios,
            'majority_class': majority_class,
            'minority_class': minority_class,
            'n_classes': len(class_counts)
        }
    
    def _create_resampler(self, method: str):
        """Create resampling object"""
        if not IMBLEARN_AVAILABLE and method not in ['cost_sensitive', 'none']:
            raise ImportError(
                "imbalanced-learn is required for resampling methods. "
                "Install with: pip install imbalanced-learn"
            )
        
        if method == 'smote':
            k_neighbors = getattr(self, 'k_neighbors', 5)
            return SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        
        elif method == 'adasyn':
            return ADASYN(random_state=self.random_state)
        
        elif method == 'borderline_smote':
            return BorderlineSMOTE(random_state=self.random_state)
        
        elif method == 'random_undersample':
            return RandomUnderSampler(random_state=self.random_state)
        
        elif method == 'smote_tomek':
            return SMOTETomek(random_state=self.random_state)
        
        elif method == 'smote_enn':
            return SMOTEENN(random_state=self.random_state)
        
        elif method == 'cost_sensitive':
            return None  # Handled separately
        
        elif method == 'none':
            return None
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset
        
        Args:
            X: Features
            y: Labels
            method: Balancing method (overrides self.method if provided)
            sampling_strategy: Sampling strategy
            k_neighbors: Number of neighbors for SMOTE
            
        Returns:
            Balanced X, y
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Use provided method or default
        use_method = method if method is not None else self.method
        
        # Detect imbalance
        imbalance_info = self.detect_imbalance(y)
        
        if not imbalance_info['is_imbalanced']:
            warnings.warn("Dataset is not significantly imbalanced. Balancing may not be necessary.")
        
        # Cost-sensitive learning (no resampling)
        if use_method == 'cost_sensitive':
            self._compute_class_weights(y)
            self.is_fitted = True
            return X, y
        
        # No balancing
        if use_method == 'none':
            self.is_fitted = True
            return X, y
        
        # Resampling methods
        self.k_neighbors = k_neighbors
        resampler = self._create_resampler(use_method)
        
        if resampler is None:
            return X, y
        
        # Apply resampling
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        self.resampler = resampler
        self.is_fitted = True
        
        return X_resampled, y_resampled
    
    def _compute_class_weights(self, y: np.ndarray):
        """Compute class weights for cost-sensitive learning"""
        if not SKLEARN_AVAILABLE:
            return
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights_ = dict(zip(classes, weights))
    
    def get_class_weights(self, y: Optional[np.ndarray] = None) -> Optional[Dict[int, float]]:
        """
        Get class weights for cost-sensitive learning
        
        Args:
            y: Labels (optional, for computing weights)
            
        Returns:
            Dictionary of class weights
        """
        if y is not None:
            self._compute_class_weights(y)
        
        return self.class_weights_
    
    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Get sample weights for cost-sensitive learning
        
        Args:
            y: Labels
            
        Returns:
            Sample weights
        """
        if not SKLEARN_AVAILABLE:
            return np.ones(len(y))
        
        if self.class_weights_ is None:
            self._compute_class_weights(y)
        
        return compute_sample_weight(self.class_weights_, y)


class ThresholdTuner:
    """
    Optimize classification threshold
    
    Find optimal threshold for binary classification based on
    various metrics (F1, precision, recall, etc.)
    """
    
    def __init__(self, metric: str = 'f1'):
        """
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')
        """
        self.metric = metric
        self.best_threshold_ = None
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find optimal classification threshold
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize (overrides self.metric)
            
        Returns:
            Dictionary with optimal threshold and scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        use_metric = metric if metric is not None else self.metric
        
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]  # Binary classification
        
        # Get thresholds from precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find best threshold based on metric
        if use_metric == 'f1':
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = f1_scores[best_idx]
        
        elif use_metric == 'precision':
            best_idx = np.argmax(precision)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = precision[best_idx]
        
        elif use_metric == 'recall':
            best_idx = np.argmax(recall)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = recall[best_idx]
        
        else:
            # Default to F1
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = f1_scores[best_idx]
        
        self.best_threshold_ = best_threshold
        
        return {
            'optimal_threshold': float(best_threshold),
            'best_score': float(best_score),
            'metric': use_metric,
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    
    def apply_threshold(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Apply threshold to probabilities
        
        Args:
            y_proba: Predicted probabilities
            threshold: Threshold to apply (uses best_threshold_ if None)
            
        Returns:
            Binary predictions
        """
        y_proba = np.asarray(y_proba)
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        use_threshold = threshold if threshold is not None else self.best_threshold_
        
        if use_threshold is None:
            use_threshold = 0.5
        
        return (y_proba >= use_threshold).astype(int)
