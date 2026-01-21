"""
Kuhn/Johnson Model Calibration
Probability calibration for classification models

Methods:
- Platt scaling (logistic regression)
- Isotonic regression
- Calibration plots
- Brier score evaluation
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
    from sklearn.calibration import (
        CalibratedClassifierCV, calibration_curve
    )
    from sklearn.metrics import brier_score_loss
    from sklearn.base import BaseEstimator, ClassifierMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# Try to import matplotlib (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ModelCalibrator:
    """
    Calibrate classification model probabilities
    """
    
    def __init__(self, method: str = 'isotonic', cv: int = 5):
        """
        Args:
            method: Calibration method ('platt', 'isotonic')
            cv: Cross-validation folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrator_ = None
        self.is_fitted = False
    
    def calibrate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None
    ) -> Any:
        """
        Calibrate model probabilities
        
        Args:
            model: Fitted classification model
            X: Features
            y: Labels
            method: Calibration method (overrides self.method)
            
        Returns:
            Calibrated model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for model calibration")
        
        use_method = method if method is not None else self.method
        
        # Map method name
        sklearn_method = 'sigmoid' if use_method == 'platt' else 'isotonic'
        
        # Create calibrated classifier
        self.calibrator_ = CalibratedClassifierCV(
            base_estimator=model,
            method=sklearn_method,
            cv=self.cv
        )
        
        # Fit calibrator
        self.calibrator_.fit(X, y)
        self.is_fitted = True
        
        return self.calibrator_
    
    def evaluate_calibration(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality
        
        Args:
            model: Model (fitted, calibrated or not)
            X: Features
            y: Labels
            
        Returns:
            Dictionary with calibration metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]  # Binary classification
        else:
            return {'error': 'Model does not have predict_proba method'}
        
        # Brier score (lower is better)
        brier_score = brier_score_loss(y, proba)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, proba, n_bins=10
        )
        
        # Expected calibration error (ECE)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece),
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'proba_mean': float(np.mean(proba)),
            'proba_std': float(np.std(proba))
        }
    
    def plot_calibration(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot calibration curve
        
        Args:
            model: Model (fitted, calibrated or not)
            X: Features
            y: Labels
            save_path: Path to save figure (optional)
            show: Whether to display plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available for plotting")
            return
        
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available for calibration curve")
            return
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]
        else:
            print("Warning: Model does not have predict_proba method")
            return
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, proba, n_bins=10
        )
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration curve
        axes[0].plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Probability distribution
        axes[1].hist(proba, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Probability Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
