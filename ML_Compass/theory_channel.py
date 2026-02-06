"""
Theory-as-Channel: ensemble error correction and channel capacity.
"""
import numpy as np
from typing import List, Any, Dict

try:
    from ml_toolbox.textbook_concepts.communication_theory import (
        ErrorCorrectingPredictions,
        channel_capacity,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def correct_predictions(
    predictions: np.ndarray,
    method: str = "majority_vote",
) -> np.ndarray:
    """
    Correct predictions from an ensemble (e.g. majority vote).
    predictions shape: (n_samples, n_models).

    Returns:
        Corrected prediction per sample.
    """
    if not _AVAILABLE:
        raise ImportError("ml_toolbox.textbook_concepts.communication_theory not available")
    n_models = predictions.shape[1]
    ec = ErrorCorrectingPredictions(redundancy_factor=n_models)
    return ec.correct_predictions(predictions, method=method)


def channel_capacity_bits(
    signal_power: float = 10.0,
    noise_power: float = 1.0,
    bandwidth: float = 1.0,
) -> float:
    """Shannon capacity C = B * log2(1 + S/N)."""
    if not _AVAILABLE:
        raise ImportError("ml_toolbox.textbook_concepts.communication_theory not available")
    return float(channel_capacity(signal_power, noise_power, bandwidth))


def recommend_redundancy(
    signal_power: float = 10.0,
    noise_power: float = 1.0,
    min_models: int = 3,
    max_models: int = 7,
) -> Dict[str, Any]:
    """
    Suggest number of ensemble models from a simple capacity view.
    Returns recommendation and capacity in bits.
    """
    C = channel_capacity_bits(signal_power, noise_power)
    # Heuristic: more noise -> more redundancy
    if noise_power >= signal_power:
        n = max_models
    elif noise_power >= 0.5 * signal_power:
        n = 5
    else:
        n = min_models
    n = max(min_models, min(max_models, n))
    return {
        "recommended_models": n,
        "channel_capacity_bits": C,
        "signal_power": signal_power,
        "noise_power": noise_power,
    }


def ensemble_predict_and_correct(
    models: List[Any],
    X: np.ndarray,
    method: str = "majority_vote",
) -> np.ndarray:
    """
    Get predictions from each model and return corrected (e.g. majority vote) predictions.
    """
    if not _AVAILABLE:
        raise ImportError("ml_toolbox.textbook_concepts.communication_theory not available")
    preds = np.column_stack([m.predict(X) for m in models])
    return correct_predictions(preds, method=method)
