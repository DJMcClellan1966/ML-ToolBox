"""
ML Toolbox Optimization Module
Model compression, calibration, and optimization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ml_toolbox.optimization.model_compression import ModelCompressor as ModelCompression
    from ml_toolbox.optimization.model_calibration import ModelCalibrator as ModelCalibration
    __all__ = ['ModelCompression', 'ModelCalibration']
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Optimization module imports failed: {e}")
