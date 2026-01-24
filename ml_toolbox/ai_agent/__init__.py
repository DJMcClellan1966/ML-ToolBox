"""
ML Toolbox AI Agent Module
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .agent import MLCodeAgent
    from .proactive_agent import ProactiveAgent, get_proactive_agent
    __all__ = ['MLCodeAgent', 'ProactiveAgent', 'get_proactive_agent']
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"AI Agent module imports failed: {e}")
