"""
Lazy Loader
Lazy loading system for features - only load when accessed

Improvement: Faster startup, less memory usage
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent))


class LazyLoader:
    """
    Lazy Loader
    
    Improvement: Features only loaded when accessed
    """
    
    def __init__(self):
        self._cache = {}
        self._loaders = {}
    
    def register(self, name: str, loader: Callable):
        """Register a lazy loader"""
        self._loaders[name] = loader
    
    def get(self, name: str) -> Any:
        """Get feature, loading if needed"""
        if name not in self._cache:
            if name in self._loaders:
                self._cache[name] = self._loaders[name]()
            else:
                raise AttributeError(f"Feature '{name}' not registered")
        return self._cache[name]
    
    def is_loaded(self, name: str) -> bool:
        """Check if feature is loaded"""
        return name in self._cache
    
    def clear_cache(self):
        """Clear loaded features cache"""
        self._cache.clear()


def lazy_property(loader_func: Callable):
    """Decorator for lazy properties"""
    attr_name = f"_{loader_func.__name__}"
    
    @property
    @wraps(loader_func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, loader_func(self))
        return getattr(self, attr_name)
    
    return wrapper


# Example usage in MLToolbox:
"""
class MLToolbox:
    @lazy_property
    def predictive_intelligence(self):
        from revolutionary_features import get_predictive_intelligence
        return get_predictive_intelligence()
    
    @lazy_property
    def third_eye(self):
        from revolutionary_features import get_third_eye
        return get_third_eye()
"""
