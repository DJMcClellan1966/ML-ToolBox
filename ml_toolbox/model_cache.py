"""
Model Caching System
Caches trained models and preprocessing results for faster repeated operations
"""
import sys
from pathlib import Path
import hashlib
import pickle
import time
from typing import Any, Optional, Dict, Tuple
from functools import lru_cache
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available for model caching")


class ModelCache:
    """
    Model caching system for ML Toolbox
    
    Features:
    - Cache trained models by data and parameters hash
    - Cache preprocessing results
    - Automatic cache invalidation
    - Memory-efficient storage
    """
    
    def __init__(self, max_size: int = 100, enable_disk_cache: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize model cache
        
        Args:
            max_size: Maximum number of cached models (LRU eviction)
            enable_disk_cache: Enable disk-based caching
            cache_dir: Directory for disk cache (default: .ml_toolbox_cache)
        """
        self.max_size = max_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".ml_toolbox_cache")
        
        # In-memory cache (LRU)
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (model, timestamp)
        self._access_times: Dict[str, float] = {}  # key -> last access time
        
        # Disk cache setup
        if self.enable_disk_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def _generate_key(self, X: Any, y: Optional[Any] = None, params: Optional[Dict] = None, operation: str = "fit") -> str:
        """
        Generate cache key from data and parameters
        
        Args:
            X: Input data
            y: Target data (optional)
            params: Parameters dictionary
            operation: Operation name (fit, predict, etc.)
        
        Returns:
            Cache key string
        """
        try:
            # Convert to hashable format
            if NUMPY_AVAILABLE and isinstance(X, np.ndarray):
                X_hash = hashlib.md5(X.tobytes()).hexdigest()
            else:
                X_hash = hashlib.md5(str(X).encode()).hexdigest()
            
            if y is not None:
                if NUMPY_AVAILABLE and isinstance(y, np.ndarray):
                    y_hash = hashlib.md5(y.tobytes()).hexdigest()
                else:
                    y_hash = hashlib.md5(str(y).encode()).hexdigest()
            else:
                y_hash = "none"
            
            params_str = str(sorted(params.items())) if params else "none"
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Combine into final key
            key = f"{operation}_{X_hash}_{y_hash}_{params_hash}"
            return key
        except Exception as e:
            # Fallback to simple hash
            key_str = f"{operation}_{str(X)}_{str(y)}_{str(params)}"
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, X: Any, y: Optional[Any] = None, params: Optional[Dict] = None, operation: str = "fit") -> Optional[Any]:
        """
        Get cached model or result
        
        Args:
            X: Input data
            y: Target data (optional)
            params: Parameters dictionary
            operation: Operation name
        
        Returns:
            Cached model/result or None if not found
        """
        self.stats['total_requests'] += 1
        
        key = self._generate_key(X, y, params, operation)
        
        # Check in-memory cache
        if key in self._cache:
            self.stats['hits'] += 1
            self._access_times[key] = time.time()
            return self._cache[key][0]  # Return model
        
        # Check disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        model = pickle.load(f)
                    # Load into memory cache
                    self._cache[key] = (model, time.time())
                    self._access_times[key] = time.time()
                    self.stats['hits'] += 1
                    return model
                except Exception as e:
                    warnings.warn(f"Failed to load from disk cache: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def set(self, X: Any, y: Optional[Any] = None, params: Optional[Dict] = None, 
            operation: str = "fit", model: Any = None):
        """
        Cache a model or result
        
        Args:
            X: Input data
            y: Target data (optional)
            params: Parameters dictionary
            operation: Operation name
            model: Model or result to cache
        """
        if model is None:
            return
        
        key = self._generate_key(X, y, params, operation)
        
        # Evict if cache is full (LRU)
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Find least recently used
            lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[lru_key]
            del self._access_times[lru_key]
            self.stats['evictions'] += 1
        
        # Store in memory
        self._cache[key] = (model, time.time())
        self._access_times[key] = time.time()
        
        # Store on disk if enabled
        if self.enable_disk_cache:
            try:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                warnings.warn(f"Failed to save to disk cache: {e}")
    
    def clear(self):
        """Clear all caches"""
        self._cache.clear()
        self._access_times.clear()
        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.stats['hits'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0.0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'total_requests': self.stats['total_requests'],
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_size': self.max_size
        }


# Global cache instance
_global_cache: Optional[ModelCache] = None

def get_model_cache(max_size: int = 100, enable_disk_cache: bool = False) -> ModelCache:
    """Get global model cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCache(max_size=max_size, enable_disk_cache=enable_disk_cache)
    return _global_cache
