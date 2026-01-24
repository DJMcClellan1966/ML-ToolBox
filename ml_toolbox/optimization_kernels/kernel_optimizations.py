"""
Kernel Optimizations - Fix Small Operations, Small Batches, and Timing Issues

Provides optimizations to handle:
1. Small operations (overhead larger than operation)
2. Small batches (parallel overhead)
3. Too fast to measure (timing accuracy)
"""
import numpy as np
from typing import Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

# Configuration constants
MIN_SIZE_FOR_KERNEL = 100  # Minimum array size to use kernel
MIN_BATCH_SIZE_FOR_PARALLEL = 10  # Minimum batch size for parallel processing
MIN_TIME_FOR_ACCURATE_MEASUREMENT = 0.001  # 1ms minimum for accurate timing


def should_use_kernel(X: np.ndarray, min_size: int = MIN_SIZE_FOR_KERNEL) -> bool:
    """
    Determine if kernel should be used based on data size
    
    For very small operations, direct NumPy is faster due to
    kernel overhead being larger than the operation itself.
    
    Parameters
    ----------
    X : array-like
        Input data
    min_size : int, default=100
        Minimum size to use kernel
        
    Returns
    -------
    use_kernel : bool
        Whether to use kernel
    """
    if not hasattr(X, 'size'):
        return True  # Unknown size, use kernel
    
    return X.size >= min_size


def should_parallelize(batch_size: int, min_batch: int = MIN_BATCH_SIZE_FOR_PARALLEL) -> bool:
    """
    Determine if parallel processing should be used
    
    For small batches, parallel overhead (thread creation, synchronization)
    can be larger than the actual work.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    min_batch : int, default=10
        Minimum batch size for parallel processing
        
    Returns
    -------
    parallelize : bool
        Whether to use parallel processing
    """
    return batch_size >= min_batch


def accurate_timing(func, *args, min_time: float = MIN_TIME_FOR_ACCURATE_MEASUREMENT, 
                    min_iterations: int = 1, max_iterations: int = 100, **kwargs) -> Tuple[float, int]:
    """
    Accurate timing for fast operations
    
    For operations that complete too quickly to measure accurately,
    run multiple iterations and average.
    
    Parameters
    ----------
    func : callable
        Function to time
    *args, **kwargs
        Arguments for function
    min_time : float, default=0.001
        Minimum time for accurate measurement (1ms)
    min_iterations : int, default=1
        Minimum iterations
    max_iterations : int, default=100
        Maximum iterations
        
    Returns
    -------
    avg_time : float
        Average time per iteration
    iterations : int
        Number of iterations used
    """
    # First, try single iteration
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    # If too fast, run multiple iterations
    if elapsed < min_time:
        iterations = max(min_iterations, int(min_time / elapsed) + 1)
        iterations = min(iterations, max_iterations)
        
        start = time.perf_counter()
        for _ in range(iterations):
            result = func(*args, **kwargs)
        total_time = time.perf_counter() - start
        avg_time = total_time / iterations
    else:
        iterations = 1
        avg_time = elapsed
    
    return avg_time, iterations


def optimize_for_size(X: np.ndarray, operation_type: str = 'general') -> dict:
    """
    Get optimization parameters based on data size
    
    Parameters
    ----------
    X : array-like
        Input data
    operation_type : str
        Type of operation ('general', 'preprocessing', 'training', 'inference')
        
    Returns
    -------
    config : dict
        Optimization configuration
    """
    if not hasattr(X, 'size'):
        return {
            'use_kernel': True,
            'parallel': True,
            'batch_size': None
        }
    
    size = X.size
    
    config = {
        'use_kernel': size >= MIN_SIZE_FOR_KERNEL,
        'parallel': size >= MIN_SIZE_FOR_KERNEL * 10,  # Larger threshold for parallel
        'batch_size': None
    }
    
    # Operation-specific thresholds
    if operation_type == 'preprocessing':
        config['use_kernel'] = size >= 50  # Lower threshold for preprocessing
    elif operation_type == 'training':
        config['use_kernel'] = size >= 200  # Higher threshold for training
    elif operation_type == 'inference':
        config['use_kernel'] = size >= 100  # Standard threshold
    
    # Determine batch size
    if hasattr(X, 'shape') and len(X.shape) > 0:
        n_samples = X.shape[0]
        if n_samples > 1000:
            config['batch_size'] = 100
        elif n_samples > 100:
            config['batch_size'] = 50
        else:
            config['batch_size'] = None  # No batching for small datasets
    
    return config
