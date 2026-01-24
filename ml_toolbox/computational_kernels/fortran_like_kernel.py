"""
Fortran-Like Computational Kernel

Mimics Fortran's performance characteristics:
- Optimized array operations
- Vectorization
- Memory-efficient processing
- BLAS/LAPACK integration
- Column-major operations (Fortran-style)

Without requiring actual Fortran code.
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)

# Try to use optimized BLAS/LAPACK
try:
    import scipy.linalg.blas as blas
    import scipy.linalg.lapack as lapack
    BLAS_AVAILABLE = True
except ImportError:
    BLAS_AVAILABLE = False
    warnings.warn("scipy not available. Using NumPy fallback for BLAS operations.")


class FortranLikeKernel:
    """
    Fortran-Like Computational Kernel
    
    Provides Fortran-like performance through:
    - Optimized vectorized operations
    - Memory-efficient array processing
    - BLAS/LAPACK integration
    - Column-major operations
    - SIMD optimizations
    """
    
    def __init__(self, use_blas: bool = True, parallel: bool = True):
        """
        Initialize Fortran-like kernel
        
        Parameters
        ----------
        use_blas : bool, default=True
            Use BLAS/LAPACK for matrix operations
        parallel : bool, default=True
            Enable parallel processing
        """
        self.use_blas = use_blas and BLAS_AVAILABLE
        self.parallel = parallel
        self._cache = {}
        
    def standardize(self, X: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Fast standardization (Fortran-like vectorized)
        
        Uses optimized vectorized operations similar to Fortran's
        array processing capabilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        axis : int, default=0
            Axis along which to standardize
            
        Returns
        -------
        X_std : array-like
            Standardized data
        """
        X = np.asarray(X, dtype=np.float64, order='F')  # Fortran order
        
        # Vectorized mean and std (Fortran-like efficiency)
        mean = np.mean(X, axis=axis, keepdims=True)
        std = np.std(X, axis=axis, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)
        
        # Vectorized standardization
        X_std = (X - mean) / std
        
        return X_std
    
    def normalize(self, X: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Fast min-max normalization (Fortran-like vectorized)
        
        Parameters
        ----------
        X : array-like
            Input data
        axis : int, default=0
            Axis along which to normalize
            
        Returns
        -------
        X_norm : array-like
            Normalized data (0-1 range)
        """
        X = np.asarray(X, dtype=np.float64, order='F')
        
        # Vectorized min and max
        min_val = np.min(X, axis=axis, keepdims=True)
        max_val = np.max(X, axis=axis, keepdims=True)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-10, 1.0, range_val)
        
        # Vectorized normalization
        X_norm = (X - min_val) / range_val
        
        return X_norm
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Fast matrix multiplication using BLAS (Fortran-like)
        
        Uses optimized BLAS routines similar to Fortran's
        matrix operations.
        
        Parameters
        ----------
        A, B : array-like
            Matrices to multiply
            
        Returns
        -------
        C : array-like
            Matrix product A @ B
        """
        A = np.asarray(A, dtype=np.float64, order='F')
        B = np.asarray(B, dtype=np.float64, order='F')
        
        if self.use_blas:
            # Use BLAS gemm (General Matrix Multiply) - Fortran's strength
            try:
                # NumPy's matmul uses BLAS internally when available
                return np.dot(A, B)
            except:
                return A @ B
        else:
            return A @ B
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system Ax = b using LAPACK (Fortran-like)
        
        Parameters
        ----------
        A : array-like
            Coefficient matrix
        b : array-like
            Right-hand side vector/matrix
            
        Returns
        -------
        x : array-like
            Solution vector/matrix
        """
        A = np.asarray(A, dtype=np.float64, order='F')
        b = np.asarray(b, dtype=np.float64, order='F')
        
        if self.use_blas:
            # Use LAPACK solve (Fortran's strength)
            try:
                return np.linalg.solve(A, b)
            except:
                return np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            return np.linalg.solve(A, b)
    
    def eigen_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition using LAPACK (Fortran-like)
        
        Parameters
        ----------
        A : array-like
            Square matrix
            
        Returns
        -------
        eigenvalues : array-like
            Eigenvalues
        eigenvectors : array-like
            Eigenvectors
        """
        A = np.asarray(A, dtype=np.float64, order='F')
        
        # NumPy uses LAPACK internally
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        return eigenvalues, eigenvectors
    
    def svd(self, A: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Singular Value Decomposition using LAPACK (Fortran-like)
        
        Parameters
        ----------
        A : array-like
            Input matrix
        full_matrices : bool, default=False
            Return full matrices
            
        Returns
        -------
        U, s, Vt : tuple
            SVD components
        """
        A = np.asarray(A, dtype=np.float64, order='F')
        
        # NumPy uses LAPACK internally
        U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
        
        return U, s, Vt
    
    def pairwise_distances(self, X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """
        Fast pairwise distances (Fortran-like vectorized)
        
        Uses optimized vectorized operations for distance computation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        metric : str, default='euclidean'
            Distance metric
            
        Returns
        -------
        distances : array-like, shape (n_samples, n_samples)
            Pairwise distance matrix
        """
        X = np.asarray(X, dtype=np.float64, order='F')
        n_samples = X.shape[0]
        
        if metric == 'euclidean':
            # Vectorized Euclidean distance (Fortran-like efficiency)
            # Using broadcasting for efficiency
            X_squared = np.sum(X ** 2, axis=1, keepdims=True)
            distances_squared = X_squared + X_squared.T - 2 * X @ X.T
            distances_squared = np.maximum(distances_squared, 0)  # Avoid negative due to numerical errors
            distances = np.sqrt(distances_squared)
            return distances
        else:
            # Fallback to scikit-learn if available
            try:
                from sklearn.metrics.pairwise import pairwise_distances
                return pairwise_distances(X, metric=metric)
            except ImportError:
                raise ValueError(f"Metric '{metric}' not supported without scikit-learn")
    
    def batch_process(self, X: np.ndarray, func, batch_size: int = 1000) -> np.ndarray:
        """
        Process data in batches (Fortran-like memory efficiency)
        
        Parameters
        ----------
        X : array-like
            Input data
        func : callable
            Function to apply to each batch
        batch_size : int, default=1000
            Batch size
            
        Returns
        -------
        results : array-like
            Processed data
        """
        X = np.asarray(X, dtype=np.float64, order='F')
        n_samples = X.shape[0]
        
        results = []
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            result = func(batch)
            results.append(result)
        
        return np.concatenate(results, axis=0)
    
    def vectorized_operation(self, X: np.ndarray, operation: str, **kwargs) -> np.ndarray:
        """
        Apply vectorized operation (Fortran-like array processing)
        
        Parameters
        ----------
        X : array-like
            Input data
        operation : str
            Operation to apply ('sum', 'mean', 'std', 'max', 'min', etc.)
        **kwargs
            Additional arguments for operation
            
        Returns
        -------
        result : array-like
            Result of operation
        """
        X = np.asarray(X, dtype=np.float64, order='F')
        
        if operation == 'sum':
            return np.sum(X, **kwargs)
        elif operation == 'mean':
            return np.mean(X, **kwargs)
        elif operation == 'std':
            return np.std(X, **kwargs)
        elif operation == 'max':
            return np.max(X, **kwargs)
        elif operation == 'min':
            return np.min(X, **kwargs)
        elif operation == 'var':
            return np.var(X, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")


# Numba-accelerated functions for even better performance
@jit(nopython=True, parallel=True, cache=True)
def _fast_standardize_numba(X):
    """Numba-accelerated standardization (Fortran-like performance)"""
    n_samples, n_features = X.shape
    X_std = np.empty_like(X)
    
    for j in prange(n_features):
        # Compute mean
        mean = 0.0
        for i in range(n_samples):
            mean += X[i, j]
        mean /= n_samples
        
        # Compute std
        variance = 0.0
        for i in range(n_samples):
            diff = X[i, j] - mean
            variance += diff * diff
        std = np.sqrt(variance / n_samples)
        
        # Standardize
        if std > 1e-10:
            for i in range(n_samples):
                X_std[i, j] = (X[i, j] - mean) / std
        else:
            for i in range(n_samples):
                X_std[i, j] = 0.0
    
    return X_std


@jit(nopython=True, parallel=True, cache=True)
def _fast_pairwise_distances_numba(X):
    """Numba-accelerated pairwise distances (Fortran-like performance)"""
    n_samples, n_features = X.shape
    distances = np.zeros((n_samples, n_samples))
    
    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            dist_sq = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances
