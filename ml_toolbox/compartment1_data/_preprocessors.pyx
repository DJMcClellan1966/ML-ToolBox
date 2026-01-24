"""
Cython implementation of data preprocessing
High-performance computational core
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs

# Type definitions
ctypedef cnp.float64_t DTYPE_t

def fast_standardize(cnp.ndarray[DTYPE_t, ndim=2] X):
    """
    Fast standardization using Cython
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    
    Returns
    -------
    X_std : array-like, shape (n_samples, n_features)
        Standardized data
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_std = np.empty((n_samples, n_features), dtype=np.float64)
    
    cdef int i, j
    cdef DTYPE_t mean, std, variance, diff, eps = 1e-10
    
    for j in range(n_features):
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
        std = sqrt(variance / n_samples)
        
        # Standardize (avoid division by zero)
        if std > eps:
            for i in range(n_samples):
                X_std[i, j] = (X[i, j] - mean) / std
        else:
            for i in range(n_samples):
                X_std[i, j] = 0.0
    
    return X_std


def fast_normalize(cnp.ndarray[DTYPE_t, ndim=2] X):
    """
    Fast min-max normalization using Cython
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    
    Returns
    -------
    X_norm : array-like, shape (n_samples, n_features)
        Normalized data (0-1 range)
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_norm = np.empty((n_samples, n_features), dtype=np.float64)
    
    cdef int i, j
    cdef DTYPE_t min_val, max_val, range_val, eps = 1e-10
    
    for j in range(n_features):
        # Find min and max
        min_val = X[0, j]
        max_val = X[0, j]
        for i in range(1, n_samples):
            if X[i, j] < min_val:
                min_val = X[i, j]
            if X[i, j] > max_val:
                max_val = X[i, j]
        
        # Normalize
        range_val = max_val - min_val
        if range_val > eps:
            for i in range(n_samples):
                X_norm[i, j] = (X[i, j] - min_val) / range_val
        else:
            for i in range(n_samples):
                X_norm[i, j] = 0.0
    
    return X_norm


def fast_euclidean_distance(cnp.ndarray[DTYPE_t, ndim=1] x1, 
                            cnp.ndarray[DTYPE_t, ndim=1] x2):
    """
    Fast Euclidean distance computation
    
    Parameters
    ----------
    x1, x2 : array-like, shape (n_features,)
        Two feature vectors
    
    Returns
    -------
    distance : float
        Euclidean distance
    """
    cdef int n_features = x1.shape[0]
    cdef DTYPE_t dist_sq = 0.0
    cdef DTYPE_t diff
    cdef int i
    
    for i in range(n_features):
        diff = x1[i] - x2[i]
        dist_sq += diff * diff
    
    return sqrt(dist_sq)


def fast_pairwise_distances(cnp.ndarray[DTYPE_t, ndim=2] X):
    """
    Fast pairwise distance matrix computation
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    
    Returns
    -------
    distances : array-like, shape (n_samples, n_samples)
        Pairwise distance matrix
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    
    cdef int i, j, k
    cdef DTYPE_t diff, dist_sq
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist_sq = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            distances[i, j] = sqrt(dist_sq)
            distances[j, i] = distances[i, j]  # Symmetric
    
    return distances
