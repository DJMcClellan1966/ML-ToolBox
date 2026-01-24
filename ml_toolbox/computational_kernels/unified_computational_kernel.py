"""
Unified Computational Kernel

Combines Fortran-like and Julia-like capabilities into a single
high-performance computational kernel. Similar to quantum-inspired
methods that provide quantum-like benefits without actual quantum hardware,
this provides Fortran/Julia-like performance without those languages.
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal
import logging

from .fortran_like_kernel import FortranLikeKernel
from .julia_like_kernel import JuliaLikeKernel

logger = logging.getLogger(__name__)


class UnifiedComputationalKernel:
    """
    Unified Computational Kernel
    
    Combines the best of Fortran-like (vectorization, BLAS/LAPACK)
    and Julia-like (JIT compilation, type specialization) approaches
    to provide maximum performance without requiring Fortran or Julia.
    
    Similar to quantum-inspired methods: provides quantum-like benefits
    without actual quantum hardware.
    """
    
    def __init__(
        self,
        mode: Literal['auto', 'fortran', 'julia', 'hybrid'] = 'auto',
        use_blas: bool = True,
        jit_enabled: bool = True,
        parallel: bool = True
    ):
        """
        Initialize unified computational kernel
        
        Parameters
        ----------
        mode : {'auto', 'fortran', 'julia', 'hybrid'}, default='auto'
            Computational mode:
            - 'auto': Automatically choose best approach
            - 'fortran': Use Fortran-like vectorization
            - 'julia': Use Julia-like JIT compilation
            - 'hybrid': Combine both approaches
        use_blas : bool, default=True
            Use BLAS/LAPACK for matrix operations
        jit_enabled : bool, default=True
            Enable JIT compilation
        parallel : bool, default=True
            Enable parallel processing
        """
        self.mode = mode
        self.fortran_kernel = FortranLikeKernel(use_blas=use_blas, parallel=parallel)
        self.julia_kernel = JuliaLikeKernel(jit_enabled=jit_enabled, cache_enabled=True)
        
        # Warmup Julia kernel (avoid first-call delay)
        if jit_enabled:
            self.julia_kernel.warmup()
        
        logger.info(f"Unified Computational Kernel initialized (mode: {mode})")
    
    def standardize(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fast standardization using best available method
        
        Parameters
        ----------
        X : array-like
            Input data
        **kwargs
            Additional arguments
            
        Returns
        -------
        X_std : array-like
            Standardized data
        """
        if self.mode == 'fortran':
            return self.fortran_kernel.standardize(X, **kwargs)
        elif self.mode == 'julia':
            return self.julia_kernel.standardize(X)
        elif self.mode == 'hybrid':
            # Try both and use faster (for large arrays, Fortran-like is often faster)
            if X.size > 100000:
                return self.fortran_kernel.standardize(X, **kwargs)
            else:
                return self.julia_kernel.standardize(X)
        else:  # auto
            # Auto-select based on data size
            if X.size > 100000:
                return self.fortran_kernel.standardize(X, **kwargs)
            else:
                return self.julia_kernel.standardize(X)
    
    def normalize(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fast normalization using best available method
        
        Parameters
        ----------
        X : array-like
            Input data
        **kwargs
            Additional arguments
            
        Returns
        -------
        X_norm : array-like
            Normalized data
        """
        if self.mode == 'fortran':
            return self.fortran_kernel.normalize(X, **kwargs)
        elif self.mode == 'julia':
            return self.julia_kernel.normalize(X)
        elif self.mode == 'hybrid':
            if X.size > 100000:
                return self.fortran_kernel.normalize(X, **kwargs)
            else:
                return self.julia_kernel.normalize(X)
        else:  # auto
            if X.size > 100000:
                return self.fortran_kernel.normalize(X, **kwargs)
            else:
                return self.julia_kernel.normalize(X)
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Fast matrix multiplication using best available method
        
        Parameters
        ----------
        A, B : array-like
            Matrices to multiply
            
        Returns
        -------
        C : array-like
            Matrix product
        """
        if self.mode == 'fortran':
            return self.fortran_kernel.matrix_multiply(A, B)
        elif self.mode == 'julia':
            return self.julia_kernel.matrix_multiply(A, B)
        elif self.mode == 'hybrid':
            # For large matrices, use BLAS (Fortran-like)
            if A.size > 10000 or B.size > 10000:
                return self.fortran_kernel.matrix_multiply(A, B)
            else:
                return self.julia_kernel.matrix_multiply(A, B)
        else:  # auto
            if A.size > 10000 or B.size > 10000:
                return self.fortran_kernel.matrix_multiply(A, B)
            else:
                return self.julia_kernel.matrix_multiply(A, B)
    
    def pairwise_distances(self, X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """
        Fast pairwise distances using best available method
        
        Parameters
        ----------
        X : array-like
            Input data
        metric : str, default='euclidean'
            Distance metric
            
        Returns
        -------
        distances : array-like
            Pairwise distance matrix
        """
        if self.mode == 'fortran':
            return self.fortran_kernel.pairwise_distances(X, metric=metric)
        elif self.mode == 'julia':
            return self.julia_kernel.pairwise_distances(X, metric=metric)
        elif self.mode == 'hybrid':
            # For large datasets, use vectorized (Fortran-like)
            if X.shape[0] > 1000:
                return self.fortran_kernel.pairwise_distances(X, metric=metric)
            else:
                return self.julia_kernel.pairwise_distances(X, metric=metric)
        else:  # auto
            if X.shape[0] > 1000:
                return self.fortran_kernel.pairwise_distances(X, metric=metric)
            else:
                return self.julia_kernel.pairwise_distances(X, metric=metric)
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system using LAPACK (Fortran-like)
        
        Parameters
        ----------
        A : array-like
            Coefficient matrix
        b : array-like
            Right-hand side
            
        Returns
        -------
        x : array-like
            Solution
        """
        return self.fortran_kernel.solve_linear_system(A, b)
    
    def eigen_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition using LAPACK (Fortran-like)
        
        Parameters
        ----------
        A : array-like
            Square matrix
            
        Returns
        -------
        eigenvalues, eigenvectors : tuple
            Eigen decomposition
        """
        return self.fortran_kernel.eigen_decomposition(A)
    
    def svd(self, A: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD using LAPACK (Fortran-like)
        
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
        return self.fortran_kernel.svd(A, full_matrices=full_matrices)
    
    def get_performance_info(self) -> dict:
        """
        Get performance information about the kernel
        
        Returns
        -------
        info : dict
            Performance information
        """
        return {
            'mode': self.mode,
            'fortran_blas': self.fortran_kernel.use_blas,
            'julia_jit': self.julia_kernel.jit_enabled,
            'julia_warmup': self.julia_kernel._warmup_done,
            'parallel': self.fortran_kernel.parallel,
        }
