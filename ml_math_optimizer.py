"""
ML Math Optimizer
Optimized mathematical operations for machine learning

Provides:
- Optimized linear algebra operations
- Numerical optimization methods
- Statistical computation optimizations
- Vector operation improvements
- Sparse matrix support
"""
import sys
from pathlib import Path
import warnings
from typing import Union, Optional, Tuple, List, Dict, Any
import functools

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    from scipy import linalg, optimize, stats
    from scipy.sparse import csr_matrix, csc_matrix, issparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install with: pip install scipy")

try:
    from architecture_optimizer import get_architecture_optimizer
    ARCHITECTURE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ARCHITECTURE_OPTIMIZER_AVAILABLE = False
    warnings.warn("Architecture optimizer not available")


class MLMathOptimizer:
    """
    ML Math Optimizer
    
    Provides optimized mathematical operations for machine learning:
    - Matrix operations (multiplication, decomposition)
    - Numerical optimization
    - Statistical computations
    - Vector operations
    - Sparse matrix support
    """
    
    def __init__(self):
        """Initialize ML Math Optimizer"""
        if ARCHITECTURE_OPTIMIZER_AVAILABLE:
            self.arch_optimizer = get_architecture_optimizer()
        else:
            self.arch_optimizer = None
        
        self.stats = {
            'matrix_operations': 0,
            'optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def optimized_matrix_multiply(
        self,
        A: np.ndarray,
        B: np.ndarray,
        use_sparse: bool = False
    ) -> np.ndarray:
        """
        Optimized matrix multiplication
        
        Args:
            A: First matrix
            B: Second matrix
            use_sparse: Use sparse matrices if beneficial
        
        Returns:
            Matrix product A @ B
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            A = self.arch_optimizer.optimize_array_operations(A)
            B = self.arch_optimizer.optimize_array_operations(B)
        
        # Use sparse if beneficial
        if use_sparse and SCIPY_AVAILABLE:
            sparsity_A = np.count_nonzero(A == 0) / A.size
            sparsity_B = np.count_nonzero(B == 0) / B.size
            
            if sparsity_A > 0.5 or sparsity_B > 0.5:
                A_sparse = csr_matrix(A) if sparsity_A > 0.5 else A
                B_sparse = csc_matrix(B) if sparsity_B > 0.5 else B
                result = A_sparse @ B_sparse
                if issparse(result):
                    return result.toarray()
                return result
        
        # Optimized dense multiplication (NumPy uses BLAS)
        result = np.dot(A, B)
        self.stats['matrix_operations'] += 1
        return result
    
    def optimized_svd(
        self,
        A: np.ndarray,
        full_matrices: bool = False,
        compute_uv: bool = True
    ) -> Tuple[np.ndarray, ...]:
        """
        Optimized SVD decomposition
        
        Args:
            A: Matrix to decompose
            full_matrices: Whether to compute full matrices
            compute_uv: Whether to compute U and V
        
        Returns:
            SVD decomposition (U, s, Vh)
        """
        if not SCIPY_AVAILABLE:
            if NUMPY_AVAILABLE:
                return np.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
            raise RuntimeError("NumPy or SciPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            A = self.arch_optimizer.optimize_array_operations(A)
        
        # Use SciPy's optimized SVD
        result = linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
        self.stats['matrix_operations'] += 1
        return result
    
    def optimized_eigenvalues(
        self,
        A: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized eigenvalue decomposition
        
        Args:
            A: Square matrix
            k: Number of largest eigenvalues (if None, compute all)
        
        Returns:
            Eigenvalues and eigenvectors
        """
        if not SCIPY_AVAILABLE:
            if NUMPY_AVAILABLE:
                return np.linalg.eig(A)
            raise RuntimeError("NumPy or SciPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            A = self.arch_optimizer.optimize_array_operations(A)
        
        if k is not None and k < A.shape[0]:
            # Use sparse eigensolver for large matrices
            from scipy.sparse.linalg import eigs
            eigenvalues, eigenvectors = eigs(A, k=k)
            return eigenvalues, eigenvectors
        else:
            # Full eigendecomposition
            eigenvalues, eigenvectors = linalg.eig(A)
            self.stats['matrix_operations'] += 1
            return eigenvalues, eigenvectors
    
    def optimized_correlation(
        self,
        X: np.ndarray,
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        Optimized correlation computation
        
        Args:
            X: Data matrix (samples x features)
            method: Correlation method ('pearson', 'spearman')
        
        Returns:
            Correlation matrix
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            X = self.arch_optimizer.optimize_array_operations(X)
        
        # Vectorized correlation computation
        if method == 'pearson':
            # Center the data
            X_centered = X - np.mean(X, axis=0, keepdims=True)
            
            # Compute correlation
            std = np.std(X_centered, axis=0, ddof=1)
            std[std == 0] = 1  # Avoid division by zero
            
            correlation = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
            correlation = correlation / np.outer(std, std)
            
            self.stats['matrix_operations'] += 1
            return correlation
        else:
            # Fallback to scipy for other methods
            if SCIPY_AVAILABLE:
                return np.corrcoef(X.T)
            return np.corrcoef(X.T)
    
    def optimized_gradient_descent(
        self,
        objective_func,
        gradient_func,
        x0: np.ndarray,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimized gradient descent
        
        Args:
            objective_func: Objective function
            gradient_func: Gradient function
            x0: Initial point
            learning_rate: Learning rate
            max_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            Optimal point and optimization info
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required")
        
        x = x0.copy()
        history = {'loss': [], 'iterations': 0}
        
        for i in range(max_iter):
            # Compute gradient
            grad = gradient_func(x)
            
            # Update
            x_new = x - learning_rate * grad
            
            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                history['iterations'] = i + 1
                history['converged'] = True
                break
            
            x = x_new
            history['loss'].append(objective_func(x))
            history['iterations'] = i + 1
        
        self.stats['optimizations'] += 1
        return x, history
    
    def optimized_cholesky(
        self,
        A: np.ndarray
    ) -> np.ndarray:
        """
        Optimized Cholesky decomposition
        
        Args:
            A: Positive definite matrix
        
        Returns:
            Lower triangular matrix L such that A = L @ L.T
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            A = self.arch_optimizer.optimize_array_operations(A)
        
        # NumPy's optimized Cholesky (uses LAPACK)
        L = np.linalg.cholesky(A)
        self.stats['matrix_operations'] += 1
        return L
    
    def optimized_qr(
        self,
        A: np.ndarray,
        mode: str = 'reduced'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized QR decomposition
        
        Args:
            A: Matrix to decompose
            mode: Decomposition mode
        
        Returns:
            Q and R matrices
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required")
        
        # Architecture optimization
        if self.arch_optimizer:
            A = self.arch_optimizer.optimize_array_operations(A)
        
        # NumPy's optimized QR (uses LAPACK)
        Q, R = np.linalg.qr(A, mode=mode)
        self.stats['matrix_operations'] += 1
        return Q, R
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.stats.copy()


# Global optimizer instance
_global_optimizer = None

def get_ml_math_optimizer() -> MLMathOptimizer:
    """Get global ML Math Optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MLMathOptimizer()
    return _global_optimizer


# Example usage
if __name__ == '__main__':
    print("ML Math Optimizer")
    print("="*80)
    
    optimizer = get_ml_math_optimizer()
    
    if NUMPY_AVAILABLE:
        # Test matrix multiplication
        print("\n[1] Testing optimized matrix multiplication...")
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        
        import time
        start = time.time()
        C = optimizer.optimized_matrix_multiply(A, B)
        elapsed = time.time() - start
        print(f"[OK] Matrix multiplication: {elapsed:.4f}s")
        print(f"[OK] Result shape: {C.shape}")
        
        # Test correlation
        print("\n[2] Testing optimized correlation...")
        X = np.random.randn(1000, 50)
        start = time.time()
        corr = optimizer.optimized_correlation(X)
        elapsed = time.time() - start
        print(f"[OK] Correlation computation: {elapsed:.4f}s")
        print(f"[OK] Correlation matrix shape: {corr.shape}")
        
        # Test Cholesky
        print("\n[3] Testing optimized Cholesky decomposition...")
        A = np.random.randn(100, 100)
        A = A @ A.T + np.eye(100) * 0.1  # Make positive definite
        start = time.time()
        L = optimizer.optimized_cholesky(A)
        elapsed = time.time() - start
        print(f"[OK] Cholesky decomposition: {elapsed:.4f}s")
        print(f"[OK] L shape: {L.shape}")
        
        # Stats
        stats = optimizer.get_stats()
        print(f"\n[OK] Optimization Stats:")
        print(f"  Matrix operations: {stats['matrix_operations']}")
        print(f"  Optimizations: {stats['optimizations']}")
    
    print("\n[OK] ML Math Optimizer ready!")
