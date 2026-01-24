"""
Test Computational Kernels (Fortran/Julia-like Performance)
"""
import sys
from pathlib import Path
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("COMPUTATIONAL KERNELS TEST")
print("="*80)

try:
    from ml_toolbox.computational_kernels import (
        FortranLikeKernel,
        JuliaLikeKernel,
        UnifiedComputationalKernel
    )
    print("\n[OK] Computational kernels imported successfully")
except ImportError as e:
    print(f"\n[ERROR] Could not import computational kernels: {e}")
    sys.exit(1)

# Generate test data
np.random.seed(42)
X = np.random.randn(1000, 20).astype(np.float64)
A = np.random.randn(100, 100).astype(np.float64)
B = np.random.randn(100, 100).astype(np.float64)

print(f"\nTest data: X shape={X.shape}, A shape={A.shape}, B shape={B.shape}")

# Test 1: Fortran-like Kernel
print("\n" + "="*80)
print("1. FORTRAN-LIKE KERNEL TEST")
print("="*80)

try:
    fortran_kernel = FortranLikeKernel(use_blas=True, parallel=True)
    
    # Standardization
    start = time.time()
    X_std = fortran_kernel.standardize(X)
    time_std = time.time() - start
    print(f"Standardization: {time_std:.4f}s")
    
    # Normalization
    start = time.time()
    X_norm = fortran_kernel.normalize(X)
    time_norm = time.time() - start
    print(f"Normalization: {time_norm:.4f}s")
    
    # Matrix multiplication
    start = time.time()
    C = fortran_kernel.matrix_multiply(A, B)
    time_matmul = time.time() - start
    print(f"Matrix multiplication: {time_matmul:.4f}s")
    
    # Pairwise distances
    start = time.time()
    distances = fortran_kernel.pairwise_distances(X[:100])  # Smaller for speed
    time_dist = time.time() - start
    print(f"Pairwise distances (100 samples): {time_dist:.4f}s")
    
    print("\n[OK] Fortran-like kernel working")
except Exception as e:
    print(f"\n[ERROR] Fortran-like kernel failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Julia-like Kernel
print("\n" + "="*80)
print("2. JULIA-LIKE KERNEL TEST")
print("="*80)

try:
    julia_kernel = JuliaLikeKernel(jit_enabled=True, cache_enabled=True)
    
    # Warmup
    print("Warming up Julia-like kernel...")
    julia_kernel.warmup()
    
    # Standardization
    start = time.time()
    X_std = julia_kernel.standardize(X)
    time_std = time.time() - start
    print(f"Standardization: {time_std:.4f}s")
    
    # Normalization
    start = time.time()
    X_norm = julia_kernel.normalize(X)
    time_norm = time.time() - start
    print(f"Normalization: {time_norm:.4f}s")
    
    # Matrix multiplication
    start = time.time()
    C = julia_kernel.matrix_multiply(A, B)
    time_matmul = time.time() - start
    print(f"Matrix multiplication: {time_matmul:.4f}s")
    
    # Pairwise distances
    start = time.time()
    distances = julia_kernel.pairwise_distances(X[:100])
    time_dist = time.time() - start
    print(f"Pairwise distances (100 samples): {time_dist:.4f}s")
    
    print("\n[OK] Julia-like kernel working")
except Exception as e:
    print(f"\n[ERROR] Julia-like kernel failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Unified Kernel
print("\n" + "="*80)
print("3. UNIFIED COMPUTATIONAL KERNEL TEST")
print("="*80)

try:
    unified_kernel = UnifiedComputationalKernel(
        mode='auto',
        use_blas=True,
        jit_enabled=True,
        parallel=True
    )
    
    # Get performance info
    info = unified_kernel.get_performance_info()
    print(f"Performance info: {info}")
    
    # Standardization
    start = time.time()
    X_std = unified_kernel.standardize(X)
    time_std = time.time() - start
    print(f"Standardization (auto): {time_std:.4f}s")
    
    # Normalization
    start = time.time()
    X_norm = unified_kernel.normalize(X)
    time_norm = time.time() - start
    print(f"Normalization (auto): {time_norm:.4f}s")
    
    # Matrix multiplication
    start = time.time()
    C = unified_kernel.matrix_multiply(A, B)
    time_matmul = time.time() - start
    print(f"Matrix multiplication (auto): {time_matmul:.4f}s")
    
    # Pairwise distances
    start = time.time()
    distances = unified_kernel.pairwise_distances(X[:100])
    time_dist = time.time() - start
    print(f"Pairwise distances (auto, 100 samples): {time_dist:.4f}s")
    
    print("\n[OK] Unified computational kernel working")
except Exception as e:
    print(f"\n[ERROR] Unified kernel failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Comparison with NumPy
print("\n" + "="*80)
print("4. COMPARISON WITH NUMPY")
print("="*80)

try:
    # NumPy standardization
    start = time.time()
    X_std_numpy = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    time_numpy = time.time() - start
    print(f"NumPy standardization: {time_numpy:.4f}s")
    
    # Unified kernel standardization
    start = time.time()
    X_std_kernel = unified_kernel.standardize(X)
    time_kernel = time.time() - start
    print(f"Unified kernel standardization: {time_kernel:.4f}s")
    
    if time_numpy > 0 and time_kernel > 0:
        speedup = time_numpy / time_kernel
        print(f"Speedup: {speedup:.2f}x")
    elif time_kernel == 0:
        print("Speedup: Very fast (kernel time < 0.0001s)")
    
    # Verify correctness
    diff = np.abs(X_std_numpy - X_std_kernel).max()
    print(f"Max difference: {diff:.2e}")
    if diff < 1e-10:
        print("[OK] Results match NumPy")
    else:
        print("[WARNING] Results differ from NumPy")
        
except Exception as e:
    print(f"\n[ERROR] Comparison failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
