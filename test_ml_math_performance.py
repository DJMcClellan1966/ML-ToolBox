"""
Test ML Math Optimizer Performance
Compare ML Math Optimizer vs Standard NumPy operations
"""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

def test_ml_math_performance():
    """Test ML Math Optimizer performance"""
    print("="*80)
    print("ML MATH OPTIMIZER PERFORMANCE TEST")
    print("="*80)
    print()
    
    try:
        from ml_math_optimizer import get_ml_math_optimizer
        
        optimizer = get_ml_math_optimizer()
        results = []
        
        # Test 1: Matrix Multiplication
        print("[1/6] Testing Matrix Multiplication...")
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for size in sizes:
            A = np.random.randn(size[0], size[1])
            B = np.random.randn(size[1], size[0])
            
            # Standard
            start = time.time()
            C_standard = np.dot(A, B)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            C_optimized = optimizer.optimized_matrix_multiply(A, B)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'Matrix Multiplication',
                'size': f"{size[0]}x{size[1]}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size[0]}x{size[1]}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Test 2: Correlation
        print("\n[2/6] Testing Correlation Computation...")
        sizes = [(100, 10), (1000, 50), (5000, 100)]
        
        for size in sizes:
            X = np.random.randn(size[0], size[1])
            
            # Standard
            start = time.time()
            corr_standard = np.corrcoef(X.T)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            corr_optimized = optimizer.optimized_correlation(X)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'Correlation',
                'size': f"{size[0]}x{size[1]}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size[0]}x{size[1]}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Test 3: Cholesky Decomposition
        print("\n[3/6] Testing Cholesky Decomposition...")
        sizes = [100, 500, 1000]
        
        for size in sizes:
            A = np.random.randn(size, size)
            A = A @ A.T + np.eye(size) * 0.1  # Make positive definite
            
            # Standard
            start = time.time()
            L_standard = np.linalg.cholesky(A)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            L_optimized = optimizer.optimized_cholesky(A)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'Cholesky',
                'size': f"{size}x{size}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size}x{size}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Test 4: QR Decomposition
        print("\n[4/6] Testing QR Decomposition...")
        sizes = [(100, 50), (500, 250), (1000, 500)]
        
        for size in sizes:
            A = np.random.randn(size[0], size[1])
            
            # Standard
            start = time.time()
            Q_standard, R_standard = np.linalg.qr(A)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            Q_optimized, R_optimized = optimizer.optimized_qr(A)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'QR Decomposition',
                'size': f"{size[0]}x{size[1]}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size[0]}x{size[1]}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Test 5: SVD
        print("\n[5/6] Testing SVD Decomposition...")
        sizes = [(100, 50), (500, 250), (1000, 500)]
        
        for size in sizes:
            A = np.random.randn(size[0], size[1])
            
            # Standard
            start = time.time()
            U_standard, s_standard, Vh_standard = np.linalg.svd(A, full_matrices=False)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            U_optimized, s_optimized, Vh_optimized = optimizer.optimized_svd(A, full_matrices=False)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'SVD',
                'size': f"{size[0]}x{size[1]}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size[0]}x{size[1]}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Test 6: Eigenvalues
        print("\n[6/6] Testing Eigenvalue Decomposition...")
        sizes = [50, 100, 200]
        
        for size in sizes:
            A = np.random.randn(size, size)
            A = (A + A.T) / 2  # Make symmetric
            
            # Standard
            start = time.time()
            eigvals_standard, eigvecs_standard = np.linalg.eig(A)
            time_standard = time.time() - start
            
            # Optimized
            start = time.time()
            eigvals_optimized, eigvecs_optimized = optimizer.optimized_eigenvalues(A)
            time_optimized = time.time() - start
            
            improvement = ((time_standard - time_optimized) / time_standard * 100) if time_standard > 0 else 0
            
            results.append({
                'operation': 'Eigenvalues',
                'size': f"{size}x{size}",
                'standard': time_standard,
                'optimized': time_optimized,
                'improvement': improvement
            })
            
            print(f"  Size {size}x{size}:")
            print(f"    Standard: {time_standard:.4f}s")
            print(f"    Optimized: {time_optimized:.4f}s")
            print(f"    Improvement: {improvement:+.1f}%")
        
        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        avg_improvement = np.mean([r['improvement'] for r in results])
        max_improvement = max([r['improvement'] for r in results])
        min_improvement = min([r['improvement'] for r in results])
        
        print(f"\nAverage Improvement: {avg_improvement:+.1f}%")
        print(f"Best Improvement: {max_improvement:+.1f}%")
        print(f"Worst Improvement: {min_improvement:+.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 80)
        for r in results:
            print(f"{r['operation']:20s} {r['size']:15s} "
                  f"Standard: {r['standard']:7.4f}s  "
                  f"Optimized: {r['optimized']:7.4f}s  "
                  f"Improvement: {r['improvement']:+6.1f}%")
        
        # Stats
        stats = optimizer.get_stats()
        print(f"\nOptimization Statistics:")
        print(f"  Matrix operations: {stats['matrix_operations']}")
        print(f"  Optimizations: {stats['optimizations']}")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_ml_math_performance()
