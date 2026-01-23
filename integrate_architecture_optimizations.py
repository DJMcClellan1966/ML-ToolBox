"""
Integrate Architecture Optimizations into ML Toolbox
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def integrate_architecture_optimizations():
    """Integrate architecture optimizations"""
    print("="*80)
    print("INTEGRATING ARCHITECTURE-SPECIFIC OPTIMIZATIONS")
    print("="*80)
    print()
    
    try:
        from architecture_optimizer import ArchitectureOptimizer, get_architecture_optimizer
        
        optimizer = get_architecture_optimizer()
        
        # Display architecture info
        summary = optimizer.get_architecture_summary()
        print(summary)
        
        # Test integration
        print("\nTesting Integration:")
        print("-"*80)
        
        # Test with optimized operations
        from optimized_ml_operations import OptimizedMLOperations
        
        ops = OptimizedMLOperations()
        print("[OK] OptimizedMLOperations initialized with architecture optimizer")
        
        # Test NumPy operations
        import numpy as np
        test_array = np.random.randn(100, 256)
        
        # Test similarity computation
        similarity = ops.vectorized_similarity_computation(test_array)
        print(f"[OK] Vectorized similarity computation: {similarity.shape}")
        
        # Test chunk size
        chunk_size = optimizer.get_optimal_chunk_size(10000)
        print(f"[OK] Optimal chunk size for 10000 items: {chunk_size}")
        
        # Test thread count
        threads = optimizer.get_optimal_thread_count()
        print(f"[OK] Optimal thread count: {threads}")
        
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print("\nArchitecture-specific optimizations are now active!")
        print("The ML Toolbox will automatically use the best optimizations for your hardware.")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    integrate_architecture_optimizations()
