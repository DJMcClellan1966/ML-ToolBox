"""
Test Medulla Toolbox Optimizer Performance Impact
Compares ML Toolbox performance with and without optimizer
"""
import sys
from pathlib import Path
import time
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

def test_optimizer_performance():
    """Test optimizer performance impact"""
    print("="*80)
    print("MEDULLA TOOLBOX OPTIMIZER PERFORMANCE TEST")
    print("="*80)
    print()
    
    try:
        from ml_toolbox import MLToolbox
        from medulla_toolbox_optimizer import MLTaskType
        
        # Test WITHOUT optimizer
        print("[TEST 1] WITHOUT Optimizer")
        print("-"*80)
        
        toolbox_without = MLToolbox(auto_start_optimizer=False)
        
        def preprocess_operation(data):
            time.sleep(0.01)
            return [x * 2 for x in data]
        
        test_data = list(range(100))
        
        # Run multiple times
        times_without = []
        for i in range(5):
            start = time.time()
            result = preprocess_operation(test_data)
            elapsed = time.time() - start
            times_without.append(elapsed)
        
        avg_without = sum(times_without) / len(times_without)
        print(f"  Average time: {avg_without:.4f}s")
        print(f"  Total time: {sum(times_without):.4f}s")
        
        # Test WITH optimizer
        print("\n[TEST 2] WITH Optimizer")
        print("-"*80)
        
        toolbox_with = MLToolbox(auto_start_optimizer=True)
        
        times_with = []
        cache_hits = 0
        
        for i in range(5):
            start = time.time()
            result = toolbox_with.optimize_operation(
                "preprocess_operation",
                preprocess_operation,
                task_type=MLTaskType.DATA_PREPROCESSING,
                use_cache=True,
                data=test_data
            )
            elapsed = time.time() - start
            times_with.append(elapsed)
            
            # Check if cached (second call should be faster)
            if i > 0 and elapsed < avg_without * 0.5:
                cache_hits += 1
        
        avg_with = sum(times_with) / len(times_with)
        print(f"  Average time: {avg_with:.4f}s")
        print(f"  Total time: {sum(times_with):.4f}s")
        print(f"  Cache hits: {cache_hits}")
        
        # Get stats
        stats = toolbox_with.get_optimization_stats()
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        
        # Comparison
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        improvement = ((avg_without - avg_with) / avg_without) * 100
        print(f"\nAverage Time:")
        print(f"  Without: {avg_without:.4f}s")
        print(f"  With: {avg_with:.4f}s")
        print(f"  Improvement: {improvement:+.1f}%")
        
        if cache_hits > 0:
            print(f"\n[SUCCESS] Caching is working!")
            print(f"  Cache hits: {cache_hits}")
            print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        
        # Cleanup
        if toolbox_with.optimizer and toolbox_with.optimizer.regulation_running:
            toolbox_with.optimizer.stop_regulation()
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_optimizer_performance()
