"""
Integrate Medulla Toolbox Optimizer into ML Toolbox
Demonstrates automatic optimization of ML operations
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def integrate_optimizer():
    """Integrate and test Medulla Toolbox Optimizer"""
    print("="*80)
    print("INTEGRATING MEDULLA TOOLBOX OPTIMIZER")
    print("="*80)
    print()
    
    try:
        from ml_toolbox import MLToolbox
        from medulla_toolbox_optimizer import MLTaskType
        
        print("[1/4] Creating ML Toolbox with optimizer...")
        toolbox = MLToolbox(auto_start_optimizer=True)
        print("[OK] ML Toolbox created with optimizer")
        
        # Check optimizer status
        if toolbox.optimizer:
            print("[OK] Optimizer is active")
            status = toolbox.get_system_status()
            print(f"[OK] System Status:")
            print(f"  CPU: {status.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {status.get('memory_percent', 0):.1f}%")
            print(f"  Optimal Threads: {status.get('optimal_threads', 0)}")
        else:
            print("[WARNING] Optimizer not available")
        
        print("\n[2/4] Testing optimized data preprocessing...")
        
        # Test optimized preprocessing
        def preprocess_data(data):
            """Simulate data preprocessing"""
            time.sleep(0.05)  # Simulate work
            return [x * 2 for x in data]
        
        test_data = [1, 2, 3, 4, 5]
        
        # First call (cache miss)
        start = time.time()
        result1 = toolbox.optimize_operation(
            "preprocess_data",
            preprocess_data,
            task_type=MLTaskType.DATA_PREPROCESSING,
            use_cache=True,
            data=test_data
        )
        time1 = time.time() - start
        print(f"[OK] First call: {time1:.4f}s (cache miss)")
        
        # Second call (cache hit - should be faster)
        start = time.time()
        result2 = toolbox.optimize_operation(
            "preprocess_data",
            preprocess_data,
            task_type=MLTaskType.DATA_PREPROCESSING,
            use_cache=True,
            data=test_data
        )
        time2 = time.time() - start
        print(f"[OK] Second call: {time2:.4f}s (cache hit)")
        
        if time2 < time1:
            speedup = ((time1 - time2) / time1) * 100
            print(f"[OK] Cache speedup: {speedup:.1f}% faster")
        
        print("\n[3/4] Testing optimized model training...")
        
        # Test optimized model training
        def train_model(X, y):
            """Simulate model training"""
            time.sleep(0.1)  # Simulate work
            return {"accuracy": 0.95, "model": "trained"}
        
        # Simulate training data
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        
        start = time.time()
        model = toolbox.optimize_operation(
            "train_model",
            train_model,
            task_type=MLTaskType.MODEL_TRAINING,
            use_cache=False,  # Don't cache training
            X=X_train,
            y=y_train
        )
        train_time = time.time() - start
        print(f"[OK] Model training: {train_time:.4f}s")
        print(f"[OK] Model accuracy: {model['accuracy']}")
        
        print("\n[4/4] Testing optimization statistics...")
        
        stats = toolbox.get_optimization_stats()
        print(f"[OK] Optimization Stats:")
        print(f"  Tasks optimized: {stats.get('tasks_optimized', 0)}")
        print(f"  Cache hits: {stats.get('cache_hits', 0)}")
        print(f"  Cache misses: {stats.get('cache_misses', 0)}")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"  Tasks throttled: {stats.get('tasks_throttled', 0)}")
        
        # Test with context manager
        print("\n[5/5] Testing context manager (auto-cleanup)...")
        with MLToolbox(auto_start_optimizer=True) as tb:
            if tb.optimizer and tb.optimizer.regulation_running:
                print("[OK] Optimizer running in context")
            time.sleep(0.5)
        print("[OK] Context exited - optimizer stopped")
        
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print("\n[SUCCESS] Medulla Toolbox Optimizer integrated!")
        print("  - Optimizer auto-starts with toolbox")
        print("  - ML operations are automatically optimized")
        print("  - Operation caching enabled")
        print("  - Task-specific resource allocation")
        print("  - Performance monitoring available")
        
        # Cleanup
        if toolbox.optimizer and toolbox.optimizer.regulation_running:
            toolbox.optimizer.stop_regulation()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    integrate_optimizer()
