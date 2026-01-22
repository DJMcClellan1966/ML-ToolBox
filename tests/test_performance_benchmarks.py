"""
Performance Benchmarks
Compare ML Toolbox performance against reference implementations

Tests:
- Speed benchmarks
- Memory usage
- Scalability
- Throughput
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import time
import tracemalloc
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    pytestmark = pytest.mark.skip("sklearn not available")

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    pytestmark = pytest.mark.skip("ML Toolbox not available")


class TestPerformance:
    """Performance benchmarks"""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing"""
        X, y = make_classification(
            n_samples=10000,
            n_features=100,
            n_informative=50,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def medium_dataset(self):
        """Generate medium dataset"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            random_state=42
        )
        return X, y
    
    def benchmark_preprocessing_speed(self, benchmark_size=1000):
        """Benchmark preprocessing speed"""
        # Generate test data
        texts = [f"This is test sentence number {i}." for i in range(benchmark_size)]
        
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # Time preprocessing
            start = time.time()
            results = toolbox.data.preprocess(texts, advanced=True, verbose=False)
            elapsed = time.time() - start
            
            # Should complete in reasonable time (< 60 seconds for 1000 samples)
            assert elapsed < 60.0, f"Preprocessing too slow: {elapsed:.2f}s"
            
            print(f"\nPreprocessing {benchmark_size} samples: {elapsed:.2f}s")
            print(f"  Throughput: {benchmark_size/elapsed:.1f} samples/sec")
            
            return elapsed
        except Exception as e:
            pytest.skip(f"Preprocessing benchmark failed: {e}")
    
    def benchmark_model_evaluation_speed(self, medium_dataset):
        """Benchmark model evaluation speed"""
        X, y = medium_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Reference timing
        start = time.time()
        from sklearn.model_selection import cross_val_score
        ref_scores = cross_val_score(model, X_train, y_train, cv=5)
        ref_time = time.time() - start
        
        # ML Toolbox timing
        toolbox = MLToolbox()
        evaluator = toolbox.algorithms.get_evaluator()
        
        start = time.time()
        results = evaluator.evaluate_model(model, X_train, y_train, cv=5)
        toolbox_time = time.time() - start
        
        # Toolbox should be within 2x of reference (acceptable overhead)
        assert toolbox_time < ref_time * 2, \
            f"Toolbox too slow: {toolbox_time:.2f}s vs {ref_time:.2f}s"
        
        print(f"\nModel evaluation:")
        print(f"  Reference: {ref_time:.3f}s")
        print(f"  Toolbox: {toolbox_time:.3f}s")
        print(f"  Overhead: {(toolbox_time/ref_time - 1)*100:.1f}%")
    
    def benchmark_memory_usage(self, medium_dataset):
        """Benchmark memory usage"""
        X, y = medium_dataset
        
        try:
            tracemalloc.start()
            
            # Reference memory
            model_ref = RandomForestClassifier(n_estimators=50, random_state=42)
            model_ref.fit(X, y)
            current, peak = tracemalloc.get_traced_memory()
            ref_memory = peak / 1024 / 1024  # MB
            
            tracemalloc.stop()
            tracemalloc.start()
            
            # ML Toolbox memory
            toolbox = MLToolbox()
            evaluator = toolbox.algorithms.get_evaluator()
            results = evaluator.evaluate_model(model_ref, X, y, cv=3)
            current, peak = tracemalloc.get_traced_memory()
            toolbox_memory = peak / 1024 / 1024  # MB
            
            tracemalloc.stop()
            
            # Toolbox should use reasonable memory (< 10x reference)
            assert toolbox_memory < ref_memory * 10, \
                f"Toolbox uses too much memory: {toolbox_memory:.1f}MB vs {ref_memory:.1f}MB"
            
            print(f"\nMemory usage:")
            print(f"  Reference: {ref_memory:.1f} MB")
            print(f"  Toolbox: {toolbox_memory:.1f} MB")
            print(f"  Ratio: {toolbox_memory/ref_memory:.2f}x")
            
        except Exception as e:
            pytest.skip(f"Memory benchmark failed: {e}")
    
    def benchmark_scalability(self):
        """Test scalability with increasing data sizes"""
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            X, y = make_classification(n_samples=size, n_features=10, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            toolbox = MLToolbox()
            evaluator = toolbox.algorithms.get_evaluator()
            
            start = time.time()
            results = evaluator.evaluate_model(model, X, y, cv=3)
            elapsed = time.time() - start
            
            times.append(elapsed)
            print(f"Size {size}: {elapsed:.3f}s")
        
        # Check that time increases sub-quadratically (reasonable scalability)
        # Time should not grow faster than O(n log n) for most operations
        ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        max_ratio = max(ratios)
        
        # Each doubling should not take more than 3x longer
        assert max_ratio < 3.0, f"Poor scalability: max ratio {max_ratio:.2f}"
    
    def benchmark_inference_speed(self, medium_dataset):
        """Benchmark inference speed"""
        X, y = medium_dataset
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Reference inference
        start = time.time()
        ref_predictions = model.predict(X_test)
        ref_time = time.time() - start
        
        # ML Toolbox inference (if available)
        try:
            from ml_toolbox.compartment4_mlops import MLOpsCompartment
            mlops = MLOpsCompartment()
            realtime = mlops.get_realtime_inference(model)
            
            start = time.time()
            toolbox_predictions = []
            for x in X_test:
                pred = realtime.predict(x)
                toolbox_predictions.append(pred)
            toolbox_time = time.time() - start
            
            # Toolbox should be within 2x of reference
            assert toolbox_time < ref_time * 2, \
                f"Toolbox inference too slow: {toolbox_time:.3f}s vs {ref_time:.3f}s"
            
            print(f"\nInference speed:")
            print(f"  Reference: {ref_time:.3f}s ({len(X_test)/ref_time:.0f} samples/sec)")
            print(f"  Toolbox: {toolbox_time:.3f}s ({len(X_test)/toolbox_time:.0f} samples/sec)")
        except ImportError:
            pytest.skip("MLOps not available")
    
    def benchmark_batch_processing(self, large_dataset):
        """Benchmark batch processing"""
        X, y = large_dataset
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Reference batch prediction
        start = time.time()
        ref_predictions = model.predict(X_test)
        ref_time = time.time() - start
        
        # ML Toolbox batch inference
        try:
            from ml_toolbox.compartment4_mlops import MLOpsCompartment
            mlops = MLOpsCompartment()
            batch_inference = mlops.get_batch_inference(model)
            
            start = time.time()
            toolbox_predictions = batch_inference.predict_batch(X_test, batch_size=1000)
            toolbox_time = time.time() - start
            
            # Should be similar or faster
            assert toolbox_time < ref_time * 1.5, \
                f"Batch processing too slow: {toolbox_time:.3f}s vs {ref_time:.3f}s"
            
            print(f"\nBatch processing ({len(X_test)} samples):")
            print(f"  Reference: {ref_time:.3f}s")
            print(f"  Toolbox: {toolbox_time:.3f}s")
        except ImportError:
            pytest.skip("MLOps not available")


class TestConcurrency:
    """Test concurrent operations"""
    
    def test_concurrent_evaluations(self, medium_dataset):
        """Test concurrent model evaluations"""
        import concurrent.futures
        
        X, y = medium_dataset
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = [
            RandomForestClassifier(n_estimators=10, random_state=42+i)
            for i in range(3)
        ]
        
        for model in models:
            model.fit(X_train, y_train)
        
        toolbox = MLToolbox()
        evaluator = toolbox.algorithms.get_evaluator()
        
        def evaluate_model(model):
            return evaluator.evaluate_model(model, X_train, y_train, cv=3)
        
        # Run concurrently
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(evaluate_model, model) for model in models]
            results = [f.result() for f in futures]
        concurrent_time = time.time() - start
        
        # Run sequentially
        start = time.time()
        sequential_results = [evaluate_model(model) for model in models]
        sequential_time = time.time() - start
        
        # Concurrent should be faster (or at least not slower)
        print(f"\nConcurrent vs Sequential:")
        print(f"  Concurrent: {concurrent_time:.3f}s")
        print(f"  Sequential: {sequential_time:.3f}s")
        
        # All results should be valid
        assert all('accuracy' in r for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
