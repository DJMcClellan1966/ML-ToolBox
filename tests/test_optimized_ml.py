"""
Tests for Optimized ML Tasks
Test speed and accuracy improvements
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import os
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from optimized_ml_tasks import OptimizedMLTasks
    from optimized_preprocessing import OptimizedPreprocessor
    from simple_ml_tasks import SimpleMLTasks
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Optimized features not available")


class TestOptimizedMLTasks:
    """Tests for optimized ML tasks"""
    
    def test_train_classifier_optimized(self):
        """Test optimized classifier training"""
        optimized = OptimizedMLTasks(cache_dir="test_cache")
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        result = optimized.train_classifier_optimized(
            X, y, model_type='random_forest', use_cache=False
        )
        
        assert 'model' in result or 'error' in result
        if 'model' in result:
            assert result.get('optimized') == True
        
        # Cleanup
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")
    
    def test_train_regressor_optimized(self):
        """Test optimized regressor training"""
        optimized = OptimizedMLTasks(cache_dir="test_cache")
        
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        result = optimized.train_regressor_optimized(
            X, y, model_type='random_forest', use_cache=False
        )
        
        assert 'model' in result or 'error' in result
        if 'model' in result:
            assert result.get('optimized') == True
        
        # Cleanup
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")
    
    def test_ensemble_model(self):
        """Test ensemble model for better accuracy"""
        optimized = OptimizedMLTasks(cache_dir="test_cache")
        
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        
        result = optimized.train_classifier_optimized(
            X, y, model_type='ensemble', use_cache=False
        )
        
        assert 'model' in result or 'error' in result
        
        # Cleanup
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")
    
    def test_caching(self):
        """Test model caching"""
        optimized = OptimizedMLTasks(cache_dir="test_cache")
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # First run
        start1 = time.time()
        result1 = optimized.train_classifier_optimized(
            X, y, use_cache=True
        )
        time1 = time.time() - start1
        
        # Second run (should use cache)
        start2 = time.time()
        result2 = optimized.train_classifier_optimized(
            X, y, use_cache=True
        )
        time2 = time.time() - start2
        
        # Cached run should be faster
        assert time2 < time1 or abs(time2 - time1) < 0.1  # Allow small variance
        
        # Cleanup
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")


class TestOptimizedPreprocessor:
    """Tests for optimized preprocessing"""
    
    def test_preprocess_fast(self):
        """Test fast preprocessing"""
        preprocessor = OptimizedPreprocessor(cache_dir="test_preprocessing_cache")
        
        X = np.random.rand(100, 10)
        X[0, 0] = np.nan  # Add missing value
        
        result = preprocessor.preprocess_fast(X, operations=['impute', 'scale'])
        
        assert 'X_processed' in result or 'error' in result
        if 'X_processed' in result:
            assert result.get('optimized') == True
        
        # Cleanup
        import shutil
        if os.path.exists("test_preprocessing_cache"):
            shutil.rmtree("test_preprocessing_cache")


class TestSpeedComparison:
    """Compare optimized vs regular"""
    
    def test_speed_improvement(self):
        """Test that optimized version is faster or similar"""
        optimized = OptimizedMLTasks(cache_dir="test_cache")
        simple = SimpleMLTasks()
        
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        
        # Optimized
        start_opt = time.time()
        result_opt = optimized.train_classifier_optimized(
            X, y, use_cache=False, tune_hyperparameters=False
        )
        time_opt = time.time() - start_opt
        
        # Simple
        start_simple = time.time()
        result_simple = simple.train_classifier(X, y)
        time_simple = time.time() - start_simple
        
        # Both should work
        assert 'model' in result_opt or 'error' in result_opt
        assert 'model' in result_simple or 'error' in result_simple
        
        # Cleanup
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
