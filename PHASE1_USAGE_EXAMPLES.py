"""
Phase 1 Integration - Usage Examples
Demonstrates how to use the newly integrated components
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ml_toolbox import MLToolbox


def example_testing():
    """Example: Using testing infrastructure"""
    print("="*60)
    print("Testing Infrastructure Example")
    print("="*60)
    
    toolbox = MLToolbox(check_dependencies=False)
    
    # Get test suite
    test_suite = toolbox.get_test_suite()
    if test_suite:
        print("\n[OK] Test suite available")
        print("Run: test_suite.run_all_tests()")
    else:
        print("\n[WARNING] Test suite not available")
    
    # Get benchmark suite
    benchmark = toolbox.get_benchmark_suite()
    if benchmark:
        print("[OK] Benchmark suite available")
        print("Run: benchmark.run_all_benchmarks()")
    else:
        print("[WARNING] Benchmark suite not available")


def example_persistence():
    """Example: Using model persistence"""
    print("\n" + "="*60)
    print("Model Persistence Example")
    print("="*60)
    
    toolbox = MLToolbox(check_dependencies=False)
    
    # Get persistence
    persistence = toolbox.get_model_persistence(
        storage_dir="saved_models",
        format='pickle',
        compress=False
    )
    
    if persistence:
        print("\n[OK] Model persistence available")
        print(f"Storage directory: {persistence.storage_dir}")
        print("\nUsage:")
        print("  # Save model")
        print("  persistence.save_model(model, 'my_model', version='1.0.0')")
        print("\n  # Load model")
        print("  model = persistence.load_model('my_model', version='1.0.0')")
    else:
        print("\n[WARNING] Model persistence not available")


def example_optimization():
    """Example: Using model optimization"""
    print("\n" + "="*60)
    print("Model Optimization Example")
    print("="*60)
    
    toolbox = MLToolbox(check_dependencies=False)
    
    # Model compression
    compression = toolbox.get_model_compression()
    if compression:
        print("\n[OK] Model compression available")
        print("Usage:")
        print("  result = compression.quantize_model(model, precision='int8')")
        print("  compressed_model = result['model']")
    else:
        print("\n[WARNING] Model compression not available")
    
    # Model calibration
    calibration = toolbox.get_model_calibration()
    if calibration:
        print("\n[OK] Model calibration available")
        print("Usage:")
        print("  calibrated_model = calibration.calibrate(model, X, y)")
    else:
        print("\n[WARNING] Model calibration not available")


def example_complete_workflow():
    """Example: Complete workflow using Phase 1 components"""
    print("\n" + "="*60)
    print("Complete Workflow Example")
    print("="*60)
    
    toolbox = MLToolbox(check_dependencies=False)
    
    # 1. Train a model
    print("\n1. Training model...")
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    try:
        result = toolbox.fit(X, y, task_type='classification')
        model = result.get('model')
        print("   [OK] Model trained")
    except Exception as e:
        print(f"   [ERROR] Training failed: {e}")
        return
    
    # 2. Save model
    print("\n2. Saving model...")
    persistence = toolbox.get_model_persistence()
    if persistence and model:
        try:
            save_info = persistence.save_model(
                model,
                'example_classifier',
                version='1.0.0',
                metadata={'accuracy': result.get('metrics', {}).get('accuracy', 0)}
            )
            print(f"   [OK] Model saved: {save_info.get('model_path')}")
        except Exception as e:
            print(f"   [ERROR] Save failed: {e}")
    
    # 3. Optimize model
    print("\n3. Optimizing model...")
    compression = toolbox.get_model_compression()
    if compression and model:
        try:
            compressed = compression.quantize_model(model, precision='int8')
            if 'compression_ratio' in compressed:
                print(f"   [OK] Model compressed: {compressed['compression_ratio']:.2%} of original size")
        except Exception as e:
            print(f"   [ERROR] Compression failed: {e}")
    
    # 4. Calibrate model
    print("\n4. Calibrating model...")
    calibration = toolbox.get_model_calibration()
    if calibration and model:
        try:
            calibrated = calibration.calibrate(model, X, y)
            print("   [OK] Model calibrated")
        except Exception as e:
            print(f"   [ERROR] Calibration failed: {e}")
    
    print("\n" + "="*60)
    print("Complete workflow finished!")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PHASE 1 INTEGRATION - USAGE EXAMPLES")
    print("="*60)
    
    example_testing()
    example_persistence()
    example_optimization()
    example_complete_workflow()
    
    print("\n" + "="*60)
    print("All Phase 1 components are ready to use!")
    print("="*60)
