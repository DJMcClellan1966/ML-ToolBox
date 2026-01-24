"""
Performance Comparison Test for All 8 Optimization Kernels

Tests performance improvements from all optimization kernels compared to
baseline implementations.
"""
import sys
from pathlib import Path
import numpy as np
import time
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ALL 8 OPTIMIZATION KERNELS - PERFORMANCE COMPARISON")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    print("\n[OK] ML Toolbox imported")
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)

# Generate test data
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features).astype(np.float64)
y = np.random.randint(0, 2, n_samples)

print(f"\nTest data: {X.shape}, {len(y)} samples")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

results = {}

# Test 1: Algorithm Kernel
print("\n1. ALGORITHM KERNEL")
print("-"*80)

if toolbox.algorithm_kernel:
    # Baseline (direct fit)
    start = time.time()
    result_baseline = toolbox.fit(X, y, task_type='classification', preprocess=False)
    time_baseline = time.time() - start
    
    # With Algorithm Kernel
    start = time.time()
    algo_kernel = toolbox.algorithm_kernel
    algo_kernel.fit(X, y, algorithm='auto')
    pred_kernel = algo_kernel.predict(X[:100])
    time_kernel = time.time() - start
    
    results['algorithm'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['algorithm']['speedup']:.2f}x")
else:
    print("[SKIP] Algorithm kernel not available")

# Test 2: Feature Engineering Kernel
print("\n2. FEATURE ENGINEERING KERNEL")
print("-"*80)

if toolbox.feature_kernel:
    # Baseline (manual)
    start = time.time()
    X_std = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    time_baseline = time.time() - start
    
    # With Feature Kernel
    start = time.time()
    feat_kernel = toolbox.feature_kernel
    X_engineered = feat_kernel.auto_engineer(X, y)
    time_kernel = time.time() - start
    
    results['feature'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['feature']['speedup']:.2f}x")
else:
    print("[SKIP] Feature kernel not available")

# Test 3: Pipeline Kernel
print("\n3. PIPELINE KERNEL")
print("-"*80)

if toolbox.pipeline_kernel:
    # Baseline (manual steps)
    start = time.time()
    X1 = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    X2 = (X1 - np.min(X1, axis=0, keepdims=True)) / (np.max(X1, axis=0, keepdims=True) - np.min(X1, axis=0, keepdims=True))
    time_baseline = time.time() - start
    
    # With Pipeline Kernel
    start = time.time()
    pipe_kernel = toolbox.pipeline_kernel
    X_pipeline = pipe_kernel.execute(X, steps=['preprocess', 'engineer'])
    time_kernel = time.time() - start
    
    results['pipeline'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['pipeline']['speedup']:.2f}x")
else:
    print("[SKIP] Pipeline kernel not available")

# Test 4: Ensemble Kernel
print("\n4. ENSEMBLE KERNEL")
print("-"*80)

if toolbox.ensemble_kernel:
    # Baseline (sequential)
    start = time.time()
    model1 = toolbox.fit(X, y, task_type='classification', preprocess=False)
    model2 = toolbox.fit(X, y, task_type='classification', preprocess=False)
    time_baseline = time.time() - start
    
    # With Ensemble Kernel
    start = time.time()
    ens_kernel = toolbox.ensemble_kernel
    ens_kernel.create_ensemble(X, y, models=['rf', 'lr'], method='voting')
    pred_ens = ens_kernel.predict(X[:100])
    time_kernel = time.time() - start
    
    results['ensemble'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['ensemble']['speedup']:.2f}x")
else:
    print("[SKIP] Ensemble kernel not available")

# Test 5: Tuning Kernel
print("\n5. TUNING KERNEL")
print("-"*80)

if toolbox.tuning_kernel:
    # Baseline (manual)
    search_space = {'n_estimators': [10, 50], 'max_depth': [3, 5]}
    start = time.time()
    # Simple manual search
    for n_est in search_space['n_estimators']:
        for depth in search_space['max_depth']:
            _ = toolbox.fit(X, y, task_type='classification', preprocess=False)
    time_baseline = time.time() - start
    
    # With Tuning Kernel
    start = time.time()
    tune_kernel = toolbox.tuning_kernel
    best_params = tune_kernel.tune('rf', X, y, search_space, method='grid')
    time_kernel = time.time() - start
    
    results['tuning'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['tuning']['speedup']:.2f}x")
else:
    print("[SKIP] Tuning kernel not available")

# Test 6: Cross-Validation Kernel
print("\n6. CROSS-VALIDATION KERNEL")
print("-"*80)

if toolbox.cv_kernel:
    # Baseline (manual CV)
    start = time.time()
    n_folds = 5
    fold_size = len(X) // n_folds
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(X)
        train_idx = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
        val_idx = np.arange(start_idx, end_idx)
        _ = toolbox.fit(X[train_idx], y[train_idx], task_type='classification', preprocess=False)
    time_baseline = time.time() - start
    
    # With CV Kernel
    start = time.time()
    cv_kernel = toolbox.cv_kernel
    cv_results = cv_kernel.cross_validate(X, y, cv=5)
    time_kernel = time.time() - start
    
    results['cv'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['cv']['speedup']:.2f}x")
else:
    print("[SKIP] CV kernel not available")

# Test 7: Evaluation Kernel
print("\n7. EVALUATION KERNEL")
print("-"*80)

if toolbox.eval_kernel:
    # Generate predictions
    result = toolbox.fit(X, y, task_type='classification', preprocess=False)
    if isinstance(result, dict) and 'model' in result:
        model = result['model']
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            y_pred = np.random.randint(0, 2, len(y))
    else:
        y_pred = np.random.randint(0, 2, len(y))
    
    # Baseline (manual metrics)
    start = time.time()
    acc = np.mean(y == y_pred)
    prec = 0.5  # Simplified
    rec = 0.5   # Simplified
    time_baseline = time.time() - start
    
    # With Evaluation Kernel
    start = time.time()
    eval_kernel = toolbox.eval_kernel
    eval_results = eval_kernel.evaluate(y, y_pred, metrics=['accuracy', 'precision', 'recall'])
    time_kernel = time.time() - start
    
    results['evaluation'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['evaluation']['speedup']:.2f}x")
else:
    print("[SKIP] Evaluation kernel not available")

# Test 8: Serving Kernel
print("\n8. SERVING KERNEL")
print("-"*80)

if toolbox.serving_kernel:
    # Train model
    result = toolbox.fit(X, y, task_type='classification', preprocess=False)
    if isinstance(result, dict) and 'model' in result:
        model = result['model']
    else:
        model = result
    
    # Baseline (sequential prediction)
    X_test = X[:100]
    start = time.time()
    if hasattr(model, 'predict'):
        pred_baseline = model.predict(X_test)
    else:
        pred_baseline = np.random.randint(0, 2, len(X_test))
    time_baseline = time.time() - start
    
    # With Serving Kernel
    start = time.time()
    serve_kernel = toolbox.serving_kernel
    pred_serve = serve_kernel.serve(model, X_test, batch_size=50)
    time_kernel = time.time() - start
    
    results['serving'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline: {time_baseline:.4f}s")
    print(f"Kernel: {time_kernel:.4f}s")
    print(f"Speedup: {results['serving']['speedup']:.2f}x")
else:
    print("[SKIP] Serving kernel not available")

# Summary
print("\n" + "="*80)
print("PERFORMANCE COMPARISON SUMMARY")
print("="*80)

print("\nKernel Performance Improvements:")
print("-"*80)

total_speedup = 0
count = 0

for kernel_name, result in results.items():
    speedup = result['speedup']
    total_speedup += speedup
    count += 1
    status = "[OK]" if speedup > 1.0 else "[SLOW]"
    print(f"{kernel_name.capitalize():20s}: {speedup:6.2f}x {status}")

if count > 0:
    avg_speedup = total_speedup / count
    print("-"*80)
    print(f"{'Average Speedup':20s}: {avg_speedup:6.2f}x")
    print(f"{'Overall Improvement':20s}: {(avg_speedup - 1) * 100:5.1f}%")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
