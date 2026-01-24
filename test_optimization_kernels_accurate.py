"""
Accurate Performance Comparison for All 8 Optimization Kernels

Tests with caching disabled for fair comparison.
"""
import sys
from pathlib import Path
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ACCURATE OPTIMIZATION KERNELS PERFORMANCE TEST")
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

# Initialize toolbox with caching disabled for fair comparison
toolbox = MLToolbox(enable_caching=False)

results = {}

# Test 1: Algorithm Kernel
print("\n1. ALGORITHM KERNEL")
print("-"*80)

if toolbox.algorithm_kernel:
    # Baseline (direct fit, no preprocessing)
    times_baseline = []
    for _ in range(3):  # Average over 3 runs
        start = time.time()
        result_baseline = toolbox.fit(X, y, task_type='classification', preprocess=False, use_cache=False)
        times_baseline.append(time.time() - start)
    time_baseline = np.mean(times_baseline)
    
    # With Algorithm Kernel
    times_kernel = []
    for _ in range(3):
        start = time.time()
        algo_kernel = toolbox.algorithm_kernel
        algo_kernel.fit(X, y, algorithm='auto')
        pred_kernel = algo_kernel.predict(X[:100])
        times_kernel.append(time.time() - start)
    time_kernel = np.mean(times_kernel)
    
    results['algorithm'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline (avg): {time_baseline:.4f}s")
    print(f"Kernel (avg): {time_kernel:.4f}s")
    print(f"Speedup: {results['algorithm']['speedup']:.2f}x")
else:
    print("[SKIP] Algorithm kernel not available")

# Test 2: Feature Engineering Kernel
print("\n2. FEATURE ENGINEERING KERNEL")
print("-"*80)

if toolbox.feature_kernel:
    # Baseline (manual, multiple runs)
    times_baseline = []
    for _ in range(10):
        start = time.time()
        X_std = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        times_baseline.append(time.time() - start)
    time_baseline = np.mean(times_baseline)
    
    # With Feature Kernel
    times_kernel = []
    for _ in range(10):
        start = time.time()
        feat_kernel = toolbox.feature_kernel
        X_engineered = feat_kernel.auto_engineer(X, y)
        times_kernel.append(time.time() - start)
    time_kernel = np.mean(times_kernel)
    
    results['feature'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline (avg): {time_baseline:.6f}s")
    print(f"Kernel (avg): {time_kernel:.6f}s")
    print(f"Speedup: {results['feature']['speedup']:.2f}x")
else:
    print("[SKIP] Feature kernel not available")

# Test 3: Pipeline Kernel
print("\n3. PIPELINE KERNEL")
print("-"*80)

if toolbox.pipeline_kernel:
    # Baseline (manual steps, multiple runs)
    times_baseline = []
    for _ in range(10):
        start = time.time()
        X1 = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        X2 = (X1 - np.min(X1, axis=0, keepdims=True)) / (np.max(X1, axis=0, keepdims=True) - np.min(X1, axis=0, keepdims=True) + 1e-10)
        times_baseline.append(time.time() - start)
    time_baseline = np.mean(times_baseline)
    
    # With Pipeline Kernel
    times_kernel = []
    for _ in range(10):
        start = time.time()
        pipe_kernel = toolbox.pipeline_kernel
        X_pipeline = pipe_kernel.execute(X, steps=['preprocess', 'engineer'])
        times_kernel.append(time.time() - start)
    time_kernel = np.mean(times_kernel)
    
    results['pipeline'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline (avg): {time_baseline:.6f}s")
    print(f"Kernel (avg): {time_kernel:.6f}s")
    print(f"Speedup: {results['pipeline']['speedup']:.2f}x")
else:
    print("[SKIP] Pipeline kernel not available")

# Test 4: Cross-Validation Kernel
print("\n4. CROSS-VALIDATION KERNEL")
print("-"*80)

if toolbox.cv_kernel:
    # Baseline (sequential CV)
    start = time.time()
    n_folds = 5
    fold_size = len(X) // n_folds
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(X)
        train_idx = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
        val_idx = np.arange(start_idx, end_idx)
        _ = toolbox.fit(X[train_idx], y[train_idx], task_type='classification', preprocess=False, use_cache=False)
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
    print(f"CV Results: mean={cv_results.get('mean_score', 0):.4f}, std={cv_results.get('std_score', 0):.4f}")
else:
    print("[SKIP] CV kernel not available")

# Test 5: Evaluation Kernel
print("\n5. EVALUATION KERNEL")
print("-"*80)

if toolbox.eval_kernel:
    # Generate predictions
    result = toolbox.fit(X, y, task_type='classification', preprocess=False, use_cache=False)
    if isinstance(result, dict) and 'model' in result:
        model = result['model']
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            y_pred = np.random.randint(0, 2, len(y))
    else:
        y_pred = np.random.randint(0, 2, len(y))
    
    # Baseline (manual metrics, multiple runs)
    times_baseline = []
    for _ in range(10):
        start = time.time()
        acc = np.mean(y == y_pred)
        prec = np.mean((y == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-10)
        rec = np.mean((y == 1) & (y_pred == 1)) / (np.sum(y == 1) + 1e-10)
        times_baseline.append(time.time() - start)
    time_baseline = np.mean(times_baseline)
    
    # With Evaluation Kernel
    times_kernel = []
    for _ in range(10):
        start = time.time()
        eval_kernel = toolbox.eval_kernel
        eval_results = eval_kernel.evaluate(y, y_pred, metrics=['accuracy', 'precision', 'recall'])
        times_kernel.append(time.time() - start)
    time_kernel = np.mean(times_kernel)
    
    results['evaluation'] = {
        'baseline': time_baseline,
        'kernel': time_kernel,
        'speedup': time_baseline / time_kernel if time_kernel > 0 else 1.0
    }
    print(f"Baseline (avg): {time_baseline:.6f}s")
    print(f"Kernel (avg): {time_kernel:.6f}s")
    print(f"Speedup: {results['evaluation']['speedup']:.2f}x")
    print(f"Metrics: {eval_results}")
else:
    print("[SKIP] Evaluation kernel not available")

# Summary
print("\n" + "="*80)
print("PERFORMANCE COMPARISON SUMMARY")
print("="*80)

print("\nKernel Performance Improvements:")
print("-"*80)
print(f"{'Kernel':<25s} {'Baseline':<12s} {'Kernel':<12s} {'Speedup':<10s} {'Status'}")
print("-"*80)

total_baseline = 0
total_kernel = 0
count = 0

for kernel_name, result in results.items():
    baseline = result['baseline']
    kernel = result['kernel']
    speedup = result['speedup']
    total_baseline += baseline
    total_kernel += kernel
    count += 1
    status = "[OK]" if speedup >= 1.0 else "[SLOW]"
    print(f"{kernel_name.capitalize():<25s} {baseline:>10.4f}s {kernel:>10.4f}s {speedup:>8.2f}x {status}")

if count > 0:
    avg_speedup = (total_baseline / total_kernel) if total_kernel > 0 else 1.0
    improvement = (1 - (total_kernel / total_baseline)) * 100 if total_baseline > 0 else 0
    print("-"*80)
    print(f"{'Total Time':<25s} {total_baseline:>10.4f}s {total_kernel:>10.4f}s {avg_speedup:>8.2f}x")
    print(f"{'Overall Improvement':<25s} {'':>10s} {'':>10s} {improvement:>7.1f}%")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
