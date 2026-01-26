"""
Quick UCI Dataset Benchmark Test

Tests ML Toolbox against sklearn on standard UCI datasets.
Uses sklearn's built-in datasets (which are UCI-based) for quick testing.
"""
import numpy as np
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("UCI DATASET BENCHMARK TEST")
print("=" * 80)
print()

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

results = {}

# ============================================================================
# Test 1: Iris Dataset (UCI)
# ============================================================================
print("=" * 80)
print("TEST 1: IRIS DATASET (UCI)")
print("=" * 80)

try:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Baseline: sklearn RandomForest
    print("\n--- Baseline: sklearn RandomForest ---")
    start = time.time()
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_pred = rf_baseline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_time = time.time() - start
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"Time: {rf_time:.4f}s")
    
    # Baseline: sklearn LogisticRegression
    print("\n--- Baseline: sklearn LogisticRegression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start = time.time()
    lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
    lr_baseline.fit(X_train_scaled, y_train)
    lr_pred = lr_baseline.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_time = time.time() - start
    print(f"Accuracy: {lr_acc:.4f}")
    print(f"Time: {lr_time:.4f}s")
    
    # Toolbox: Try to use ML Toolbox
    print("\n--- Toolbox: ML Toolbox ---")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox()
        
        # Use toolbox fit method
        start = time.time()
        result = toolbox.fit(X_train, y_train, task_type='classification', use_cache=False, preprocess=False)
        toolbox_time = time.time() - start
        
        if result and 'model' in result:
            model = result['model']
            if hasattr(model, 'predict'):
                toolbox_pred = model.predict(X_test)
                toolbox_acc = accuracy_score(y_test, toolbox_pred)
                print(f"Accuracy: {toolbox_acc:.4f}")
                print(f"Time: {toolbox_time:.4f}s")
                
                # Compare
                best_baseline = max(rf_acc, lr_acc)
                improvement = ((toolbox_acc - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0
                print(f"\nBest Baseline: {best_baseline:.4f}")
                print(f"Toolbox: {toolbox_acc:.4f}")
                print(f"Improvement: {improvement:+.2f}%")
                
                results['iris'] = {
                    'baseline_rf': rf_acc,
                    'baseline_lr': lr_acc,
                    'best_baseline': best_baseline,
                    'toolbox': toolbox_acc,
                    'improvement': improvement
                }
            else:
                print("Model doesn't have predict method")
                results['iris'] = {'error': 'Model prediction failed'}
        else:
            print("Toolbox fit failed or returned no model")
            results['iris'] = {'error': 'Toolbox fit failed'}
            
    except Exception as e:
        print(f"Toolbox error: {e}")
        import traceback
        traceback.print_exc()
        results['iris'] = {'error': str(e)}
        
except Exception as e:
    print(f"Error in Iris test: {e}")
    import traceback
    traceback.print_exc()
    results['iris'] = {'error': str(e)}

# ============================================================================
# Test 2: Wine Dataset (UCI)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: WINE DATASET (UCI)")
print("=" * 80)

try:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Baseline: sklearn RandomForest
    print("\n--- Baseline: sklearn RandomForest ---")
    start = time.time()
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_pred = rf_baseline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    rf_time = time.time() - start
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")
    print(f"Time: {rf_time:.4f}s")
    
    # Toolbox
    print("\n--- Toolbox: ML Toolbox ---")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox()
        
        start = time.time()
        result = toolbox.fit(X_train, y_train, task_type='classification', use_cache=False, preprocess=False)
        toolbox_time = time.time() - start
        
        if result and 'model' in result:
            model = result['model']
            if hasattr(model, 'predict'):
                toolbox_pred = model.predict(X_test)
                toolbox_acc = accuracy_score(y_test, toolbox_pred)
                toolbox_f1 = f1_score(y_test, toolbox_pred, average='weighted')
                print(f"Accuracy: {toolbox_acc:.4f}")
                print(f"F1 Score: {toolbox_f1:.4f}")
                print(f"Time: {toolbox_time:.4f}s")
                
                improvement = ((toolbox_acc - rf_acc) / rf_acc * 100) if rf_acc > 0 else 0
                print(f"\nBaseline: {rf_acc:.4f}")
                print(f"Toolbox: {toolbox_acc:.4f}")
                print(f"Improvement: {improvement:+.2f}%")
                
                results['wine'] = {
                    'baseline': rf_acc,
                    'toolbox': toolbox_acc,
                    'improvement': improvement
                }
            else:
                results['wine'] = {'error': 'Model prediction failed'}
        else:
            results['wine'] = {'error': 'Toolbox fit failed'}
            
    except Exception as e:
        print(f"Toolbox error: {e}")
        results['wine'] = {'error': str(e)}
        
except Exception as e:
    print(f"Error in Wine test: {e}")
    results['wine'] = {'error': str(e)}

# ============================================================================
# Test 3: Breast Cancer Dataset (UCI)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: BREAST CANCER DATASET (UCI)")
print("=" * 80)

try:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Baseline: sklearn RandomForest
    print("\n--- Baseline: sklearn RandomForest ---")
    start = time.time()
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_pred = rf_baseline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    rf_time = time.time() - start
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")
    print(f"Time: {rf_time:.4f}s")
    
    # Toolbox
    print("\n--- Toolbox: ML Toolbox ---")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox()
        
        start = time.time()
        result = toolbox.fit(X_train, y_train, task_type='classification', use_cache=False, preprocess=False)
        toolbox_time = time.time() - start
        
        if result and 'model' in result:
            model = result['model']
            if hasattr(model, 'predict'):
                toolbox_pred = model.predict(X_test)
                toolbox_acc = accuracy_score(y_test, toolbox_pred)
                toolbox_f1 = f1_score(y_test, toolbox_pred, average='weighted')
                print(f"Accuracy: {toolbox_acc:.4f}")
                print(f"F1 Score: {toolbox_f1:.4f}")
                print(f"Time: {toolbox_time:.4f}s")
                
                improvement = ((toolbox_acc - rf_acc) / rf_acc * 100) if rf_acc > 0 else 0
                print(f"\nBaseline: {rf_acc:.4f}")
                print(f"Toolbox: {toolbox_acc:.4f}")
                print(f"Improvement: {improvement:+.2f}%")
                
                results['breast_cancer'] = {
                    'baseline': rf_acc,
                    'toolbox': toolbox_acc,
                    'improvement': improvement
                }
            else:
                results['breast_cancer'] = {'error': 'Model prediction failed'}
        else:
            results['breast_cancer'] = {'error': 'Toolbox fit failed'}
            
    except Exception as e:
        print(f"Toolbox error: {e}")
        results['breast_cancer'] = {'error': str(e)}
        
except Exception as e:
    print(f"Error in Breast Cancer test: {e}")
    results['breast_cancer'] = {'error': str(e)}

# ============================================================================
# Test 4: Digits Dataset (UCI-based)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: DIGITS DATASET (UCI-based)")
print("=" * 80)

try:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Baseline: sklearn RandomForest
    print("\n--- Baseline: sklearn RandomForest ---")
    start = time.time()
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_pred = rf_baseline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_time = time.time() - start
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"Time: {rf_time:.4f}s")
    
    # Toolbox
    print("\n--- Toolbox: ML Toolbox ---")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox()
        
        start = time.time()
        result = toolbox.fit(X_train, y_train, task_type='classification', use_cache=False, preprocess=False)
        toolbox_time = time.time() - start
        
        if result and 'model' in result:
            model = result['model']
            if hasattr(model, 'predict'):
                toolbox_pred = model.predict(X_test)
                toolbox_acc = accuracy_score(y_test, toolbox_pred)
                print(f"Accuracy: {toolbox_acc:.4f}")
                print(f"Time: {toolbox_time:.4f}s")
                
                improvement = ((toolbox_acc - rf_acc) / rf_acc * 100) if rf_acc > 0 else 0
                print(f"\nBaseline: {rf_acc:.4f}")
                print(f"Toolbox: {toolbox_acc:.4f}")
                print(f"Improvement: {improvement:+.2f}%")
                
                results['digits'] = {
                    'baseline': rf_acc,
                    'toolbox': toolbox_acc,
                    'improvement': improvement
                }
            else:
                results['digits'] = {'error': 'Model prediction failed'}
        else:
            results['digits'] = {'error': 'Toolbox fit failed'}
            
    except Exception as e:
        print(f"Toolbox error: {e}")
        results['digits'] = {'error': str(e)}
        
except Exception as e:
    print(f"Error in Digits test: {e}")
    results['digits'] = {'error': str(e)}

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

wins = 0
losses = 0
ties = 0
improvements = []

for dataset, result in results.items():
    if 'error' not in result and 'improvement' in result:
        improvement = result['improvement']
        improvements.append(improvement)
        if improvement > 1.0:
            wins += 1
        elif improvement < -1.0:
            losses += 1
        else:
            ties += 1

if improvements:
    avg_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)
    
    print(f"\nDatasets Tested: {len(results)}")
    print(f"Successful Tests: {len(improvements)}")
    print(f"Toolbox Wins: {wins}")
    print(f"Baseline Wins: {losses}")
    print(f"Ties: {ties}")
    print(f"\nAverage Improvement: {avg_improvement:+.2f}%")
    print(f"Median Improvement: {median_improvement:+.2f}%")
    print(f"Max Improvement: {max_improvement:+.2f}%")
    print(f"Min Improvement: {min_improvement:+.2f}%")
    
    print("\nDetailed Results:")
    for dataset, result in results.items():
        if 'error' not in result:
            print(f"\n{dataset.upper()}:")
            if 'baseline' in result:
                print(f"  Baseline: {result.get('baseline', 'N/A'):.4f}")
            if 'toolbox' in result:
                print(f"  Toolbox: {result.get('toolbox', 'N/A'):.4f}")
            if 'improvement' in result:
                print(f"  Improvement: {result.get('improvement', 0):+.2f}%")
        else:
            print(f"\n{dataset.upper()}: ERROR - {result['error']}")
else:
    print("\nNo successful tests completed. Check errors above.")

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)
