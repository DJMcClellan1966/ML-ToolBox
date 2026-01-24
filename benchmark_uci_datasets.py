"""
Comprehensive UCI Dataset Benchmark
Compares ML Toolbox vs scikit-learn, PyTorch, and TensorFlow
"""
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# Try to import all frameworks
FRAMEWORKS = {
    'ml_toolbox': False,
    'sklearn': False,
    'pytorch': False,
    'tensorflow': False
}

# ML Toolbox
try:
    from ml_toolbox import MLToolbox
    FRAMEWORKS['ml_toolbox'] = True
except ImportError:
    print("Warning: ML Toolbox not available")

# scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_california_housing
    FRAMEWORKS['sklearn'] = True
except ImportError:
    print("Warning: scikit-learn not available")

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    FRAMEWORKS['pytorch'] = True
except ImportError:
    print("Warning: PyTorch not available")

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    FRAMEWORKS['tensorflow'] = True
except ImportError:
    print("Warning: TensorFlow not available")

print("="*80)
print("UCI DATASET BENCHMARK - ML Toolbox vs Competitors")
print("="*80)
print(f"\nAvailable frameworks: {[k for k, v in FRAMEWORKS.items() if v]}")
print()


class PyTorchClassifier(nn.Module):
    """Simple PyTorch classifier"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class TensorFlowClassifier:
    """Simple TensorFlow classifier"""
    def __init__(self, input_size, num_classes):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X, y, epochs=50, verbose=0):
        self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=32)
    
    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)


def benchmark_iris():
    """Benchmark on Iris dataset"""
    print("\n" + "="*80)
    print("1. IRIS DATASET (Classification)")
    print("="*80)
    
    if not FRAMEWORKS['sklearn']:
        return None
    
    # Load data
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        'dataset': 'Iris',
        'samples': len(X),
        'features': X.shape[1],
        'classes': len(np.unique(y)),
        'frameworks': {}
    }
    
    # ML Toolbox
    if FRAMEWORKS['ml_toolbox']:
        try:
            toolbox = MLToolbox()
            start = time.time()
            result = toolbox.fit(X_train, y_train, task_type='classification')
            fit_time = time.time() - start
            
            start = time.time()
            predictions = toolbox.predict(result['model'], X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['ML Toolbox'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] ML Toolbox:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['ML Toolbox'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] ML Toolbox:     Error - {e}")
    
    # scikit-learn
    if FRAMEWORKS['sklearn']:
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['scikit-learn'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] scikit-learn:   Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['scikit-learn'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] scikit-learn:   Error - {e}")
    
    # PyTorch
    if FRAMEWORKS['pytorch']:
        try:
            model = PyTorchClassifier(X_train.shape[1], len(np.unique(y_train)))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_test_t = torch.FloatTensor(X_test)
            
            start = time.time()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            fit_time = time.time() - start
            
            start = time.time()
            with torch.no_grad():
                outputs = model(X_test_t)
                predictions = torch.argmax(outputs, dim=1).numpy()
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['PyTorch'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] PyTorch:        Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['PyTorch'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] PyTorch:        Error - {e}")
    
    # TensorFlow
    if FRAMEWORKS['tensorflow']:
        try:
            model = TensorFlowClassifier(X_train.shape[1], len(np.unique(y_train)))
            start = time.time()
            model.fit(X_train, y_train, epochs=50, verbose=0)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['TensorFlow'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] TensorFlow:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['TensorFlow'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] TensorFlow:     Error - {e}")
    
    return results


def benchmark_wine():
    """Benchmark on Wine dataset"""
    print("\n" + "="*80)
    print("2. WINE DATASET (Classification)")
    print("="*80)
    
    if not FRAMEWORKS['sklearn']:
        return None
    
    # Load data
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        'dataset': 'Wine',
        'samples': len(X),
        'features': X.shape[1],
        'classes': len(np.unique(y)),
        'frameworks': {}
    }
    
    # ML Toolbox
    if FRAMEWORKS['ml_toolbox']:
        try:
            toolbox = MLToolbox()
            start = time.time()
            result = toolbox.fit(X_train, y_train, task_type='classification')
            fit_time = time.time() - start
            
            start = time.time()
            predictions = toolbox.predict(result['model'], X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['ML Toolbox'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] ML Toolbox:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['ML Toolbox'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] ML Toolbox:     Error - {e}")
    
    # scikit-learn
    if FRAMEWORKS['sklearn']:
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['scikit-learn'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] scikit-learn:   Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['scikit-learn'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] scikit-learn:   Error - {e}")
    
    # PyTorch
    if FRAMEWORKS['pytorch']:
        try:
            model = PyTorchClassifier(X_train.shape[1], len(np.unique(y_train)))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_test_t = torch.FloatTensor(X_test)
            
            start = time.time()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            fit_time = time.time() - start
            
            start = time.time()
            with torch.no_grad():
                outputs = model(X_test_t)
                predictions = torch.argmax(outputs, dim=1).numpy()
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['PyTorch'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] PyTorch:        Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['PyTorch'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] PyTorch:        Error - {e}")
    
    # TensorFlow
    if FRAMEWORKS['tensorflow']:
        try:
            model = TensorFlowClassifier(X_train.shape[1], len(np.unique(y_train)))
            start = time.time()
            model.fit(X_train, y_train, epochs=50, verbose=0)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['TensorFlow'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] TensorFlow:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['TensorFlow'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] TensorFlow:     Error - {e}")
    
    return results


def benchmark_breast_cancer():
    """Benchmark on Breast Cancer dataset"""
    print("\n" + "="*80)
    print("3. BREAST CANCER DATASET (Classification)")
    print("="*80)
    
    if not FRAMEWORKS['sklearn']:
        return None
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        'dataset': 'Breast Cancer',
        'samples': len(X),
        'features': X.shape[1],
        'classes': len(np.unique(y)),
        'frameworks': {}
    }
    
    # ML Toolbox
    if FRAMEWORKS['ml_toolbox']:
        try:
            toolbox = MLToolbox()
            start = time.time()
            result = toolbox.fit(X_train, y_train, task_type='classification')
            fit_time = time.time() - start
            
            start = time.time()
            predictions = toolbox.predict(result['model'], X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['ML Toolbox'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] ML Toolbox:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['ML Toolbox'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] ML Toolbox:     Error - {e}")
    
    # scikit-learn
    if FRAMEWORKS['sklearn']:
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['scikit-learn'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] scikit-learn:   Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['scikit-learn'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] scikit-learn:   Error - {e}")
    
    # PyTorch
    if FRAMEWORKS['pytorch']:
        try:
            model = PyTorchClassifier(X_train.shape[1], len(np.unique(y_train)))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_test_t = torch.FloatTensor(X_test)
            
            start = time.time()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            fit_time = time.time() - start
            
            start = time.time()
            with torch.no_grad():
                outputs = model(X_test_t)
                predictions = torch.argmax(outputs, dim=1).numpy()
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['PyTorch'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] PyTorch:        Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['PyTorch'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] PyTorch:        Error - {e}")
    
    # TensorFlow
    if FRAMEWORKS['tensorflow']:
        try:
            model = TensorFlowClassifier(X_train.shape[1], len(np.unique(y_train)))
            start = time.time()
            model.fit(X_train, y_train, epochs=50, verbose=0)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            accuracy = accuracy_score(y_test, predictions)
            
            results['frameworks']['TensorFlow'] = {
                'accuracy': accuracy,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] TensorFlow:     Accuracy={accuracy:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['TensorFlow'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] TensorFlow:     Error - {e}")
    
    return results


def benchmark_california_housing():
    """Benchmark on California Housing dataset (Regression)"""
    print("\n" + "="*80)
    print("4. CALIFORNIA HOUSING DATASET (Regression)")
    print("="*80)
    
    if not FRAMEWORKS['sklearn']:
        return None
    
    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        'dataset': 'California Housing',
        'samples': len(X),
        'features': X.shape[1],
        'task': 'regression',
        'frameworks': {}
    }
    
    # ML Toolbox
    if FRAMEWORKS['ml_toolbox']:
        try:
            toolbox = MLToolbox()
            start = time.time()
            result = toolbox.fit(X_train, y_train, task_type='regression')
            fit_time = time.time() - start
            
            start = time.time()
            predictions = toolbox.predict(result['model'], X_test)
            predict_time = time.time() - start
            
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            results['frameworks']['ML Toolbox'] = {
                'r2_score': r2,
                'mse': mse,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] ML Toolbox:     R²={r2:.4f}, MSE={mse:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['ML Toolbox'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] ML Toolbox:     Error - {e}")
    
    # scikit-learn
    if FRAMEWORKS['sklearn']:
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test)
            predict_time = time.time() - start
            
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            results['frameworks']['scikit-learn'] = {
                'r2_score': r2,
                'mse': mse,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] scikit-learn:   R²={r2:.4f}, MSE={mse:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['scikit-learn'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] scikit-learn:   Error - {e}")
    
    # PyTorch (Regression)
    if FRAMEWORKS['pytorch']:
        try:
            class PyTorchRegressor(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.fc1 = nn.Linear(input_size, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 1)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            model = PyTorchRegressor(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
            X_test_t = torch.FloatTensor(X_test)
            
            start = time.time()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            fit_time = time.time() - start
            
            start = time.time()
            with torch.no_grad():
                predictions = model(X_test_t).numpy().flatten()
            predict_time = time.time() - start
            
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            results['frameworks']['PyTorch'] = {
                'r2_score': r2,
                'mse': mse,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] PyTorch:        R²={r2:.4f}, MSE={mse:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['PyTorch'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] PyTorch:        Error - {e}")
    
    # TensorFlow (Regression)
    if FRAMEWORKS['tensorflow']:
        try:
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            start = time.time()
            model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=32)
            fit_time = time.time() - start
            
            start = time.time()
            predictions = model.predict(X_test, verbose=0).flatten()
            predict_time = time.time() - start
            
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            results['frameworks']['TensorFlow'] = {
                'r2_score': r2,
                'mse': mse,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'status': 'success'
            }
            print(f"[OK] TensorFlow:     R²={r2:.4f}, MSE={mse:.4f}, Time={fit_time+predict_time:.4f}s")
        except Exception as e:
            results['frameworks']['TensorFlow'] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] TensorFlow:     Error - {e}")
    
    return results


def generate_summary(all_results):
    """Generate summary comparison"""
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    # Calculate averages
    summary = {
        'ml_toolbox': {'accuracy': [], 'r2': [], 'time': [], 'count': 0},
        'sklearn': {'accuracy': [], 'r2': [], 'time': [], 'count': 0},
        'pytorch': {'accuracy': [], 'r2': [], 'time': [], 'count': 0},
        'tensorflow': {'accuracy': [], 'r2': [], 'time': [], 'count': 0}
    }
    
    for result in all_results:
        if result:
            for framework, metrics in result['frameworks'].items():
                if metrics.get('status') == 'success':
                    # Map framework names to summary keys
                    key_map = {
                        'ML Toolbox': 'ml_toolbox',
                        'scikit-learn': 'sklearn',
                        'PyTorch': 'pytorch',
                        'TensorFlow': 'tensorflow'
                    }
                    key = key_map.get(framework, framework.lower().replace('-', '_').replace(' ', '_'))
                    
                    if key in summary:
                        if 'accuracy' in metrics:
                            summary[key]['accuracy'].append(metrics['accuracy'])
                            summary[key]['count'] += 1
                        if 'r2_score' in metrics:
                            summary[key]['r2'].append(metrics['r2_score'])
                            summary[key]['count'] += 1
                        if 'total_time' in metrics:
                            summary[key]['time'].append(metrics['total_time'])
    
    # Print summary
    print("\nAverage Accuracy (Classification):")
    print("-" * 80)
    for framework in ['ml_toolbox', 'sklearn', 'pytorch', 'tensorflow']:
        if summary[framework]['accuracy']:
            avg_acc = np.mean(summary[framework]['accuracy'])
            count = len(summary[framework]['accuracy'])
            print(f"{framework:15s}: {avg_acc:.4f} ({count} datasets)")
    
    print("\nAverage R² Score (Regression):")
    print("-" * 80)
    for framework in ['ml_toolbox', 'sklearn', 'pytorch', 'tensorflow']:
        if summary[framework]['r2']:
            avg_r2 = np.mean(summary[framework]['r2'])
            count = len(summary[framework]['r2'])
            print(f"{framework:15s}: {avg_r2:.4f} ({count} datasets)")
    
    print("\nAverage Training Time:")
    print("-" * 80)
    for framework in ['ml_toolbox', 'sklearn', 'pytorch', 'tensorflow']:
        if summary[framework]['time']:
            avg_time = np.mean(summary[framework]['time'])
            count = len(summary[framework]['time'])
            print(f"{framework:15s}: {avg_time:.4f}s ({count} datasets)")
    
    # Speed comparison
    if summary['sklearn']['time'] and summary['ml_toolbox']['time']:
        sklearn_avg = np.mean(summary['sklearn']['time'])
        toolbox_avg = np.mean(summary['ml_toolbox']['time'])
        ratio = toolbox_avg / sklearn_avg if sklearn_avg > 0 else 0
        print(f"\nML Toolbox vs scikit-learn Speed: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
    
    return summary


def run_all_benchmarks():
    """Run all benchmarks"""
    all_results = []
    
    # Run benchmarks
    all_results.append(benchmark_iris())
    all_results.append(benchmark_wine())
    all_results.append(benchmark_breast_cancer())
    all_results.append(benchmark_california_housing())
    
    # Generate summary
    summary = generate_summary(all_results)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': [r for r in all_results if r],
        'summary': summary
    }
    
    with open('uci_benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to: uci_benchmark_results.json")
    
    return output


if __name__ == '__main__':
    results = run_all_benchmarks()
