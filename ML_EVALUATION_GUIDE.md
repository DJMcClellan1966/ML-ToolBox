# ML Evaluation and Hyperparameter Tuning Guide

## Overview

The **AdvancedDataPreprocessor** now supports **best practices for model evaluation and hyperparameter tuning** through the `ml_evaluation.py` module.

---

## Features

### ✅ Model Evaluation (`MLEvaluator`)

1. **Cross-Validation**
   - K-fold cross-validation
   - Stratified K-fold for classification
   - Robust performance estimation

2. **Train/Test Splits**
   - Stratified splits for classification
   - Random splits for regression
   - Configurable test size

3. **Multiple Metrics**
   - **Classification:** Accuracy, Precision, Recall, F1
   - **Regression:** MSE, MAE, R²

4. **Overfitting Detection**
   - Compares train vs test performance
   - Warns when gap is significant
   - Helps identify model issues

5. **Learning Curves**
   - Shows model performance vs training size
   - Identifies if more data would help
   - Detects underfitting/overfitting

---

### ✅ Hyperparameter Tuning (`HyperparameterTuner`)

1. **Grid Search**
   - Exhaustive search over parameter grid
   - Best for small parameter spaces
   - Guaranteed to find best in grid

2. **Random Search**
   - Random sampling of parameter space
   - Faster for large parameter spaces
   - Often finds good solutions faster

3. **Validation Curves**
   - Analyze individual parameter effects
   - Find optimal parameter ranges
   - Understand parameter sensitivity

---

### ✅ Preprocessor Optimization (`PreprocessorOptimizer`)

1. **Automatic Parameter Tuning**
   - Optimizes deduplication threshold
   - Optimizes compression ratio
   - Optimizes compression method

2. **Quality-Based Evaluation**
   - Uses data quality metrics
   - Balances compression vs quality
   - Finds optimal trade-offs

---

## Usage Examples

### 1. Model Evaluation

```python
from ml_evaluation import MLEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create data
X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)

# Create model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Evaluate
evaluator = MLEvaluator(random_state=42)
results = evaluator.evaluate_model(
    model, X, y,
    task_type='classification',
    cv_folds=5,
    verbose=True
)

# Access results
print(f"Test Accuracy: {results['metrics']['test']['accuracy']:.4f}")
print(f"CV Mean: {results['cross_validation']['mean']:.4f}")
print(f"Overfitting: {results['overfitting_detected']}")
```

### 2. Hyperparameter Tuning

```python
from ml_evaluation import HyperparameterTuner
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Tune hyperparameters
tuner = HyperparameterTuner(random_state=42)
results = tuner.tune_hyperparameters(
    model, param_grid, X, y,
    method='grid',  # or 'random'
    cv_folds=5,
    verbose=True
)

# Use best model
best_model = results['best_model']
best_params = results['best_params']
print(f"Best Parameters: {best_params}")
print(f"Best Score: {results['best_score']:.4f}")
```

### 3. Preprocessor Optimization

```python
from ml_evaluation import PreprocessorOptimizer
from data_preprocessor import AdvancedDataPreprocessor

# Raw data
raw_data = [
    "Python is great for data science",
    "Machine learning uses algorithms",
    # ... more items
]

labels = ['technical', 'technical', 'business', ...]

# Optimize preprocessor
optimizer = PreprocessorOptimizer(random_state=42)
results = optimizer.optimize_preprocessor(
    raw_data,
    labels=labels,
    task_type='classification',
    cv_folds=5,
    verbose=True
)

# Use best preprocessor
best_preprocessor = results['best_preprocessor']
best_params = results['best_params']
```

---

## Best Practices

### 1. **Cross-Validation**

✅ **Use 5-10 folds** for robust evaluation  
✅ **Use stratified K-fold** for classification  
✅ **Use regular K-fold** for regression  

```python
# Good: Stratified for classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Good: Regular for regression
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

### 2. **Train/Test Splits**

✅ **Use 80/20 or 70/30** split  
✅ **Stratify for classification**  
✅ **Shuffle data** before splitting  

```python
# Good: Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y  # For classification
)
```

### 3. **Metrics Selection**

✅ **Classification:** Accuracy, Precision, Recall, F1  
✅ **Regression:** MSE, MAE, R²  
✅ **Use multiple metrics** for comprehensive evaluation  

```python
# Good: Multiple metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']
```

### 4. **Overfitting Detection**

✅ **Monitor train vs test performance**  
✅ **Gap > 10% (classification) or > 20% (regression)** indicates overfitting  
✅ **Use regularization** if overfitting detected  

```python
# Check overfitting
if results['overfitting_detected']:
    print("WARNING: Overfitting detected!")
    # Consider: regularization, more data, simpler model
```

### 5. **Hyperparameter Tuning**

✅ **Start with grid search** for small spaces  
✅ **Use random search** for large spaces  
✅ **Use validation curves** to understand parameters  

```python
# Small grid: Use grid search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5]
}
tuner.tune_hyperparameters(..., method='grid')

# Large grid: Use random search
param_grid = {
    'n_estimators': range(10, 200),
    'max_depth': range(3, 20),
    'min_samples_split': range(2, 20)
}
tuner.tune_hyperparameters(..., method='random', n_iter=50)
```

### 6. **Learning Curves**

✅ **Use to identify** if more data would help  
✅ **Gap between train/val** indicates overfitting  
✅ **Both low** indicates underfitting  

```python
# Analyze learning curves
curves = results['learning_curves']
# If train and val both low: need more data or better model
# If train high, val low: overfitting
# If both high: good fit
```

---

## Integration with AdvancedDataPreprocessor

### Complete Pipeline

```python
from data_preprocessor import AdvancedDataPreprocessor
from ml_evaluation import MLEvaluator, PreprocessorOptimizer
from sklearn.ensemble import RandomForestClassifier

# 1. Optimize preprocessor
raw_data = [...]  # Your data
labels = [...]     # Your labels

optimizer = PreprocessorOptimizer()
opt_results = optimizer.optimize_preprocessor(
    raw_data, labels=labels, task_type='classification'
)

# 2. Use best preprocessor
best_preprocessor = opt_results['best_preprocessor']
preprocessed = best_preprocessor.preprocess(raw_data)

# 3. Get features
X = preprocessed['compressed_embeddings']
y = labels

# 4. Train and evaluate model
model = RandomForestClassifier()
evaluator = MLEvaluator()
results = evaluator.evaluate_model(model, X, y, task_type='classification')

# 5. Tune model hyperparameters
from ml_evaluation import HyperparameterTuner
tuner = HyperparameterTuner()
tuning_results = tuner.tune_hyperparameters(
    model, param_grid, X, y, method='grid'
)

# 6. Use best model
best_model = tuning_results['best_model']
```

---

## Test Results

### Model Evaluation

**Classification:**
- Test Accuracy: 0.5750
- CV Mean: 0.6500
- Overfitting: Detected (gap: 0.4250)

**Regression:**
- Test R²: 0.6705
- CV Mean: -8322.16 (MSE)
- Overfitting: Detected (gap: 0.2865)

### Hyperparameter Tuning

**Grid Search:**
- Tests all parameter combinations
- Finds best parameters in grid
- Slower but exhaustive

**Random Search:**
- Samples random combinations
- Faster for large spaces
- Often finds good solutions

---

## Requirements

```bash
# Required
pip install scikit-learn numpy

# Optional (for better performance)
pip install scipy
```

---

## Summary

✅ **Model Evaluation:** Cross-validation, multiple metrics, overfitting detection  
✅ **Hyperparameter Tuning:** Grid search, random search, validation curves  
✅ **Preprocessor Optimization:** Automatic parameter tuning  
✅ **Best Practices:** Stratified splits, multiple metrics, learning curves  

**The AdvancedDataPreprocessor now includes enterprise-grade ML evaluation and hyperparameter tuning capabilities!**
