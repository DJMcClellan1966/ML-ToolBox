# Machine Learning Best Practices Guide

## Overview

Comprehensive best practices for machine learning covering missing data handling, train/test setup, bias/variance, neural networks, decision trees, ensemble modeling, and model optimization.

---

## Table of Contents

1. [Missing Data Handling](#missing-data-handling)
2. [Training and Testing Setup](#training-and-testing-setup)
3. [Bias and Variance](#bias-and-variance)
4. [Artificial Neural Networks](#artificial-neural-networks)
5. [Decision Trees](#decision-trees)
6. [Ensemble Modeling](#ensemble-modeling)
7. [Model Optimization](#model-optimization)

---

## 1. Missing Data Handling

### Types of Missing Data

1. **MCAR (Missing Completely At Random)** - Missingness is independent of observed and unobserved data
2. **MAR (Missing At Random)** - Missingness depends only on observed data
3. **MNAR (Missing Not At Random)** - Missingness depends on unobserved data

### Strategies for Handling Missing Data

#### Strategy 1: Deletion

```python
# Listwise deletion (remove rows with any missing values)
import pandas as pd
import numpy as np

df = pd.DataFrame(data)
df_clean = df.dropna()  # Remove rows with any NaN

# Pairwise deletion (use available data)
df_clean = df.dropna(subset=['important_column'])
```

**When to use:**
- ✅ MCAR data
- ✅ Small percentage of missing data (< 5%)
- ❌ Avoid if missing data is informative

#### Strategy 2: Imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean/Median/Mode imputation
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# KNN imputation (more sophisticated)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Forward fill / Backward fill (for time series)
df_ffill = df.fillna(method='ffill')  # Forward fill
df_bfill = df.fillna(method='bfill')  # Backward fill
```

**When to use:**
- ✅ MAR data
- ✅ Preserve sample size
- ✅ Maintain relationships

#### Strategy 3: Advanced Imputation

```python
# Using AdvancedDataPreprocessor with scrubbing
from data_preprocessor import AdvancedDataPreprocessor

# Preprocessor can handle missing data in text
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    scrubbing_options={
        'normalize_whitespace': True,
        'fix_encoding': True
    }
)

# For structured data, use ML-based imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iterative_imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(iterative_imputer.fit_transform(df), columns=df.columns)
```

### Best Practices

1. **Understand missingness pattern**
   - Check percentage of missing data
   - Identify patterns (MCAR, MAR, MNAR)
   - Visualize missing data

2. **Choose appropriate strategy**
   - MCAR: Deletion or simple imputation
   - MAR: Advanced imputation (KNN, iterative)
   - MNAR: Model missingness explicitly

3. **Document decisions**
   - Record what was done
   - Justify choices
   - Track impact on model

4. **Validate imputation**
   - Compare distributions before/after
   - Check for introduced bias
   - Monitor model performance

---

## 2. Training and Testing Setup

### Data Splitting Strategies

#### Strategy 1: Simple Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducibility
    stratify=y           # Maintain class distribution
)

# For time series
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

#### Strategy 2: Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Stratified K-Fold (for imbalanced data)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

# Using ML Toolbox
from ml_toolbox import MLToolbox
toolbox = MLToolbox()

evaluator = toolbox.algorithms.get_evaluator()
results = evaluator.evaluate_model(
    model=model,
    X=X_train,
    y=y_train,
    cv=5  # 5-fold cross-validation
)
```

#### Strategy 3: Train/Validation/Test Split

```python
# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Use validation set for hyperparameter tuning
# Use test set only for final evaluation
```

### Best Practices

1. **Maintain data distribution**
   - Use stratified splits for classification
   - Preserve class proportions
   - Handle imbalanced data

2. **Avoid data leakage**
   - Don't use test data for training
   - Fit preprocessors on training data only
   - Apply same preprocessing to test data

3. **Use appropriate splits**
   - **80/20** for large datasets (>10,000 samples)
   - **70/30** for medium datasets (1,000-10,000)
   - **60/40** for small datasets (<1,000)
   - **Cross-validation** for small datasets

4. **Time-aware splitting**
   - Use time-based splits for time series
   - Don't shuffle time series data
   - Respect temporal order

5. **Preprocessing order**
   ```python
   # Correct order:
   # 1. Split data first
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   
   # 2. Fit preprocessor on training data
   preprocessor = AdvancedDataPreprocessor()
   results_train = preprocessor.preprocess(X_train)
   
   # 3. Apply same preprocessor to test data
   results_test = preprocessor.preprocess(X_test)
   ```

---

## 3. Bias and Variance

### Understanding Bias and Variance

**Bias:** Error from overly simplistic assumptions
- **High Bias:** Underfitting, model too simple
- **Low Bias:** Model captures true relationships

**Variance:** Error from sensitivity to small fluctuations
- **High Variance:** Overfitting, model too complex
- **Low Variance:** Model generalizes well

### Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error
```

### Diagnosing Bias and Variance

#### High Bias (Underfitting)

**Symptoms:**
- Poor performance on training data
- Poor performance on test data
- Simple model (e.g., linear regression on non-linear data)

**Solutions:**
- Increase model complexity
- Add more features
- Reduce regularization
- Use more powerful algorithms

```python
# Example: Increase complexity
from sklearn.neural_network import MLPClassifier

# Low complexity (high bias)
model_simple = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)

# Higher complexity (lower bias)
model_complex = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    max_iter=500,
    learning_rate='adaptive'
)
```

#### High Variance (Overfitting)

**Symptoms:**
- Excellent performance on training data
- Poor performance on test data
- Large gap between train/test performance
- Complex model memorizing data

**Solutions:**
- Reduce model complexity
- Add regularization
- Use more training data
- Feature selection
- Early stopping

```python
# Example: Reduce variance
from sklearn.ensemble import RandomForestClassifier

# High variance (overfitting)
model_complex = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,  # No limit
    min_samples_split=2
)

# Lower variance (better generalization)
model_regularized = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Limit depth
    min_samples_split=10,  # Require more samples
    min_samples_leaf=5
)
```

### Using ML Toolbox for Diagnosis

```python
from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier

toolbox = MLToolbox()

# Preprocess data
results = toolbox.data.preprocess(texts, advanced=True)
X = results['compressed_embeddings']

# Evaluate model
evaluator = toolbox.algorithms.get_evaluator()
eval_results = evaluator.evaluate_model(
    model=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    cv=5
)

# Check for overfitting
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
cv_score = eval_results['accuracy']

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"CV Score: {cv_score:.4f}")

# Large gap indicates overfitting (high variance)
if train_score - test_score > 0.1:
    print("Warning: Possible overfitting detected")
```

### Best Practices

1. **Monitor train/test performance**
   - Track both metrics
   - Calculate gap
   - Use learning curves

2. **Use cross-validation**
   - More reliable than single split
   - Reduces variance in estimates
   - Better model selection

3. **Regularization**
   - L1/L2 regularization for neural networks
   - Pruning for decision trees
   - Dropout for neural networks

4. **Early stopping**
   - Stop training when validation performance plateaus
   - Prevents overfitting
   - Saves computation

---

## 4. Artificial Neural Networks

### Architecture Design

#### Input Layer

```python
import torch
import torch.nn as nn

# Determine input size from preprocessed data
from data_preprocessor import AdvancedDataPreprocessor

preprocessor = AdvancedDataPreprocessor(enable_compression=True)
results = preprocessor.preprocess(texts)
X = results['compressed_embeddings']

input_size = X.shape[1]  # Use compressed embedding size
```

#### Hidden Layers

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### Best Practices for Neural Networks

#### 1. **Data Preprocessing**

```python
# Use AdvancedDataPreprocessor
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    enable_compression=True,
    compression_ratio=0.5
)

results = preprocessor.preprocess(texts)
X = results['compressed_embeddings']  # Ready for neural network

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 2. **Architecture Guidelines**

- **Start simple:** 1-2 hidden layers
- **Increase gradually:** Add layers/neurons if needed
- **Use dropout:** 0.3-0.5 for regularization
- **Activation functions:** ReLU for hidden, softmax/sigmoid for output

#### 3. **Training**

```python
import torch.optim as optim

model = NeuralNetwork(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(100):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### 4. **Hyperparameter Tuning**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
tuner = toolbox.algorithms.get_tuner()

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (128, 64)],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout': [0.2, 0.3, 0.5]
}

# Tune hyperparameters
best_params = tuner.tune(
    model=NeuralNetwork,
    X=X_train,
    y=y_train,
    param_grid=param_grid
)
```

### Common Pitfalls

1. ❌ **Too many layers** - Can cause overfitting
2. ❌ **Too large learning rate** - Training instability
3. ❌ **No regularization** - Overfitting
4. ❌ **Improper data scaling** - Training issues
5. ❌ **No validation set** - Can't detect overfitting

---

## 5. Decision Trees

### Decision Tree Basics

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Classification tree
tree_clf = DecisionTreeClassifier(
    max_depth=10,           # Limit depth to prevent overfitting
    min_samples_split=20,   # Minimum samples to split
    min_samples_leaf=10,    # Minimum samples in leaf
    criterion='gini'        # or 'entropy'
)

tree_clf.fit(X_train, y_train)
```

### Best Practices for Decision Trees

#### 1. **Prevent Overfitting**

```python
# Pruning strategies
tree = DecisionTreeClassifier(
    max_depth=10,              # Limit tree depth
    min_samples_split=20,      # Require more samples to split
    min_samples_leaf=10,        # Require more samples in leaves
    max_leaf_nodes=50,         # Limit number of leaves
    min_impurity_decrease=0.01 # Minimum impurity decrease
)
```

#### 2. **Feature Importance**

```python
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_train, y_train)

# Get feature importance
importances = tree.feature_importances_
feature_names = ['feature1', 'feature2', ...]

# Sort by importance
indices = np.argsort(importances)[::-1]

print("Feature Importances:")
for i in range(min(10, len(indices))):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

#### 3. **Visualization**

```python
# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    max_depth=3  # Show first 3 levels
)
plt.show()
```

#### 4. **Handling Categorical Variables**

```python
# One-hot encode categorical variables
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Or use label encoding for tree-based models
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_categorical_encoded = encoder.fit_transform(X_categorical)
```

### Decision Tree Advantages

- ✅ Interpretable
- ✅ Handles non-linear relationships
- ✅ No feature scaling needed
- ✅ Handles mixed data types

### Decision Tree Disadvantages

- ❌ Prone to overfitting
- ❌ Unstable (small data changes → different tree)
- ❌ Biased toward features with more levels
- ❌ Can't extrapolate beyond training range

### Best Practices

1. **Use ensemble methods** (Random Forest, Gradient Boosting)
2. **Prune aggressively** to prevent overfitting
3. **Limit depth** based on data size
4. **Use min_samples_split/leaf** to control complexity
5. **Visualize** to understand decisions

---

## 6. Ensemble Modeling

### Types of Ensembles

#### 1. Voting Ensemble

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Hard voting (majority class)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('svm', SVC())
    ],
    voting='hard'
)

# Soft voting (weighted probabilities)
voting_clf = VotingClassifier(
    estimators=[...],
    voting='soft',
    weights=[2, 1, 1]  # Weight each model
)
```

#### 2. Bagging (Bootstrap Aggregating)

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Bagging with base estimator
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,  # 80% of data per tree
    max_features=0.8,  # 80% of features per tree
    random_state=42
)

# Random Forest (specialized bagging)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
```

#### 3. Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# AdaBoost
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```

#### 4. Stacking

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
ensemble = toolbox.algorithms.get_ensemble()

# Stacking ensemble
from sklearn.linear_model import LogisticRegression

stacking = ensemble.create_stacking_ensemble(
    base_models=[
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50),
        LogisticRegression()
    ],
    meta_model=LogisticRegression()
)

stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)
```

### Using ML Toolbox for Ensembles

```python
from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

toolbox = MLToolbox()

# Preprocess data
results = toolbox.data.preprocess(texts, advanced=True)
X = results['compressed_embeddings']

# Create ensemble
ensemble = toolbox.algorithms.get_ensemble()

# Voting ensemble
voting_results = ensemble.create_voting_ensemble(
    base_models=[
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50)
    ],
    X=X_train,
    y=y_train
)

# Evaluate ensemble
evaluator = toolbox.algorithms.get_evaluator()
eval_results = evaluator.evaluate_model(
    model=ensemble,
    X=X_train,
    y=y_train,
    cv=5
)
```

### Best Practices for Ensembles

1. **Diversity is key**
   - Use different algorithms
   - Use different hyperparameters
   - Use different subsets of data

2. **Start with simple ensembles**
   - Voting with 2-3 models
   - Random Forest (built-in ensemble)
   - Gradually increase complexity

3. **Balance bias and variance**
   - Bagging reduces variance
   - Boosting reduces bias
   - Stacking can reduce both

4. **Monitor performance**
   - Compare ensemble vs individual models
   - Check if ensemble improves performance
   - Avoid overfitting ensemble

5. **Use appropriate ensemble type**
   - **Voting:** Similar performance models
   - **Bagging:** High variance models (trees)
   - **Boosting:** Weak learners
   - **Stacking:** Different model types

---

## 7. Model Optimization

### Hyperparameter Tuning

#### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

#### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=50,  # Number of iterations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
```

#### Using ML Toolbox

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Preprocess data
results = toolbox.data.preprocess(texts, advanced=True)
X = results['compressed_embeddings']

# Get tuner
tuner = toolbox.algorithms.get_tuner()

# Grid search
best_params = tuner.tune(
    model=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    },
    method='grid',
    cv=5
)

# Random search
best_params = tuner.tune(
    model=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    },
    method='random',
    n_iter=50,
    cv=5
)
```

### Feature Engineering

#### Using AdvancedDataPreprocessor

```python
from data_preprocessor import AdvancedDataPreprocessor

preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    enable_compression=True,
    compression_ratio=0.5
)

results = preprocessor.preprocess(texts)

# Automatic features created:
X_embeddings = results['compressed_embeddings']  # Semantic embeddings
categories = results['categorized']              # Categories
quality_scores = results['quality_scores']       # Quality metrics

# Combine features
import numpy as np
import pandas as pd

# One-hot encode categories
category_features = pd.get_dummies([cat for cat in categories.values()])

# Extract quality features
quality_features = np.array([
    [s['score'], s['length'], s['word_count']]
    for s in quality_scores
])

# Combine all features
X_combined = np.column_stack([
    X_embeddings,
    category_features.values,
    quality_features
])
```

### Model Selection

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'SVM': SVC()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['mean'])
print(f"\nBest model: {best_model_name}")
```

### Best Practices for Optimization

1. **Start with defaults**
   - Use default hyperparameters first
   - Establish baseline performance
   - Then optimize

2. **Use cross-validation**
   - More reliable than single split
   - Reduces overfitting to validation set
   - Better hyperparameter selection

3. **Optimize systematically**
   - Start with most important hyperparameters
   - Use grid search for small spaces
   - Use random search for large spaces

4. **Avoid overfitting to validation set**
   - Use separate test set
   - Don't tune on test set
   - Final evaluation on test set only

5. **Consider computational cost**
   - Grid search: Exhaustive but slow
   - Random search: Faster, good coverage
   - Bayesian optimization: Most efficient

6. **Document everything**
   - Record all hyperparameters tried
   - Track performance
   - Note what worked and why

---

## Complete Workflow Example

```python
from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Step 1: Preprocess data
texts = ["text1", "text2", ...]
labels = [0, 1, ...]

results = toolbox.data.preprocess(
    texts,
    advanced=True,
    enable_scrubbing=True,
    enable_compression=True
)

X = results['compressed_embeddings']
y = np.array(labels[:len(X)])

# Step 2: Handle missing data (if any)
# Already handled by preprocessor, but check:
if np.isnan(X).any():
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
evaluator = toolbox.algorithms.get_evaluator()
eval_results = evaluator.evaluate_model(
    model=model,
    X=X_train,
    y=y_train,
    cv=5
)

# Check for overfitting
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"CV Score: {eval_results['accuracy']:.4f}")

if train_score - test_score > 0.1:
    print("Warning: Possible overfitting")

# Step 6: Optimize hyperparameters
tuner = toolbox.algorithms.get_tuner()
best_params = tuner.tune(
    model=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }
)

# Step 7: Train best model
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

# Step 8: Final evaluation on test set
final_score = best_model.score(X_test, y_test)
print(f"Final Test Score: {final_score:.4f}")
```

---

## Summary

### Missing Data
- ✅ Understand missingness pattern (MCAR, MAR, MNAR)
- ✅ Choose appropriate strategy (deletion, imputation)
- ✅ Validate imputation

### Training/Testing Setup
- ✅ Use appropriate splits (80/20, 70/30, 60/40)
- ✅ Maintain data distribution (stratify)
- ✅ Avoid data leakage
- ✅ Use cross-validation for small datasets

### Bias and Variance
- ✅ Monitor train/test performance
- ✅ Use regularization to reduce variance
- ✅ Increase complexity to reduce bias
- ✅ Find optimal balance

### Neural Networks
- ✅ Start simple, increase gradually
- ✅ Use dropout for regularization
- ✅ Standardize features
- ✅ Use early stopping

### Decision Trees
- ✅ Prune to prevent overfitting
- ✅ Limit depth and leaf nodes
- ✅ Use ensemble methods (Random Forest)

### Ensemble Modeling
- ✅ Ensure diversity in models
- ✅ Start with simple ensembles
- ✅ Use appropriate ensemble type
- ✅ Monitor performance

### Model Optimization
- ✅ Start with defaults
- ✅ Use cross-validation
- ✅ Optimize systematically
- ✅ Document everything

---

## Key Takeaways

1. **Data Quality First** - Clean, preprocess, and validate data
2. **Proper Splitting** - Avoid leakage, maintain distribution
3. **Balance Bias/Variance** - Monitor and adjust
4. **Start Simple** - Add complexity gradually
5. **Validate Everything** - Use cross-validation, separate test set
6. **Document Decisions** - Track what works and why

**Use the ML Toolbox and AdvancedDataPreprocessor to implement these best practices!** 🚀
