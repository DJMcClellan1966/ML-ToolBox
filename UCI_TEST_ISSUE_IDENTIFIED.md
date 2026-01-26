# UCI Test Issue - Root Cause Identified

## The Problem

The toolbox is getting ~30% accuracy on UCI datasets when sklearn gets 97-100%.

## Root Cause

### **Issue 1: Double Data Splitting**

The `simple_ml_tasks.train_classifier()` method does its OWN train_test_split:

```python
# In simple_ml_tasks.py line 69-71
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

But when we call `toolbox.fit(X_train, y_train)`, we're passing in **already-split data** from our test. So:
- We split: 70% train, 30% test
- `simple_ml_tasks` splits AGAIN: 80% of our 70% = 56% of original data
- Model trains on only 56% of data!

### **Issue 2: Preprocessing May Be Corrupting Data**

The `toolbox.fit()` method preprocesses data with computational kernels:

```python
# In ml_toolbox/__init__.py line 799-801
if preprocess and use_kernels and self._comp_kernel is not None:
    X = self._comp_kernel.standardize(X)
```

This preprocessing may be:
- Changing data in unexpected ways
- Corrupting feature relationships
- Causing shape mismatches

### **Issue 3: Model Not Trained on Full Data**

Because of double splitting, the model is:
- Trained on 56% of original data (80% of 70%)
- Then tested on our 30% test set
- This mismatch causes poor performance

## Evidence

When we test `simple_ml_tasks` directly (without toolbox wrapper):
- Model type: RandomForestClassifier ✅
- Model accuracy: 90.48% ✅ (on its own test set)
- Test accuracy: 100% ✅ (on our test set)

So `simple_ml_tasks` WORKS when used directly, but fails when used through `toolbox.fit()`.

## The Fix

1. **Don't split data in `simple_ml_tasks`** if data is already split
2. **Or**: Don't pre-split data before calling `toolbox.fit()`
3. **Or**: Add a parameter to `simple_ml_tasks` to skip splitting
4. **Check preprocessing**: Verify computational kernel standardization isn't corrupting data

## Quick Fix Test

Test without preprocessing:

```python
result = toolbox.fit(X_train, y_train, preprocess=False)
```

This should help identify if preprocessing is the issue.
