# Core ML Only - What Actually Works

## After Removing Bloat

### ✅ **Core Features** (Keep These)

1. **Core ML Models**
   - Regression (Linear, Logistic)
   - Classification (Decision Trees, SVMs, Neural Networks)
   - Evaluation Metrics

2. **Data Preprocessing**
   - Standardization/Normalization (fix preprocessing bug first)
   - Basic feature selection
   - Data cleaning

3. **Pipelines**
   - Feature Pipeline
   - Training Pipeline
   - Inference Pipeline

4. **Proven Specialized Features**
   - Evolutionary Algorithms (works for optimization)
   - Concept Drift Detection (major win)
   - Information Theory (useful for feature selection)

5. **Basic Infrastructure**
   - Model caching
   - Model registry
   - Basic agents (if used)

---

## ❌ **Removed/Disabled** (Bloat)

- Quantum Mechanics
- Philosophy/Religion
- Science Fiction
- Experimental Psychology
- Most Experimental Math

---

## Usage

```python
# Core ML only (default)
from ml_toolbox import MLToolbox
toolbox = MLToolbox()  # experimental_features=False by default

# Use core features
result = toolbox.fit(X, y, preprocess=False)  # Disable broken preprocessing
model = result['model']
predictions = toolbox.predict(model, X_test)
```

**Focus on what works, ignore the rest.**
