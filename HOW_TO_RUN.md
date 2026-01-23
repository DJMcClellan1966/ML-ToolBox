# How to Run the ML Toolbox - Simple Guide

## 🚀 **Quick Start (3 Steps)**

### **Step 1: Import**
```python
from ml_toolbox import MLToolbox
```

### **Step 2: Initialize**
```python
toolbox = MLToolbox()
```

### **Step 3: Use It!**
```python
import numpy as np

# Generate sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train model (one line!)
result = toolbox.fit(X, y, task_type='classification')

# Done! Check results
print(f"Accuracy: {result.get('accuracy', 0):.2%}")
```

---

## 📝 **Complete Example**

```python
from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox (all features auto-enabled)
toolbox = MLToolbox()

# Your data
X = np.random.randn(200, 10)  # Features
y = np.random.randint(0, 2, 200)  # Labels

# Train model
result = toolbox.fit(X, y, task_type='classification')

# Results
print(f"✅ Model trained!")
print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
print(f"   Model ID: {result.get('model_id')}")

# Make predictions
if result.get('model_id'):
    predictions = toolbox.predict(result['model_id'], X)
    print(f"   Predictions: {predictions[:5]}")
```

---

## 🎯 **Run the Example Script**

```bash
python run_toolbox_example.py
```

This demonstrates:
- Basic model training
- Predictive Intelligence
- Natural Language Pipeline
- Self-Healing Code
- Auto-Optimizer

---

## 💡 **Common Use Cases**

### **1. Train a Classification Model**
```python
toolbox = MLToolbox()
result = toolbox.fit(X, y, task_type='classification')
```

### **2. Train a Regression Model**
```python
toolbox = MLToolbox()
result = toolbox.fit(X, y, task_type='regression')
```

### **3. Use Natural Language**
```python
toolbox = MLToolbox()
result = toolbox.natural_language_pipeline.execute_pipeline(
    "Classify emails as spam or not spam"
)
```

### **4. Use AI Orchestrator**
```python
toolbox = MLToolbox()
result = toolbox.ai_orchestrator.build_optimal_model(
    X, y,
    task_type='classification',
    time_budget=60
)
```

### **5. Use Universal Preprocessor**
```python
toolbox = MLToolbox()
result = toolbox.universal_preprocessor.preprocess(
    data,
    task_type='classification'
)
```

---

## 🎨 **Run the Revolutionary IDE**

```bash
python revolutionary_ide/revolutionary_ide.py
```

**Features:**
- AI-powered code editor
- ML Toolbox integration
- Real-time error fixing
- Visual model training

---

## 📚 **Available Features**

### **Core ML:**
- `toolbox.fit(X, y)` - Train model
- `toolbox.predict(model_id, X)` - Make predictions
- `toolbox.universal_preprocessor` - Preprocess data
- `toolbox.ai_orchestrator` - AI model orchestration
- `toolbox.ai_feature_selector` - Feature selection

### **Revolutionary:**
- `toolbox.predictive_intelligence` - Predicts next actions
- `toolbox.self_healing_code` - Fixes code automatically
- `toolbox.natural_language_pipeline` - Natural language to ML
- `toolbox.collaborative_intelligence` - Community learning
- `toolbox.auto_optimizer` - Auto-optimizes code

### **AI Agent:**
- `toolbox.ai_agent` - AI code generation

---

## ✅ **That's It!**

The toolbox is ready to use. Just:

1. **Import:** `from ml_toolbox import MLToolbox`
2. **Initialize:** `toolbox = MLToolbox()`
3. **Use:** `toolbox.fit(X, y)`

**All revolutionary features are automatically enabled!**

---

## 📖 **More Documentation**

- `QUICK_START_GUIDE.md` - Detailed quick start guide
- `REVOLUTIONARY_FEATURES_GUIDE.md` - Complete feature guide
- `MINDBLOWING_REVOLUTIONARY_TOOLBOX.md` - Complete summary

---

**Happy ML Building!** 🚀
