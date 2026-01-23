# ML Toolbox - Quick Start Guide

## 🚀 **How to Run the Toolbox**

### **1. Basic Setup**

```python
from ml_toolbox import MLToolbox

# Initialize toolbox
toolbox = MLToolbox()

# That's it! All features are automatically enabled
```

---

## 📋 **Basic Usage Examples**

### **Example 1: Simple Model Training**

```python
from ml_toolbox import MLToolbox
import numpy as np

# Initialize
toolbox = MLToolbox()

# Generate sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train model (one line!)
result = toolbox.fit(X, y, task_type='classification')

# Check results
print(f"Accuracy: {result.get('accuracy', 0):.2%}")
print(f"Model ID: {result.get('model_id', 'N/A')}")
```

---

### **Example 2: Using Natural Language Pipeline**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Just describe what you want!
description = "Classify data into 3 classes"

# Build and execute pipeline
result = toolbox.natural_language_pipeline.execute_pipeline(description)

# Complete pipeline built and executed automatically!
```

---

### **Example 3: Using Predictive Intelligence**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Train a model
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y)

# Record action (automatic, but you can do it manually)
toolbox.predictive_intelligence.record_action('train_model', {'model_type': 'auto'})

# Get suggestions for what to do next
suggestions = toolbox.predictive_intelligence.get_suggestions({'action': 'train_model'})
print("Suggestions:", suggestions)
# Output: ['Evaluate the trained model', 'Save the model for later use']
```

---

### **Example 4: Using Self-Healing Code**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Code with potential issues
code = """
toolbox.fit(X, y)
"""

# Analyze code
analysis = toolbox.self_healing_code.analyze_code(code)
print(f"Issues found: {len(analysis['issues'])}")

# Heal code automatically
result = toolbox.self_healing_code.heal_code(code)
print(f"Issues fixed: {result['issues_fixed']}")
print("Healed code:")
print(result['healed_code'])
```

---

### **Example 5: Using Auto-Optimizer**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Code that could be optimized
code = """
for i in range(len(X)):
    process(X[i])
toolbox.fit(X, y)
"""

# Analyze for optimization opportunities
analysis = toolbox.auto_optimizer.analyze_code(code)
print(f"Optimization opportunities: {len(analysis['opportunities'])}")

# Optimize automatically
result = toolbox.auto_optimizer.optimize_code(code)
print(f"Optimizations applied: {result['optimizations_applied']}")
print("Optimized code:")
print(result['optimized_code'])
```

---

## 🎯 **Complete Workflow Example**

```python
from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# 1. Preprocess data (using Universal Preprocessor)
text_data = ["sample text 1", "sample text 2", "sample text 3"]
preprocessed = toolbox.universal_preprocessor.preprocess(text_data, task_type='classification')

# 2. Train model
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y, task_type='classification')

# 3. Get predictions
model_id = result.get('model_id')
if model_id:
    predictions = toolbox.predict(model_id, X)
    print(f"Predictions: {predictions}")

# 4. Use AI Agent to generate more code
agent_result = toolbox.ai_agent.build("Create a visualization of the model results")
if agent_result['success']:
    print("Generated code:")
    print(agent_result['code'])
```

---

## 🔧 **Available Features**

### **Core Features:**
- `toolbox.fit(X, y)` - Train model
- `toolbox.predict(model_id, X)` - Make predictions
- `toolbox.universal_preprocessor` - Universal preprocessor
- `toolbox.ai_orchestrator` - AI model orchestrator
- `toolbox.ai_feature_selector` - AI feature selector

### **Revolutionary Features:**
- `toolbox.predictive_intelligence` - Predicts next actions
- `toolbox.self_healing_code` - Fixes code automatically
- `toolbox.natural_language_pipeline` - Natural language to ML
- `toolbox.collaborative_intelligence` - Community learning
- `toolbox.auto_optimizer` - Auto-optimizes code

### **AI Agent:**
- `toolbox.ai_agent` - AI code generation

---

## 📝 **Command Line Usage**

### **Run Test Suite:**
```bash
python test_revolutionary_features.py
```

### **Run Comprehensive Tests:**
```bash
python comprehensive_ml_test_suite.py
```

### **Run Revolutionary IDE:**
```bash
python revolutionary_ide/revolutionary_ide.py
```

---

## 🎨 **Using the Revolutionary IDE**

```bash
# Launch the IDE
python revolutionary_ide/revolutionary_ide.py
```

**Features:**
- AI-powered code editor
- ML Toolbox integration
- Real-time error fixing
- Visual model training
- Code execution

---

## 💡 **Quick Tips**

1. **Start Simple:**
   ```python
   toolbox = MLToolbox()
   result = toolbox.fit(X, y)
   ```

2. **Use Natural Language:**
   ```python
   toolbox.natural_language_pipeline.execute_pipeline("Classify data")
   ```

3. **Let It Predict:**
   - The toolbox automatically suggests next steps
   - Check `toolbox.predictive_intelligence.get_suggestions()`

4. **Let It Heal:**
   - Code is automatically checked and fixed
   - Use `toolbox.self_healing_code.heal_code(code)`

5. **Let It Optimize:**
   - Code is automatically optimized
   - Use `toolbox.auto_optimizer.optimize_code(code)`

---

## 🚀 **Advanced Usage**

### **Using AI Orchestrator:**

```python
# AI automatically selects best model, tunes hyperparameters, creates ensemble
result = toolbox.ai_orchestrator.build_optimal_model(
    X, y,
    task_type='classification',
    time_budget=60,  # seconds
    accuracy_target=0.95
)
```

### **Using AI Feature Selector:**

```python
# AI selects best features using ensemble methods
result = toolbox.ai_feature_selector.select_features(X, y, n_features=20)
selected_features = result['selected_features']
```

### **Using Self-Improving Toolbox:**

```python
from self_improving_toolbox import get_self_improving_toolbox

# Toolbox that gets better with every use
improving_toolbox = get_self_improving_toolbox(base_toolbox=toolbox)
result = improving_toolbox.fit(X, y)

# Check improvement stats
stats = improving_toolbox.get_improvement_stats()
print(f"Success rate: {stats['success_rate']:.1%}")
```

---

## 📚 **Documentation**

- `REVOLUTIONARY_FEATURES_GUIDE.md` - Complete guide to revolutionary features
- `MINDBLOWING_REVOLUTIONARY_TOOLBOX.md` - Complete summary
- `IDE_INTEGRATION_GUIDE.md` - IDE integration guide
- `AI_AGENT_ROADMAP.md` - AI Agent documentation

---

## ✅ **That's It!**

The toolbox is ready to use. Just import and start building ML models!

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
# All features are automatically enabled and ready to use!
```

**Happy ML Building!** 🚀
