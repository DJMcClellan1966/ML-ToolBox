# Compartment Kernels Guide 🧠

## Overview

Each compartment can now be used as a **unified algorithm or kernel** with a simple, consistent interface. This allows you to treat each compartment as a single callable function/algorithm.

---

## 🎯 **Concept**

Instead of accessing compartments through multiple methods, each compartment is now a **unified kernel** with:
- `fit()` - Train/configure the kernel
- `transform()` - Process data through the kernel
- `process()` - Complete processing with metadata
- `__call__()` - Make kernel directly callable

---

## 📦 **Available Kernels**

### **1. DataKernel** - Compartment 1 as Algorithm

Treats the entire data preprocessing compartment as a single algorithm.

```python
from ml_toolbox import MLToolbox
from ml_toolbox.compartment_kernels import DataKernel

toolbox = MLToolbox()

# Create data kernel
data_kernel = DataKernel(toolbox.data)

# Use as algorithm
result = data_kernel.fit(X_train).transform(X_test)

# Or use process() for complete pipeline
result = data_kernel.process(X_train)
# Returns: {
#   'processed_data': preprocessed array,
#   'quality_score': 0.95,
#   'metadata': {...}
# }

# Or call directly
result = data_kernel(X_train)
```

**Features:**
- ✅ Auto-detects best preprocessor
- ✅ Handles all preprocessing automatically
- ✅ Quality scoring
- ✅ Returns metadata

---

### **2. InfrastructureKernel** - Compartment 2 as Algorithm

Treats the entire infrastructure compartment as a single kernel.

```python
from ml_toolbox.compartment_kernels import InfrastructureKernel

# Create infrastructure kernel
infra_kernel = InfrastructureKernel(toolbox.infrastructure)

# Fit on corpus
infra_kernel.fit(documents)

# Transform (semantic operations)
embeddings = infra_kernel.transform(query, operation='embed')
understanding = infra_kernel.transform(query, operation='understand')
search_results = infra_kernel.transform(query, operation='search', corpus=documents)

# Or use process() for complete pipeline
result = infra_kernel.process(query)
# Returns: {
#   'embeddings': semantic embeddings,
#   'understanding': intent/meaning,
#   'reasoning': logical reasoning,
#   'metadata': {...}
# }
```

**Features:**
- ✅ Semantic embeddings
- ✅ Intent understanding
- ✅ Logical reasoning
- ✅ Knowledge graph building

---

### **3. AlgorithmsKernel** - Compartment 3 as Algorithm

Treats the entire algorithms compartment as a single ML algorithm.

```python
from ml_toolbox.compartment_kernels import AlgorithmsKernel

# Create algorithms kernel
algo_kernel = AlgorithmsKernel(toolbox.algorithms)

# Fit (train model)
algo_kernel.fit(X_train, y_train, task_type='classification')

# Transform (predictions)
predictions = algo_kernel.transform(X_test)

# Or use process() for complete pipeline
result = algo_kernel.process(X_train, y_train, task_type='classification')
# Returns: {
#   'model': trained model,
#   'predictions': predictions,
#   'metrics': performance metrics,
#   'metadata': {...}
# }
```

**Features:**
- ✅ Auto-selects best algorithm
- ✅ Automatic hyperparameter optimization
- ✅ Ensemble support
- ✅ Performance metrics

---

### **4. MLOpsKernel** - Compartment 4 as Algorithm

Treats the entire MLOps compartment as a single deployment kernel.

```python
from ml_toolbox.compartment_kernels import MLOpsKernel

# Create MLOps kernel
mlops_kernel = MLOpsKernel(toolbox.mlops)

# Fit (set up infrastructure)
mlops_kernel.fit(X_train)

# Transform (deploy/monitor)
deployment = mlops_kernel.transform(
    X_test,
    operation='deploy',
    model=trained_model
)

# Or use process() for complete pipeline
result = mlops_kernel.process(X_test, model=trained_model)
# Returns: {
#   'deployment': deployment info,
#   'monitoring': monitoring setup,
#   'experiment_id': experiment ID,
#   'metadata': {...}
# }
```

**Features:**
- ✅ Model deployment
- ✅ Performance monitoring
- ✅ Experiment tracking
- ✅ A/B testing

---

## 🔗 **Unified Kernel - All Compartments as One**

Combine all compartments into a **single unified kernel**:

```python
from ml_toolbox.compartment_kernels import get_unified_kernel

# Create unified kernel
unified = get_unified_kernel(toolbox)

# Fit all compartments
unified.fit(X_train, y_train, task_type='classification')

# Transform through all compartments
results = unified.transform(X_test)
# Returns: {
#   'data': {...},           # Preprocessing results
#   'infrastructure': {...}, # Semantic understanding
#   'algorithms': {...},     # Model predictions
#   'mlops': {...}           # Deployment/monitoring
# }

# Or call directly
results = unified(X_train, y_train, task_type='classification')
```

---

## 📊 **Usage Examples**

### **Example 1: Data Kernel Only**

```python
from ml_toolbox import MLToolbox
from ml_toolbox.compartment_kernels import DataKernel

toolbox = MLToolbox()
data_kernel = DataKernel(toolbox.data)

# Process raw data
raw_data = load_data()
result = data_kernel.process(raw_data)

print(f"Processed shape: {result['processed_data'].shape}")
print(f"Quality score: {result['quality_score']}")
print(f"Preprocessor: {result['metadata']['preprocessor_type']}")
```

### **Example 2: Infrastructure Kernel for Semantic Search**

```python
from ml_toolbox.compartment_kernels import InfrastructureKernel

infra_kernel = InfrastructureKernel(toolbox.infrastructure)

# Build knowledge base
documents = ["doc1", "doc2", "doc3", ...]
infra_kernel.fit(documents)

# Semantic search
query = "machine learning"
results = infra_kernel.transform(
    query,
    operation='search',
    corpus=documents
)

print(f"Found {len(results['search'])} relevant documents")
```

### **Example 3: Algorithms Kernel for ML**

```python
from ml_toolbox.compartment_kernels import AlgorithmsKernel

algo_kernel = AlgorithmsKernel(toolbox.algorithms)

# Train and predict
result = algo_kernel.process(
    X_train, y_train,
    task_type='classification'
)

print(f"Model: {result['metadata']['model_type']}")
print(f"Accuracy: {result['metrics']['accuracy']}")
print(f"Predictions: {result['predictions']}")
```

### **Example 4: Complete Pipeline with Unified Kernel**

```python
from ml_toolbox.compartment_kernels import get_unified_kernel

# Create unified kernel
unified = get_unified_kernel(toolbox)

# Complete ML pipeline in one call
results = unified(X_train, y_train, X_test, task_type='classification')

# Access results from each compartment
preprocessed = results['data']['processed_data']
embeddings = results['infrastructure']['embeddings']
predictions = results['algorithms']['predictions']
deployment = results['mlops']['deployment']

print(f"Pipeline complete!")
print(f"Predictions: {predictions}")
print(f"Deployed: {deployment is not None}")
```

---

## 🔄 **Kernel Composition**

You can compose kernels together:

```python
# Sequential composition
data_result = data_kernel.process(X)
infra_result = infrastructure_kernel.process(data_result['processed_data'])
algo_result = algorithms_kernel.process(infra_result['embeddings'], y)

# Or use unified kernel for automatic composition
unified = get_unified_kernel(toolbox)
results = unified(X, y)
```

---

## 🎯 **Benefits**

### **1. Simple Interface**
- Each compartment is now a single callable algorithm
- Consistent `fit()` / `transform()` / `process()` interface
- No need to know internal compartment structure

### **2. Modularity**
- Use compartments independently
- Compose kernels as needed
- Mix and match functionality

### **3. Unified API**
- All kernels follow same interface
- Easy to swap implementations
- Consistent error handling

### **4. Metadata & Transparency**
- Each kernel returns metadata
- Understand what happened at each step
- Debug and optimize easily

---

## 📝 **Configuration**

Each kernel can be configured:

```python
# Configure data kernel
data_kernel = DataKernel(
    toolbox.data,
    config={
        'auto_detect': True,
        'use_advanced': True,
        'use_universal': True,
        'quality_check': True
    }
)

# Configure infrastructure kernel
infra_kernel = InfrastructureKernel(
    toolbox.infrastructure,
    config={
        'use_quantum': True,
        'use_ai_system': True,
        'use_reasoning': True
    }
)

# Configure algorithms kernel
algo_kernel = AlgorithmsKernel(
    toolbox.algorithms,
    config={
        'auto_select': True,
        'use_ensemble': True,
        'optimize_hyperparameters': True
    }
)

# Configure unified kernel
unified = get_unified_kernel(
    toolbox,
    config={
        'data': {...},
        'infrastructure': {...},
        'algorithms': {...},
        'mlops': {...}
    }
)
```

---

## 🚀 **Quick Start**

```python
from ml_toolbox import MLToolbox
from ml_toolbox.compartment_kernels import (
    DataKernel,
    InfrastructureKernel,
    AlgorithmsKernel,
    get_unified_kernel
)

# Initialize toolbox
toolbox = MLToolbox()

# Option 1: Use individual kernels
data_kernel = DataKernel(toolbox.data)
infra_kernel = InfrastructureKernel(toolbox.infrastructure)
algo_kernel = AlgorithmsKernel(toolbox.algorithms)

# Option 2: Use unified kernel (all compartments)
unified = get_unified_kernel(toolbox)

# Use as algorithms
result = unified(X_train, y_train, X_test, task_type='classification')
```

---

## 🎯 **Summary**

**Each compartment is now a unified algorithm/kernel:**

1. **DataKernel** - Preprocessing as algorithm
2. **InfrastructureKernel** - AI/semantic operations as algorithm
3. **AlgorithmsKernel** - ML models as algorithm
4. **MLOpsKernel** - Deployment/monitoring as algorithm
5. **UnifiedCompartmentKernel** - All compartments as one algorithm

**Benefits:**
- ✅ Simple, consistent interface
- ✅ Modular and composable
- ✅ Easy to use and understand
- ✅ Metadata and transparency

**Use cases:**
- Quick prototyping
- Pipeline composition
- Modular ML systems
- Agent-based workflows

**The compartments are now algorithms you can call directly!** 🚀
