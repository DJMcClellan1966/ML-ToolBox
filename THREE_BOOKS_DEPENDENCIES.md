# Three Books Methods: Dependencies and Installation Guide

## Overview

This document lists all dependencies needed for methods from three influential ML books:

1. **Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)
2. **Pattern Recognition and Machine Learning** (Bishop)
3. **Deep Learning** (Goodfellow, Bengio, Courville)

---

## 📚 Book 1: Elements of Statistical Learning

### **Key Methods Implemented:**
- Support Vector Machines (SVM)
- AdaBoost
- Generalized Additive Models (GAM)
- Lasso, Ridge, Elastic Net (already in toolbox)
- Random Forests (already in toolbox)
- Gradient Boosting (already in toolbox)

### **Dependencies Required:**

```bash
# Core dependencies
pip install scikit-learn>=1.5.0
pip install scipy>=1.11.0
pip install numpy>=1.26.0

# Optional (for full GAM)
pip install pygam>=0.9.0
```

### **Code Dependencies:**
```python
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
```

---

## 📚 Book 2: Pattern Recognition and Machine Learning

### **Key Methods Implemented:**
- Gaussian Processes
- Gaussian Mixture Models (GMM)
- Expectation-Maximization (EM) Algorithm
- Probabilistic PCA

### **Dependencies Required:**

```bash
# Core dependencies
pip install scikit-learn>=1.5.0
pip install scipy>=1.11.0
pip install numpy>=1.26.0

# Optional (for advanced probabilistic methods)
pip install pymc3>=5.0.0  # For Bayesian methods
pip install pyro-ppl>=1.8.0  # For probabilistic programming
```

### **Code Dependencies:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy import stats
```

---

## 📚 Book 3: Deep Learning

### **Key Methods Implemented:**
- Basic Neural Networks (Multi-layer Perceptron)
- Dropout Regularization
- Batch Normalization
- Optimization Algorithms (Adam, RMSprop - via PyTorch)

### **Dependencies Required:**

```bash
# Core dependencies
pip install torch>=2.3.0
pip install numpy>=1.26.0

# Optional (for advanced features)
pip install torchvision>=0.18.0  # For computer vision
pip install torchaudio>=2.3.0  # For audio processing
```

### **Code Dependencies:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout, BatchNorm1d, ReLU, Sigmoid, Tanh
```

---

## 📦 Complete Dependency List

### **All Methods Combined:**

```bash
# Essential dependencies
pip install scikit-learn>=1.5.0
pip install scipy>=1.11.0
pip install numpy>=1.26.0
pip install torch>=2.3.0

# Optional but recommended
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### **For Full ML Toolbox (Including All Previous Methods):**

```bash
# Core ML libraries
pip install scikit-learn>=1.5.0
pip install scipy>=1.11.0
pip install numpy>=1.26.0
pip install pandas>=2.0.0

# Deep Learning
pip install torch>=2.3.0

# Interpretability
pip install shap>=0.42.0
pip install lime>=0.2.0

# Time Series
pip install statsmodels>=0.14.0

# Probabilistic Models
pip install pgmpy>=0.1.19
pip install hmmlearn>=0.2.7

# Imbalanced Learning
pip install imbalanced-learn>=0.11.0

# Optimization
pip install scikit-optimize>=0.9.0

# Visualization
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

---

## 🔧 Installation Script

### **Quick Install (All Dependencies):**

```bash
# Create requirements file
cat > requirements_three_books.txt << EOF
# Core ML
scikit-learn>=1.5.0
scipy>=1.11.0
numpy>=1.26.0
pandas>=2.0.0

# Deep Learning
torch>=2.3.0

# Interpretability
shap>=0.42.0
lime>=0.2.0

# Time Series
statsmodels>=0.14.0

# Probabilistic Models
pgmpy>=0.1.19
hmmlearn>=0.2.7

# Imbalanced Learning
imbalanced-learn>=0.11.0

# Optimization
scikit-optimize>=0.9.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
EOF

# Install
pip install -r requirements_three_books.txt
```

---

## 📋 Dependency Summary by Method

### **ESL Methods:**
- **SVM**: `sklearn`
- **AdaBoost**: `sklearn`
- **GAM**: `sklearn`, `scipy` (full: `pygam`)

### **Bishop Methods:**
- **Gaussian Processes**: `sklearn` (Gaussian Process module)
- **GMM**: `sklearn`
- **EM Algorithm**: `sklearn` (via GMM)
- **Probabilistic PCA**: `sklearn`

### **Deep Learning Methods:**
- **Neural Networks**: `torch` or `sklearn`
- **Dropout**: `torch`
- **Batch Normalization**: `torch`

---

## ⚠️ Platform-Specific Notes

### **Windows:**
- PyTorch: May need to install from pytorch.org for CUDA support
- Some packages may require Visual C++ Build Tools

### **Linux:**
- Most packages install cleanly via pip
- May need system libraries: `sudo apt-get install python3-dev`

### **macOS:**
- Most packages install cleanly via pip
- May need Xcode Command Line Tools: `xcode-select --install`

---

## 🧪 Testing Dependencies

### **Check Installation:**
```python
# Test script
import sys

dependencies = {
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'numpy': 'numpy',
    'torch': 'torch',
    'pandas': 'pandas',
    'shap': 'shap',
    'lime': 'lime',
    'statsmodels': 'statsmodels',
    'pgmpy': 'pgmpy',
    'hmmlearn': 'hmmlearn',
    'imbalanced-learn': 'imbalanced-learn',
    'scikit-optimize': 'scikit-optimize',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

print("Checking dependencies...")
for module_name, package_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
```

---

## 📊 Dependency Size Estimates

### **Approximate Installation Sizes:**

- **scikit-learn**: ~15 MB
- **scipy**: ~30 MB
- **numpy**: ~20 MB
- **pandas**: ~50 MB
- **torch**: ~500 MB (CPU) / ~2 GB (CUDA)
- **shap**: ~10 MB
- **lime**: ~5 MB
- **statsmodels**: ~20 MB
- **pgmpy**: ~5 MB
- **hmmlearn**: ~2 MB
- **imbalanced-learn**: ~2 MB
- **scikit-optimize**: ~5 MB
- **matplotlib**: ~30 MB
- **seaborn**: ~5 MB

**Total (CPU PyTorch)**: ~720 MB
**Total (CUDA PyTorch)**: ~2.2 GB

---

## 🎯 Minimal Installation (Core Only)

If you only need specific methods:

### **ESL Methods Only:**
```bash
pip install scikit-learn scipy numpy
```

### **Bishop Methods Only:**
```bash
pip install scikit-learn scipy numpy
```

### **Deep Learning Methods Only:**
```bash
pip install torch numpy
```

---

## ✅ Verification

After installation, verify with:

```python
from three_books_methods import ThreeBooksMethods, get_all_dependencies

# Check dependencies
methods = ThreeBooksMethods()
deps = methods.get_dependencies()
print("Required dependencies:", deps)

# Get all dependencies
all_deps = get_all_dependencies()
print("\nDependencies by book:")
for book, deps in all_deps.items():
    print(f"\n{book}:")
    for dep, version in deps.items():
        print(f"  - {dep}: {version}")
```

---

## 📝 Notes

1. **PyTorch**: Large download. Consider CPU-only version if GPU not needed:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Optional Dependencies**: Some methods work with fallbacks if optional libraries aren't installed

3. **Version Compatibility**: All versions specified are minimums. Newer versions generally work

4. **Virtual Environment**: Recommended to use virtual environment:
   ```bash
   python -m venv ml_toolbox_env
   source ml_toolbox_env/bin/activate  # Linux/Mac
   ml_toolbox_env\Scripts\activate  # Windows
   pip install -r requirements_three_books.txt
   ```
