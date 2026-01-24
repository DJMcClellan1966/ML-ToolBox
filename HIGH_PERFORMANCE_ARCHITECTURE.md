# High-Performance Architecture Guide 🚀

## Overview

This guide explains how to structure the ML Toolbox similar to scikit-learn, with Python for the API and Cython/C/C++ for computationally intensive components.

---

## 🏗️ **scikit-learn Architecture Pattern**

### **How scikit-learn Works:**

```
┌─────────────────────────────────────────────────────────┐
│                    Python API Layer                       │
│  (User-facing, easy to use, high-level interface)         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  Cython Wrapper Layer                    │
│  (Type checking, memory management, Python integration) │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              C/C++ Computational Core                    │
│  (Fast algorithms, optimized loops, BLAS/LAPACK)         │
└─────────────────────────────────────────────────────────┘
```

**Key Principles:**
1. **Python** - User API, high-level logic, easy to use
2. **Cython** - Type checking, memory management, Python integration
3. **C/C++** - Computational core, optimized algorithms, BLAS/LAPACK

---

## 🎯 **ML Toolbox Architecture Plan**

### **Proposed Structure:**

```
ml_toolbox/
├── __init__.py                    # Python API (user-facing)
├── compartment1_data/
│   ├── __init__.py                # Python API
│   ├── preprocessors.py           # Python wrapper
│   ├── _preprocessors.pyx         # Cython implementation
│   └── _preprocessors_core.c      # C computational core
├── compartment3_algorithms/
│   ├── __init__.py                # Python API
│   ├── classifiers.py             # Python wrapper
│   ├── _classifiers.pyx           # Cython implementation
│   └── _classifiers_core.cpp      # C++ computational core
└── ...
```

---

## 🔧 **Language Options for Computational Code**

### **0. Fortran & Julia (Not Recommended for Primary Use)**

**Fortran:**
- ✅ Excellent numerical performance (often faster than C for pure math)
- ❌ Complex Python integration (f2py)
- ❌ Slower development, harder maintenance
- ❌ Cython/C++ can match performance with better integration
- **Verdict:** Not recommended - use Cython/C++ instead

**Julia:**
- ✅ Excellent performance (often as fast as C)
- ✅ Modern language, Python-like syntax
- ⚠️ Runtime dependency (requires Julia installation)
- ⚠️ Smaller ecosystem than Python/C/C++
- ⚠️ Some overhead when calling from Python
- **Verdict:** Consider for specific use cases, but Cython is better for primary implementation

**See `FORTRAN_JULIA_ANALYSIS.md` for detailed analysis.**

---

### **1. Cython (Recommended for ML Toolbox)**

**Best For:** ML Toolbox (similar to scikit-learn)

**Pros:**
- ✅ **Python-like syntax** - Easy to learn
- ✅ **Seamless integration** - Works directly with Python
- ✅ **Type checking** - C-like performance with Python ease
- ✅ **NumPy integration** - Excellent NumPy support
- ✅ **Mature ecosystem** - Used by scikit-learn, pandas, etc.

**Cons:**
- ⚠️ **Compilation step** - Requires building
- ⚠️ **Less control** - Not as low-level as C/C++

**Performance:** 10-100x faster than pure Python

**Example:**
```cython
# _preprocessors.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def fast_standardize(cnp.ndarray[cnp.float64_t, ndim=2] X):
    """Fast standardization using Cython"""
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] X_std = X.copy()
    cdef int i, j
    cdef double mean, std
    
    for j in range(n_features):
        # Compute mean
        mean = 0.0
        for i in range(n_samples):
            mean += X[i, j]
        mean /= n_samples
        
        # Compute std
        std = 0.0
        for i in range(n_samples):
            std += (X[i, j] - mean) ** 2
        std = sqrt(std / n_samples)
        
        # Standardize
        for i in range(n_samples):
            X_std[i, j] = (X[i, j] - mean) / std
    
    return X_std
```

---

### **2. Rust (Modern Alternative)**

**Best For:** New projects, maximum performance

**Pros:**
- ✅ **Memory safety** - No segfaults, no data races
- ✅ **Excellent performance** - Often faster than C/C++
- ✅ **Modern language** - Great tooling, package management
- ✅ **Python bindings** - PyO3 for easy integration
- ✅ **Growing ecosystem** - Many ML libraries using Rust

**Cons:**
- ⚠️ **Learning curve** - Different from Python
- ⚠️ **Less mature** - Newer than Cython for Python ML

**Performance:** Often faster than C/C++, 50-200x faster than Python

**Example:**
```rust
// preprocessors.rs
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

#[pymodule]
fn preprocessors(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn fast_standardize<'py>(
        py: Python<'py>,
        x: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        
        let mut x_std = x.to_owned();
        
        for j in 0..n_features {
            // Compute mean
            let mean: f64 = (0..n_samples)
                .map(|i| x[[i, j]])
                .sum::<f64>() / n_samples as f64;
            
            // Compute std
            let variance: f64 = (0..n_samples)
                .map(|i| (x[[i, j]] - mean).powi(2))
                .sum::<f64>() / n_samples as f64;
            let std = variance.sqrt();
            
            // Standardize
            for i in 0..n_samples {
                x_std[[i, j]] = (x[[i, j]] - mean) / std;
            }
        }
        
        Ok(PyArray2::from_array(py, &x_std))
    }
    
    Ok(())
}
```

---

### **3. C/C++ (Maximum Control)**

**Best For:** Maximum performance, existing C/C++ code

**Pros:**
- ✅ **Maximum performance** - Full control, no overhead
- ✅ **BLAS/LAPACK** - Direct access to optimized libraries
- ✅ **Mature** - Decades of optimization
- ✅ **Industry standard** - Used by scikit-learn, NumPy

**Cons:**
- ⚠️ **Complex** - Memory management, compilation
- ⚠️ **Error-prone** - Easy to introduce bugs
- ⚠️ **Integration** - More complex Python bindings

**Performance:** Fastest possible, 50-200x faster than Python

**Example:**
```c
// preprocessors_core.c
#include <math.h>
#include <stdlib.h>

void fast_standardize(double* X, int n_samples, int n_features, double* X_std) {
    for (int j = 0; j < n_features; j++) {
        // Compute mean
        double mean = 0.0;
        for (int i = 0; i < n_samples; i++) {
            mean += X[i * n_features + j];
        }
        mean /= n_samples;
        
        // Compute std
        double variance = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double diff = X[i * n_features + j] - mean;
            variance += diff * diff;
        }
        double std = sqrt(variance / n_samples);
        
        // Standardize
        for (int i = 0; i < n_samples; i++) {
            X_std[i * n_features + j] = (X[i * n_features + j] - mean) / std;
        }
    }
}
```

---

### **4. Numba JIT (Easy Option)**

**Best For:** Quick wins, existing Python code

**Pros:**
- ✅ **Easy** - Just add decorator to Python code
- ✅ **No compilation** - JIT compilation at runtime
- ✅ **NumPy optimized** - Excellent NumPy support
- ✅ **Python syntax** - Write in Python

**Cons:**
- ⚠️ **Limited features** - Not all Python features supported
- ⚠️ **Warmup time** - First call slower (compilation)
- ⚠️ **Less control** - Can't optimize as much as Cython

**Performance:** 10-50x faster than pure Python

**Example:**
```python
# preprocessors.py
from numba import jit
import numpy as np

@jit(nopython=True)
def fast_standardize(X):
    """Fast standardization using Numba JIT"""
    n_samples, n_features = X.shape
    X_std = X.copy()
    
    for j in range(n_features):
        # Compute mean
        mean = 0.0
        for i in range(n_samples):
            mean += X[i, j]
        mean /= n_samples
        
        # Compute std
        variance = 0.0
        for i in range(n_samples):
            diff = X[i, j] - mean
            variance += diff * diff
        std = np.sqrt(variance / n_samples)
        
        # Standardize
        for i in range(n_samples):
            X_std[i, j] = (X[i, j] - mean) / std
    
    return X_std
```

---

## 📊 **Language Comparison**

| Language | Performance | Ease of Use | Integration | Best For |
|----------|-------------|-------------|-------------|----------|
| **Cython** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **ML Toolbox (recommended)** |
| **Rust** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | New projects, maximum safety |
| **C/C++** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Maximum performance, existing code |
| **Numba** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Quick wins, existing Python |

---

## 🎯 **Recommended Approach for ML Toolbox**

### **Hybrid Strategy:**

1. **Start with Numba** (Quick wins)
   - Add `@jit` decorators to hot paths
   - Easy to implement
   - Immediate performance gains

2. **Migrate to Cython** (Long-term)
   - Similar to scikit-learn
   - Better performance than Numba
   - More control

3. **Use Rust for New Components** (Future)
   - New algorithms in Rust
   - Maximum performance and safety
   - Modern tooling

---

## 🚀 **Implementation Plan**

### **Phase 1: Numba JIT (Immediate - 1-2 weeks)**

**Goal:** Quick performance wins with minimal changes

**Steps:**
1. Identify hot paths (loops, numerical operations)
2. Add `@jit` decorators
3. Test performance improvements
4. Measure impact

**Expected Improvement:** 10-50x faster on hot paths

**Example:**
```python
# Add to existing code
from numba import jit

@jit(nopython=True)
def compute_similarity_matrix(embeddings):
    # Existing Python code, now JIT compiled
    ...
```

---

### **Phase 2: Cython Migration (Short-term - 1-2 months)**

**Goal:** scikit-learn-like architecture

**Steps:**
1. Create `.pyx` files for hot paths
2. Add type annotations
3. Build with Cython
4. Replace Python implementations

**Expected Improvement:** 50-100x faster

**Structure:**
```
ml_toolbox/
├── compartment1_data/
│   ├── preprocessors.py          # Python API
│   └── _preprocessors.pyx        # Cython implementation
├── compartment3_algorithms/
│   ├── classifiers.py            # Python API
│   └── _classifiers.pyx           # Cython implementation
└── setup.py                       # Build configuration
```

---

### **Phase 3: Rust Components (Long-term - 3-6 months)**

**Goal:** Maximum performance for new components

**Steps:**
1. Identify new algorithms/components
2. Implement in Rust
3. Create Python bindings (PyO3)
4. Integrate with ML Toolbox

**Expected Improvement:** 100-200x faster

---

## 📝 **Example: Cython Implementation**

### **Step 1: Create Cython File**

```cython
# ml_toolbox/compartment1_data/_preprocessors.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs

def fast_standardize(cnp.ndarray[cnp.float64_t, ndim=2] X):
    """Fast standardization - Cython version"""
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] X_std = np.empty_like(X)
    
    cdef int i, j
    cdef double mean, std, variance, diff
    
    for j in range(n_features):
        # Compute mean
        mean = 0.0
        for i in range(n_samples):
            mean += X[i, j]
        mean /= n_samples
        
        # Compute std
        variance = 0.0
        for i in range(n_samples):
            diff = X[i, j] - mean
            variance += diff * diff
        std = sqrt(variance / n_samples)
        
        # Standardize (avoid division by zero)
        if std > 1e-10:
            for i in range(n_samples):
                X_std[i, j] = (X[i, j] - mean) / std
        else:
            for i in range(n_samples):
                X_std[i, j] = 0.0
    
    return X_std
```

### **Step 2: Create Python Wrapper**

```python
# ml_toolbox/compartment1_data/preprocessors.py
try:
    from ._preprocessors import fast_standardize as _fast_standardize_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    # Fallback to Python implementation
    def _fast_standardize_python(X):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)

def standardize(X, use_cython=True):
    """Standardize data - uses Cython if available"""
    if use_cython and CYTHON_AVAILABLE:
        return _fast_standardize_cython(X)
    else:
        return _fast_standardize_python(X)
```

### **Step 3: Setup Configuration**

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "ml_toolbox.compartment1_data._preprocessors",
        ["ml_toolbox/compartment1_data/_preprocessors.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3', '-march=native']  # Optimize
    ),
    Extension(
        "ml_toolbox.compartment3_algorithms._classifiers",
        ["ml_toolbox/compartment3_algorithms/_classifiers.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3', '-march=native']
    ),
]

setup(
    name='ml_toolbox',
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
```

---

## 🔧 **Build Instructions**

### **Build Cython Extensions:**

```bash
# Install Cython
pip install Cython numpy

# Build extensions
python setup.py build_ext --inplace

# Or use pip
pip install -e .
```

---

## 📊 **Performance Expectations**

### **Before (Pure Python):**
- Standardization: ~1.0s for 100K samples
- Classification: ~5.0s for 10K samples
- Clustering: ~10.0s for 10K samples

### **After (Cython):**
- Standardization: ~0.01s (100x faster)
- Classification: ~0.05s (100x faster)
- Clustering: ~0.1s (100x faster)

### **After (Rust):**
- Standardization: ~0.005s (200x faster)
- Classification: ~0.02s (250x faster)
- Clustering: ~0.05s (200x faster)

---

## 🎯 **Priority Components for Optimization**

### **High Priority (Most Impact):**

1. **Data Preprocessing**
   - Standardization/normalization
   - Missing value imputation
   - Feature engineering

2. **Distance/Similarity Computations**
   - Euclidean distance
   - Cosine similarity
   - Matrix operations

3. **Model Training Loops**
   - Gradient descent
   - Tree building
   - Clustering algorithms

4. **Matrix Operations**
   - Matrix multiplication
   - Eigenvalue decomposition
   - SVD

---

## 🚀 **Quick Start: Numba Implementation**

### **Immediate Performance Gains:**

```python
# ml_toolbox/compartment1_data/preprocessors.py
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def fast_standardize_numba(X):
    """Fast standardization with Numba"""
    n_samples, n_features = X.shape
    X_std = np.empty_like(X)
    
    for j in range(n_features):
        mean = np.mean(X[:, j])
        std = np.std(X[:, j])
        if std > 1e-10:
            X_std[:, j] = (X[:, j] - mean) / std
        else:
            X_std[:, j] = 0.0
    
    return X_std

# Use in preprocessor
def standardize(X, method='numba'):
    if method == 'numba':
        return fast_standardize_numba(X)
    else:
        # Fallback to sklearn
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)
```

**Installation:**
```bash
pip install numba
```

**Expected Improvement:** 10-50x faster immediately!

---

## 📈 **Migration Strategy**

### **Step-by-Step Migration:**

1. **Identify Hot Paths**
   ```python
   # Profile code to find bottlenecks
   import cProfile
   cProfile.run('your_function()')
   ```

2. **Start with Numba**
   - Add `@jit` to hot paths
   - Test performance
   - Verify correctness

3. **Migrate to Cython**
   - Convert Numba functions to Cython
   - Add type annotations
   - Build and test

4. **Optimize Further**
   - Use BLAS/LAPACK for matrix ops
   - Parallel processing
   - SIMD optimizations

---

## 🔍 **Best Practices**

### **1. Maintain Python API**

```python
# Always keep Python API simple
def standardize(X):
    """Simple Python API"""
    if CYTHON_AVAILABLE:
        return _fast_standardize_cython(X)
    elif NUMBA_AVAILABLE:
        return _fast_standardize_numba(X)
    else:
        return _fast_standardize_python(X)
```

### **2. Graceful Fallbacks**

```python
# Always have Python fallback
try:
    from ._preprocessors import fast_standardize
    FAST_AVAILABLE = True
except ImportError:
    FAST_AVAILABLE = False
    # Fallback to Python
    def fast_standardize(X):
        return StandardScaler().fit_transform(X)
```

### **3. Type Safety**

```cython
# Use type annotations in Cython
cdef double compute_mean(double[:] X):
    cdef double sum = 0.0
    cdef int i
    for i in range(X.shape[0]):
        sum += X[i]
    return sum / X.shape[0]
```

---

## 🎯 **Recommended Languages by Use Case**

### **For ML Toolbox:**

1. **Cython** ⭐⭐⭐⭐⭐ (Recommended)
   - Similar to scikit-learn
   - Easy Python integration
   - Good performance
   - Mature ecosystem

2. **Numba** ⭐⭐⭐⭐ (Quick wins)
   - Easy to add
   - No compilation needed
   - Good for prototyping

3. **Rust** ⭐⭐⭐ (Future)
   - Maximum performance
   - Memory safety
   - Modern tooling

4. **C/C++** ⭐⭐ (If needed)
   - Maximum control
   - BLAS/LAPACK integration
   - More complex

---

## 📝 **Summary**

### **Recommended Approach:**

1. **Immediate:** Add Numba JIT to hot paths (10-50x faster)
2. **Short-term:** Migrate to Cython (50-100x faster)
3. **Long-term:** Use Rust for new components (100-200x faster)

### **Architecture:**

```
Python API (User-facing)
    ↓
Cython Wrapper (Type checking, integration)
    ↓
C/C++/Rust Core (Computational algorithms)
```

### **Benefits:**

- ✅ **10-200x faster** performance
- ✅ **scikit-learn-like** architecture
- ✅ **Maintains Python API** (easy to use)
- ✅ **Graceful fallbacks** (works without compilation)

**This approach will make ML Toolbox competitive with scikit-learn in terms of performance!** 🚀
