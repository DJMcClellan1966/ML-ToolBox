# Computational Kernels Guide 🚀

## Overview

The ML Toolbox now includes **Fortran-like** and **Julia-like** computational kernels that provide the performance benefits of those languages **without requiring Fortran or Julia code**.

Similar to quantum-inspired methods that provide quantum-like benefits without actual quantum hardware, these kernels provide Fortran/Julia-like performance using Python, NumPy, Numba, and BLAS/LAPACK.

---

## 🎯 **Concept**

### **Quantum-Inspired Analogy:**

Just as quantum-inspired methods provide:
- ✅ Quantum-like benefits (superposition, interference, entanglement)
- ✅ Without actual quantum hardware
- ✅ Using classical computers

These computational kernels provide:
- ✅ Fortran-like performance (vectorization, BLAS/LAPACK)
- ✅ Julia-like performance (JIT compilation, type specialization)
- ✅ Without requiring Fortran or Julia code
- ✅ Using Python, NumPy, Numba, BLAS/LAPACK

---

## 🔧 **Available Kernels**

### **1. FortranLikeKernel**

Mimics Fortran's performance characteristics:
- ✅ Optimized vectorized operations
- ✅ BLAS/LAPACK integration
- ✅ Memory-efficient processing
- ✅ Column-major operations (Fortran-style)

**Example:**
```python
from ml_toolbox.computational_kernels import FortranLikeKernel
import numpy as np

# Initialize kernel
fortran_kernel = FortranLikeKernel(use_blas=True, parallel=True)

# Fast standardization (Fortran-like vectorized)
X = np.random.randn(1000, 20)
X_std = fortran_kernel.standardize(X)

# Fast matrix multiplication (BLAS)
A = np.random.randn(100, 100)
B = np.random.randn(100, 100)
C = fortran_kernel.matrix_multiply(A, B)

# Fast pairwise distances (vectorized)
distances = fortran_kernel.pairwise_distances(X)
```

---

### **2. JuliaLikeKernel**

Mimics Julia's performance characteristics:
- ✅ JIT compilation (Numba)
- ✅ Type specialization
- ✅ Fast array operations
- ✅ Multiple dispatch concepts

**Example:**
```python
from ml_toolbox.computational_kernels import JuliaLikeKernel
import numpy as np

# Initialize kernel
julia_kernel = JuliaLikeKernel(jit_enabled=True, cache_enabled=True)

# Warmup (Julia-like first-call compilation)
julia_kernel.warmup()

# Fast standardization (JIT-compiled)
X = np.random.randn(1000, 20)
X_std = julia_kernel.standardize(X)

# Fast matrix multiplication (JIT-compiled)
A = np.random.randn(100, 100)
B = np.random.randn(100, 100)
C = julia_kernel.matrix_multiply(A, B)
```

---

### **3. UnifiedComputationalKernel** ⭐ **Recommended**

Combines both approaches for maximum performance:
- ✅ Automatically selects best method
- ✅ Fortran-like for large arrays
- ✅ Julia-like for smaller arrays
- ✅ Hybrid mode available

**Example:**
```python
from ml_toolbox.computational_kernels import UnifiedComputationalKernel
import numpy as np

# Initialize unified kernel (auto-selects best method)
kernel = UnifiedComputationalKernel(
    mode='auto',  # or 'fortran', 'julia', 'hybrid'
    use_blas=True,
    jit_enabled=True,
    parallel=True
)

# Fast operations (automatically uses best method)
X = np.random.randn(1000, 20)
X_std = kernel.standardize(X)  # Auto-selects Fortran-like (large array)
X_norm = kernel.normalize(X)    # Auto-selects Fortran-like

# Matrix operations
A = np.random.randn(100, 100)
B = np.random.randn(100, 100)
C = kernel.matrix_multiply(A, B)  # Uses BLAS (Fortran-like)

# Pairwise distances
distances = kernel.pairwise_distances(X)  # Auto-selects best method
```

---

## 🚀 **Integration with ML Toolbox**

### **Using with ML Toolbox:**

```python
from ml_toolbox import MLToolbox
from ml_toolbox.computational_kernels import UnifiedComputationalKernel
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Initialize computational kernel
comp_kernel = UnifiedComputationalKernel(mode='auto')

# Use kernel for preprocessing
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Fast preprocessing using computational kernel
X_std = comp_kernel.standardize(X)
X_norm = comp_kernel.normalize(X)

# Then use with toolbox
result = toolbox.fit(X_std, y, task_type='classification')
```

---

## 📊 **Performance Comparison**

### **Expected Performance:**

| Operation | Pure Python | Fortran-like | Julia-like | Unified |
|-----------|-------------|--------------|------------|---------|
| **Standardization** | 1.0s | 0.01s (100x) | 0.01s (100x) | 0.01s (100x) |
| **Matrix Multiply** | 1.0s | 0.005s (200x) | 0.01s (100x) | 0.005s (200x) |
| **Pairwise Distances** | 10.0s | 0.1s (100x) | 0.1s (100x) | 0.1s (100x) |

**Note:** Performance depends on data size, hardware, and BLAS availability.

---

## 🎯 **Modes**

### **Auto Mode (Recommended):**
- Automatically selects best method based on data size
- Large arrays → Fortran-like (vectorization, BLAS)
- Small arrays → Julia-like (JIT compilation)

### **Fortran Mode:**
- Always uses vectorized operations
- Best for large arrays
- Uses BLAS/LAPACK when available

### **Julia Mode:**
- Always uses JIT compilation
- Best for smaller arrays
- Type specialization

### **Hybrid Mode:**
- Combines both approaches
- Uses Fortran-like for large operations
- Uses Julia-like for smaller operations

---

## 🔧 **Configuration**

### **FortranLikeKernel:**
```python
fortran_kernel = FortranLikeKernel(
    use_blas=True,      # Use BLAS/LAPACK
    parallel=True       # Enable parallel processing
)
```

### **JuliaLikeKernel:**
```python
julia_kernel = JuliaLikeKernel(
    jit_enabled=True,   # Enable JIT compilation
    cache_enabled=True  # Cache compiled functions
)
```

### **UnifiedComputationalKernel:**
```python
unified_kernel = UnifiedComputationalKernel(
    mode='auto',        # 'auto', 'fortran', 'julia', 'hybrid'
    use_blas=True,      # Use BLAS/LAPACK
    jit_enabled=True,   # Enable JIT compilation
    parallel=True       # Enable parallel processing
)
```

---

## 📝 **Available Operations**

### **All Kernels Support:**
- `standardize(X)` - Fast standardization
- `normalize(X)` - Fast min-max normalization
- `matrix_multiply(A, B)` - Fast matrix multiplication
- `pairwise_distances(X, metric='euclidean')` - Fast pairwise distances

### **Fortran-like Only:**
- `solve_linear_system(A, b)` - Solve Ax = b using LAPACK
- `eigen_decomposition(A)` - Eigenvalue decomposition using LAPACK
- `svd(A)` - Singular Value Decomposition using LAPACK
- `batch_process(X, func, batch_size)` - Batch processing
- `vectorized_operation(X, operation)` - Vectorized operations

---

## 🎯 **Best Practices**

### **1. Use Unified Kernel:**
```python
# Recommended: Auto-selects best method
kernel = UnifiedComputationalKernel(mode='auto')
```

### **2. Warmup Julia Kernel:**
```python
# Warmup to avoid first-call delay
julia_kernel.warmup()
```

### **3. Use for Large Arrays:**
```python
# Computational kernels shine with large arrays
if X.size > 10000:
    X_std = kernel.standardize(X)
else:
    # Small arrays: regular NumPy is fine
    X_std = (X - X.mean()) / X.std()
```

### **4. Check Performance Info:**
```python
# Get performance information
info = kernel.get_performance_info()
print(info)
# {'mode': 'auto', 'fortran_blas': True, 'julia_jit': True, ...}
```

---

## 🔍 **How It Works**

### **Fortran-like:**
1. Uses NumPy vectorization (Fortran's strength)
2. BLAS/LAPACK for matrix operations
3. Column-major arrays (Fortran order)
4. Memory-efficient processing

### **Julia-like:**
1. Numba JIT compilation (Julia's strength)
2. Type specialization
3. Function caching
4. Parallel processing

### **Unified:**
1. Automatically selects best approach
2. Combines both methods
3. Optimizes for data size
4. Maximum performance

---

## 📈 **Performance Tips**

1. **Large Arrays:** Use Fortran-like or auto mode
2. **Small Arrays:** Use Julia-like or auto mode
3. **Matrix Operations:** Always use BLAS (Fortran-like)
4. **Repeated Operations:** Enable caching (Julia-like)
5. **Parallel Processing:** Enable for large datasets

---

## 🎯 **Summary**

### **Benefits:**
- ✅ **Fortran-like performance** without Fortran code
- ✅ **Julia-like performance** without Julia code
- ✅ **Automatic optimization** (unified kernel)
- ✅ **Easy to use** (Python API)
- ✅ **No dependencies** on Fortran/Julia

### **Use Cases:**
- Large-scale data preprocessing
- Fast matrix operations
- Pairwise distance computations
- Linear algebra operations
- Performance-critical ML operations

**These kernels provide Fortran/Julia-like performance benefits without requiring those languages, similar to how quantum-inspired methods provide quantum-like benefits without quantum hardware!** 🚀
