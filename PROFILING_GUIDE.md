# ML Toolbox Profiling Guide

## 🎯 **Overview**

The ML Profiling System helps identify bottlenecks and optimization opportunities in your ML pipelines. It provides comprehensive performance analysis to focus optimization efforts where they matter most.

---

## 🚀 **Quick Start**

### **Basic Usage**

```python
from ml_profiler import MLProfiler

# Create profiler
profiler = MLProfiler()

# Profile a function
@profiler.profile_function
def my_ml_function(data):
    # Your ML code here
    return result

# Run function
result = my_ml_function(data)

# Generate report
report = profiler.generate_report('profiling_report.txt')
print(report)

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks()
for bottleneck in bottlenecks:
    print(f"{bottleneck['function']}: {bottleneck['total_time']:.4f}s")
```

### **Profile Entire Pipeline**

```python
from ml_profiler import MLProfiler

profiler = MLProfiler()

# Profile a pipeline
with profiler.profile_pipeline('data_preprocessing'):
    # Step 1
    cleaned_data = clean_data(raw_data)
    
    # Step 2
    transformed_data = transform_data(cleaned_data)
    
    # Step 3
    features = engineer_features(transformed_data)

# Generate report
report = profiler.generate_report()
```

---

## 📊 **Features**

### **1. Function-Level Profiling**

Profile individual functions with decorators:

```python
@profiler.profile_function
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
```

**Tracks:**
- Execution time (total, mean, min, max, std, percentiles)
- Call count
- Memory usage (optional)
- Error tracking

### **2. Pipeline Profiling**

Profile entire pipelines with context managers:

```python
with profiler.profile_pipeline('full_ml_pipeline'):
    data = load_data()
    preprocessed = preprocess(data)
    model = train(preprocessed)
    predictions = predict(model, preprocessed)
```

**Tracks:**
- Total pipeline time
- Individual step times
- Step-by-step breakdown

### **3. cProfile Integration**

Detailed profiling with Python's cProfile:

```python
result = profiler.profile_with_cprofile(my_function, arg1, arg2)
print(result['stats'])  # Detailed cProfile output
```

### **4. Bottleneck Identification**

Automatically identify slow functions:

```python
bottlenecks = profiler.identify_bottlenecks(threshold_percentile=95)

for bottleneck in bottlenecks:
    print(f"Function: {bottleneck['function']}")
    print(f"Time: {bottleneck['total_time']:.4f}s ({bottleneck['percentage']:.2f}%)")
    print(f"Priority: {bottleneck['priority']}")
    print("Recommendations:")
    for rec in bottleneck['recommendations']:
        print(f"  • {rec}")
```

### **5. Statistical Analysis**

Get detailed statistics:

```python
stats = profiler.get_function_statistics('my_function')

print(f"Total Time: {stats['total_time']:.4f}s")
print(f"Mean Time: {stats['mean_time']:.6f}s")
print(f"Median Time: {stats['median_time']:.6f}s")
print(f"P95 Time: {stats['p95_time']:.6f}s")
print(f"Call Count: {stats['call_count']:,}")
```

---

## 📈 **Report Generation**

### **Comprehensive Reports**

Generate detailed profiling reports:

```python
report = profiler.generate_report('my_report.txt')
```

**Report Includes:**
- Overall statistics
- Top 10 slowest functions
- Bottleneck analysis with priorities
- Pipeline execution breakdown
- Memory usage analysis (if enabled)
- Optimization recommendations

### **Export Data**

Export profiling data to JSON:

```python
profiler.export_data('profiling_data.json')
```

---

## 🔍 **Profiling ML Toolbox Components**

### **Profile All Components**

Run comprehensive profiling:

```bash
python profile_ml_toolbox.py
```

This profiles:
- Data preprocessing
- Model training
- Feature selection
- Ensemble learning
- Full ML pipeline

### **Profile Specific Component**

```python
from profile_ml_toolbox import profile_data_preprocessing

profiler = profile_data_preprocessing()
report = profiler.generate_report()
print(report)
```

---

## 💡 **Best Practices**

### **1. Profile Before Optimizing**

Always profile first to identify actual bottlenecks:

```python
# Don't guess - profile!
profiler = MLProfiler()

@profiler.profile_function
def slow_function():
    # Your code
    pass

# Run multiple times
for _ in range(100):
    slow_function()

# Check results
bottlenecks = profiler.identify_bottlenecks()
```

### **2. Focus on High-Impact Functions**

Prioritize optimization based on:
- **Total time** (not just mean time)
- **Call count** (frequently called functions)
- **Percentage of total time**

```python
bottlenecks = profiler.identify_bottlenecks()

# Focus on high priority
high_priority = [b for b in bottlenecks if b['priority'] == 'high']
```

### **3. Profile Real Workloads**

Use realistic data sizes and scenarios:

```python
# Use production-like data
real_data = load_production_data()
profiler = MLProfiler()

@profiler.profile_function
def process_data(data):
    return preprocess(data)

result = process_data(real_data)
```

### **4. Compare Before/After**

Profile before and after optimizations:

```python
# Before optimization
profiler_before = MLProfiler()
# ... profile ...

# After optimization
profiler_after = MLProfiler()
# ... profile ...

# Compare
stats_before = profiler_before.get_function_statistics()
stats_after = profiler_after.get_function_statistics()
```

### **5. Use Memory Profiling**

Enable memory profiling for memory-intensive operations:

```python
profiler = MLProfiler(enable_memory_profiling=True)

@profiler.profile_function
def memory_intensive_function():
    large_data = create_large_dataset()
    return process(large_data)
```

---

## 🎯 **Optimization Recommendations**

The profiler automatically generates recommendations:

### **High Call Count**
- **Recommendation:** Consider caching or memoization
- **Example:** Cache preprocessed data

### **High Variance**
- **Recommendation:** Check for conditional logic or data-dependent operations
- **Example:** Optimize conditional branches

### **Slow Execution**
- **Recommendation:** Consider optimization or parallelization
- **Example:** Use vectorized operations (NumPy/Pandas)

### **High Memory Usage**
- **Recommendation:** Consider memory-efficient algorithms
- **Example:** Use generators instead of lists

### **Training Functions**
- **Recommendation:** Consider early stopping, batch processing, or model simplification
- **Example:** Use early stopping in training loops

### **Prediction Functions**
- **Recommendation:** Consider batch prediction or model optimization
- **Example:** Batch predictions instead of one-by-one

### **Preprocessing Functions**
- **Recommendation:** Consider caching transformed data
- **Example:** Cache preprocessed features

---

## 📊 **Example Output**

### **Profiling Report**

```
================================================================================
ML TOOLBOX PROFILING REPORT
================================================================================

Generated: 2024-01-20 15:30:00

OVERALL STATISTICS
--------------------------------------------------------------------------------
Total Functions Profiled: 15
Total Execution Time: 12.3456 seconds
Total Function Calls: 1,234
Average Time per Call: 0.010012 seconds

TOP 10 SLOWEST FUNCTIONS (by total time)
--------------------------------------------------------------------------------

1. data_preprocessor.AdvancedDataPreprocessor.clean_data
   Total Time: 5.2341s (42.4%)
   Calls: 100
   Mean Time: 0.052341s
   Min/Max: 0.045123s / 0.061234s
   P95 Time: 0.059876s

2. ml_toolbox.algorithms.train_classifier
   Total Time: 3.1234s (25.3%)
   Calls: 50
   Mean Time: 0.062468s
   ...

BOTTLENECKS IDENTIFIED
--------------------------------------------------------------------------------
Found 3 potential bottlenecks:

1. data_preprocessor.AdvancedDataPreprocessor.clean_data [HIGH PRIORITY]
   Total Time: 5.2341s (42.4% of total)
   Calls: 100
   Mean Time: 0.052341s
   
   Recommendations:
   • High call count (100): Consider caching or memoization
   • Slow execution (0.052341s): Consider optimization or parallelization
   • Preprocessing function: Consider caching transformed data
```

---

## 🔧 **Integration with ML Toolbox**

### **Automatic Profiling**

Use `ProfiledMLToolbox` for automatic profiling:

```python
from ml_profiler import ProfiledMLToolbox
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
profiled = ProfiledMLToolbox(toolbox)

# All operations are automatically profiled
result = profiled.toolbox.algorithms.get_simple_ml_tasks().train_classifier(X, y)

# Get report
report = profiled.get_profiling_report()
bottlenecks = profiled.get_bottlenecks()
```

### **Via ML Toolbox API**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Get profiler
profiler = toolbox.algorithms.get_ml_profiler()

# Profile operations
@profiler.profile_function
def my_operation():
    return toolbox.algorithms.get_simple_ml_tasks().train_classifier(X, y)
```

---

## 📚 **Advanced Usage**

### **Custom Profiling**

```python
profiler = MLProfiler()

# Profile with custom context
with profiler.profile_pipeline('custom_pipeline') as pipeline:
    step1_time = time.perf_counter()
    result1 = step1()
    pipeline.add_step('step1', time.perf_counter() - step1_time)
    
    step2_time = time.perf_counter()
    result2 = step2()
    pipeline.add_step('step2', time.perf_counter() - step2_time)
```

### **Reset Profiler**

```python
profiler.reset()  # Clear all profiling data
```

### **Filter Functions**

```python
# Get statistics for specific function
stats = profiler.get_function_statistics('my_function')
```

---

## 🎓 **Tips for Effective Profiling**

1. **Profile in Production-Like Environment**
   - Use realistic data sizes
   - Use actual workloads

2. **Profile Multiple Runs**
   - Get statistical significance
   - Identify variance

3. **Focus on Hot Paths**
   - Profile frequently executed code
   - Profile critical paths

4. **Compare Optimizations**
   - Profile before and after
   - Measure actual improvements

5. **Use Appropriate Tools**
   - Function-level: decorators
   - Pipeline-level: context managers
   - Detailed: cProfile

---

## 📞 **Troubleshooting**

### **"Profiler not tracking functions"**
- Ensure decorator is applied correctly
- Check function is actually being called

### **"Memory profiling not working"**
- Install: `pip install memory-profiler psutil`
- Enable: `MLProfiler(enable_memory_profiling=True)`

### **"Reports are empty"**
- Ensure functions are being profiled
- Check profiling data exists: `profiler.function_times`

---

**Happy Profiling! 🚀**
