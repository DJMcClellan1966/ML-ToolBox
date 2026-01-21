# Clustering Analysis with AdvancedDataPreprocessor

## Overview

**Yes, AdvancedDataPreprocessor can work with unlabeled data for clustering analysis!**

Clustering is an **unsupervised learning** technique that doesn't require labels. The AdvancedDataPreprocessor is perfect for this because it:

1. ✅ **Works with unlabeled data** - No labels needed
2. ✅ **Creates semantic embeddings** - Quantum Kernel provides rich features
3. ✅ **Removes duplicates** - Semantic deduplication finds similar items
4. ✅ **Compresses data** - Dimensionality reduction for faster clustering
5. ✅ **Categorizes automatically** - Can pre-categorize before clustering

---

## How It Works

### Step 1: Preprocess Unlabeled Data

```python
from data_preprocessor import AdvancedDataPreprocessor

# Create preprocessor
preprocessor = AdvancedDataPreprocessor(
    dedup_threshold=0.85,      # Lower = keep more samples
    enable_compression=True,
    compression_ratio=0.5      # 50% of original dimensions
)

# Preprocess unlabeled data
results = preprocessor.preprocess(unlabeled_texts, verbose=True)
```

### Step 2: Extract Features for Clustering

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Option 1: Use compressed embeddings (recommended)
if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
    X = results['compressed_embeddings']
    print(f"Using compressed embeddings: {X.shape}")

# Option 2: Use original quantum embeddings
else:
    processed_texts = results['deduplicated']
    X = np.array([preprocessor.quantum_kernel.embed(text) for text in processed_texts])
    print(f"Using quantum embeddings: {X.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 3: Apply Clustering Algorithms

```python
# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Agglomerative Clustering (hierarchical)
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X_scaled)
```

### Step 4: Evaluate Clustering Quality

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Silhouette Score (higher is better, range: -1 to 1)
silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Davies-Bouldin Score (lower is better)
davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")

# Calinski-Harabasz Score (higher is better)
calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
```

---

## Complete Example

```python
from data_preprocessor import AdvancedDataPreprocessor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Unlabeled data
unlabeled_texts = [
    "Python programming is great for data science",
    "Machine learning uses neural networks",
    "JavaScript is used for web development",
    "Customer satisfaction drives business growth",
    "Revenue increased by twenty percent",
    "I need help with technical issues",
    "Support team provides assistance",
    # ... more unlabeled texts
]

# 2. Preprocess with AdvancedDataPreprocessor
preprocessor = AdvancedDataPreprocessor(
    dedup_threshold=0.85,      # Adjust based on your data
    enable_compression=True,
    compression_ratio=0.5
)

results = preprocessor.preprocess(unlabeled_texts, verbose=True)

# 3. Extract features
processed_texts = results['deduplicated']

# Use compressed embeddings if available
if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
    X = results['compressed_embeddings']
else:
    # Fallback to original embeddings
    X = np.array([preprocessor.quantum_kernel.embed(text) for text in processed_texts])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Cluster
n_clusters = 4  # Adjust based on your data
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 5. Evaluate
silhouette = silhouette_score(X_scaled, labels)
print(f"\nClustering Results:")
print(f"  Samples: {len(processed_texts)}")
print(f"  Clusters: {n_clusters}")
print(f"  Silhouette Score: {silhouette:.4f}")

# 6. View clusters
for cluster_id in range(n_clusters):
    cluster_texts = [processed_texts[i] for i in range(len(processed_texts)) if labels[i] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_texts)} items):")
    for text in cluster_texts[:5]:  # Show first 5
        print(f"  - {text}")
```

---

## Key Benefits for Clustering

### ✅ **Semantic Understanding**

- **Quantum Kernel embeddings** capture meaning, not just keywords
- Finds similar content even with different wording
- Better clustering for semantic similarity

### ✅ **Automatic Deduplication**

- Removes semantic duplicates before clustering
- Reduces noise in clusters
- Focuses on unique content

### ✅ **Dimensionality Reduction**

- **Compressed embeddings** reduce feature space
- Faster clustering algorithms
- Lower memory usage
- Retains most variance (typically 90%+)

### ✅ **Quality Filtering**

- Filters unsafe content (PocketFence)
- Scores data quality
- Keeps only high-quality samples

---

## Configuration Tips

### For More Samples (Less Aggressive Deduplication)

**Important:** AdvancedDataPreprocessor performs semantic deduplication, which means it removes items that are semantically similar (even with different wording). For clustering, you may want to keep more samples:

```python
preprocessor = AdvancedDataPreprocessor(
    dedup_threshold=0.70,  # Lower = keep more samples (recommended for clustering)
    enable_compression=True,
    compression_ratio=0.5
)
```

**Note:** The default `dedup_threshold=0.85` is quite aggressive and may reduce your dataset significantly. For clustering with unlabeled data, consider:
- **0.70-0.75**: Good balance for clustering (keeps more samples)
- **0.80-0.85**: More aggressive (removes more duplicates)
- **0.90+**: Very aggressive (only keeps very unique items)

### For Better Clustering Quality

```python
preprocessor = AdvancedDataPreprocessor(
    dedup_threshold=0.90,  # Higher = more aggressive deduplication
    enable_compression=True,
    compression_ratio=0.3  # Lower = more compression, faster clustering
)
```

### For Maximum Compression

```python
preprocessor = AdvancedDataPreprocessor(
    dedup_threshold=0.85,
    enable_compression=True,
    compression_ratio=0.1  # 10% of original dimensions
)
```

---

## Clustering Algorithms Comparison

### K-Means
- ✅ Fast and scalable
- ✅ Works well with compressed embeddings
- ❌ Requires number of clusters upfront
- ❌ Assumes spherical clusters

### DBSCAN
- ✅ Finds arbitrary-shaped clusters
- ✅ Identifies noise/outliers
- ❌ Sensitive to parameters (eps, min_samples)
- ❌ May not work well with high-dimensional data

### Agglomerative Clustering
- ✅ Creates hierarchical clusters
- ✅ No assumption about cluster shape
- ❌ Slower for large datasets
- ❌ Requires number of clusters

---

## Test Results Summary

From `tests/test_clustering_analysis.py`:

### AdvancedDataPreprocessor Advantages:

1. **Better Davies-Bouldin Score** (lower is better)
   - KMeans: 1.1881 vs 1.3159 (improvement: +0.13)
   - Agglomerative: 0.7746 vs 1.4916 (improvement: +0.72)

2. **Compressed Features**
   - 7 features vs 393 features (97% reduction)
   - Faster clustering
   - Lower memory usage

3. **Semantic Deduplication**
   - Removes semantic duplicates
   - Keeps unique content
   - Better cluster quality

### Trade-offs:

1. **Fewer Samples**
   - More aggressive deduplication
   - May reduce to fewer samples
   - Adjust `dedup_threshold` accordingly

2. **Lower Silhouette Score** (in some cases)
   - Due to fewer samples
   - But better Davies-Bouldin score
   - Overall clustering quality is good

---

## Use Cases

### 1. **Document Clustering**
```python
# Cluster documents by topic
documents = [...]  # Unlabeled documents
results = preprocessor.preprocess(documents)
# Cluster by semantic similarity
```

### 2. **Customer Feedback Analysis**
```python
# Cluster customer feedback by theme
feedback = [...]  # Unlabeled feedback
results = preprocessor.preprocess(feedback)
# Discover common themes
```

### 3. **Content Organization**
```python
# Organize content by category
content = [...]  # Unlabeled content
results = preprocessor.preprocess(content)
# Automatic categorization + clustering
```

### 4. **Anomaly Detection**
```python
# Find outliers in unlabeled data
data = [...]  # Unlabeled data
results = preprocessor.preprocess(data)
# Use DBSCAN to find anomalies
```

---

## Best Practices

### 1. **Adjust Deduplication Threshold**

- **High threshold (0.9-0.95)**: More aggressive, fewer samples
- **Low threshold (0.7-0.8)**: Less aggressive, more samples
- **Start with 0.85** and adjust based on results

### 2. **Choose Compression Ratio**

- **0.5 (50%)**: Good balance
- **0.3 (30%)**: More compression, faster
- **0.7 (70%)**: Less compression, more information

### 3. **Evaluate Multiple Algorithms**

- Try KMeans, DBSCAN, and Agglomerative
- Compare metrics
- Choose best for your data

### 4. **Use Compressed Embeddings**

- Faster clustering
- Lower memory
- Usually retains 90%+ variance

---

## Summary

**AdvancedDataPreprocessor is excellent for clustering analysis with unlabeled data:**

✅ **Works with unlabeled data** - No labels required  
✅ **Semantic embeddings** - Quantum Kernel provides rich features  
✅ **Automatic deduplication** - Removes semantic duplicates  
✅ **Dimensionality reduction** - Compressed embeddings for faster clustering  
✅ **Quality filtering** - Safety + quality scoring  
✅ **Multiple clustering algorithms** - KMeans, DBSCAN, Agglomerative  

**Key Configuration:**
- `dedup_threshold`: Control deduplication aggressiveness
- `compression_ratio`: Control feature compression
- `enable_compression`: Use compressed embeddings

**Result:** Clean, compressed, high-quality features ready for clustering!
