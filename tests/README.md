# Kernel Comparison Tests

## Overview

Tests comparing Quantum Kernel (Python) vs PocketFence Kernel (C#).

**Important:** These kernels serve different purposes:
- **Quantum Kernel**: Semantic AI (embeddings, similarity, relationships)
- **PocketFence Kernel**: Content filtering (safety, URL checking)

## Test Files

### 1. `test_kernel_comparison.py`
Comprehensive comparison tests:
- Quantum Kernel semantic search
- Quantum Kernel similarity computation
- Quantum Kernel relationship discovery
- Quantum Kernel performance metrics
- PocketFence Kernel capabilities (info)
- Hybrid approach demonstration

**Run:**
```bash
python tests/test_kernel_comparison.py
```

### 2. `test_pocketfence_integration.py`
Tests PocketFence Kernel via REST API (requires service to be running).

**Prerequisites:**
1. Start PocketFence Kernel service:
   ```bash
   cd PocketFenceKernel
   dotnet run -- --kernel
   ```

2. Service runs on `http://localhost:5000`

**Run:**
```bash
python tests/test_pocketfence_integration.py
```

## What Gets Tested

### Quantum Kernel Tests
- ✅ Semantic search (find similar documents)
- ✅ Similarity computation (text similarity scores)
- ✅ Relationship discovery (build knowledge graphs)
- ✅ Performance (embedding speed, cache efficiency)

### PocketFence Kernel Tests
- ✅ Service availability (health check)
- ✅ URL safety checking
- ✅ Content filtering
- ✅ Batch processing
- ✅ Statistics and monitoring

### Hybrid Tests
- ✅ How both kernels could work together
- ✅ Semantic understanding + Safety filtering

## Expected Results

### Quantum Kernel
- Fast semantic search (< 100ms for 100 documents)
- High similarity scores for related texts
- Relationship graphs with multiple connections
- Cache hits improve performance

### PocketFence Kernel
- URL checks return threat scores
- Content filtering identifies unsafe content
- Batch processing handles multiple items
- Statistics track usage

## Notes

- **Different Domains**: These kernels complement each other, don't compete
- **Different Languages**: Python vs C# (different ecosystems)
- **Different Architectures**: Library vs Service
- **Can Work Together**: Quantum for understanding, PocketFence for safety
