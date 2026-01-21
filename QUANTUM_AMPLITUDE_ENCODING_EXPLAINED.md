# Quantum Amplitude Encoding Explained

## What is Amplitude Encoding?

**Amplitude encoding** is a quantum-inspired technique that represents data as **quantum amplitudes** (wave-like patterns) rather than traditional numerical vectors. In quantum computing, amplitudes are complex numbers that represent the probability of finding a quantum system in a particular state.

---

## Quantum Computing Background

### In Real Quantum Computing

In quantum computing, a quantum state is represented as:
```
|ψ⟩ = α|0⟩ + β|1⟩
```

Where:
- `α` and `β` are **amplitudes** (complex numbers)
- `|α|²` is the probability of measuring state `|0⟩`
- `|β|²` is the probability of measuring state `|1⟩`
- `|α|² + |β|² = 1` (normalization)

### Amplitude Encoding

Amplitude encoding stores classical data in quantum amplitudes:
- A vector `[x₁, x₂, ..., xₙ]` becomes quantum state: `|ψ⟩ = x₁|0⟩ + x₂|1⟩ + ... + xₙ|n-1⟩`
- The data values become the amplitudes
- This allows exponential compression (n classical bits → log₂(n) qubits)

---

## How It's Used in This Codebase

### Implementation: `_quantum_amplitude_embedding()`

The quantum kernel uses a **simplified, classical simulation** of amplitude encoding:

```python
def _quantum_amplitude_embedding(text: str, dim: int) -> np.ndarray:
    """
    Quantum-inspired amplitude-based embedding
    Uses sinusoidal amplitude patterns for better pattern recognition
    """
    embedding = np.zeros(dim)
    words = text.lower().split()
    
    # Quantum amplitude encoding with phase/amplitude
    for i, word in enumerate(words[:50]):
        # Generate quantum phase from word
        phase = (hash(word) % (2 * np.pi * 1000)) / 1000.0
        
        # Quantum amplitude based on word properties
        amplitude = 1.0 / (1.0 + len(word))
        
        # Create quantum-like superposition (sinusoidal pattern)
        for j in range(dim):
            # Quantum amplitude pattern
            embedding[j] += amplitude * np.sin(phase + 2 * np.pi * j / dim)
    
    # Add character-level quantum encoding
    for i, char in enumerate(text[:min(100, len(text))]):
        char_phase = ord(char) / 255.0 * 2 * np.pi
        char_amplitude = 0.1 / (1.0 + i * 0.01)
        embedding[i % dim] += char_amplitude * np.cos(char_phase)
    
    # Quantum normalization (preserve amplitude information)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding
```

---

## Key Concepts

### 1. **Phase and Amplitude**

- **Phase**: The position in a wave cycle (0 to 2π)
  - Derived from word hash: `phase = (hash(word) % (2 * π * 1000)) / 1000.0`
  - Each word gets a unique phase based on its hash

- **Amplitude**: The strength/intensity of the wave
  - Based on word length: `amplitude = 1.0 / (1.0 + len(word))`
  - Shorter words get higher amplitude (more emphasis)

### 2. **Superposition (Wave Patterns)**

Instead of storing discrete values, amplitude encoding creates **wave patterns**:
```python
embedding[j] += amplitude * np.sin(phase + 2 * π * j / dim)
```

This creates:
- **Sinusoidal patterns** that encode word information
- **Interference effects** when multiple words contribute
- **Smooth, continuous representations** rather than discrete values

### 3. **Character-Level Encoding**

Also encodes at character level:
```python
char_phase = ord(char) / 255.0 * 2 * π
char_amplitude = 0.1 / (1.0 + i * 0.01)
embedding[i % dim] += char_amplitude * np.cos(char_phase)
```

- Each character contributes a small amplitude
- Position-dependent amplitude (earlier characters have more weight)

### 4. **Normalization**

```python
norm = np.linalg.norm(embedding)
if norm > 0:
    embedding = embedding / norm
```

- Preserves amplitude information
- Ensures unit length (like quantum state normalization)

---

## Why Use Amplitude Encoding?

### Advantages

1. **Pattern Recognition**
   - Wave patterns can capture subtle relationships
   - Sinusoidal patterns are good for periodic/cyclic data
   - Better at detecting similarities in structure

2. **Superposition Effects**
   - Multiple words create interference patterns
   - Constructive interference = similar concepts
   - Destructive interference = different concepts

3. **Smooth Representations**
   - Continuous rather than discrete
   - Better for gradient-based optimization
   - More robust to small variations

4. **Quantum-Inspired**
   - Based on quantum computing principles
   - Potential for quantum advantage (if run on real quantum hardware)
   - Novel approach to embedding

### Limitations

1. **Classical Simulation**
   - This is a **classical simulation** of quantum behavior
   - Not using real quantum hardware
   - Limited quantum advantage on classical computers

2. **Computational Cost**
   - More expensive than simple embeddings
   - Requires trigonometric functions (sin, cos)
   - Slower than direct vector operations

3. **Empirical Performance**
   - In tests, quantum methods often tie or underperform classical methods
   - Benefits are subtle and context-dependent
   - May not provide significant advantage for all tasks

---

## How It's Integrated

### In the Quantum Kernel

```python
# Configuration
config = KernelConfig(
    use_quantum_methods=True,
    quantum_amplitude_encoding=True  # Enable amplitude encoding
)

# When creating embeddings
def _create_embedding(self, text: str) -> np.ndarray:
    if self.config.use_sentence_transformers:
        # Get base embedding from SentenceTransformers
        embedding = self.embedding_model.encode(text)
        
        # Optionally enhance with quantum amplitude encoding
        if self.config.use_quantum_methods and self.config.quantum_amplitude_encoding:
            quantum_component = _quantum_amplitude_embedding(text, len(embedding))
            # Blend: 85% classical, 15% quantum
            embedding = 0.85 * embedding + 0.15 * quantum_component
    
    else:
        # Fallback: use pure quantum amplitude encoding
        if self.config.use_quantum_methods and self.config.quantum_amplitude_encoding:
            return _quantum_amplitude_embedding(text, self.config.embedding_dim)
```

### Blending Strategy

The implementation **blends** classical and quantum embeddings:
- **85% classical** (SentenceTransformers) - proven, effective
- **15% quantum** (amplitude encoding) - experimental enhancement

This provides:
- ✅ Stability from classical methods
- ✅ Potential benefits from quantum-inspired methods
- ✅ Fallback if quantum methods don't help

---

## Example

### Input Text
```
"Python programming language"
```

### Amplitude Encoding Process

1. **Word-level encoding:**
   - "python": phase = hash("python") mod (2π), amplitude = 1/(1+6) = 0.143
   - "programming": phase = hash("programming") mod (2π), amplitude = 1/(1+11) = 0.083
   - "language": phase = hash("language") mod (2π), amplitude = 1/(1+8) = 0.111

2. **Create wave patterns:**
   ```python
   for j in range(dim):
       embedding[j] += 0.143 * sin(phase_python + 2π * j / dim)
       embedding[j] += 0.083 * sin(phase_programming + 2π * j / dim)
       embedding[j] += 0.111 * sin(phase_language + 2π * j / dim)
   ```

3. **Character-level encoding:**
   - Each character adds small amplitude with cosine pattern

4. **Normalize:**
   - Divide by norm to get unit vector

### Result

A 256-dimensional vector where:
- Each dimension represents a point in the wave pattern
- Similar texts create similar wave patterns
- Wave interference captures relationships

---

## Comparison to Classical Embeddings

### Classical Embedding (e.g., SentenceTransformers)
```
"Python programming" → [0.23, -0.15, 0.42, ..., 0.08]
```
- Direct numerical representation
- Learned from training data
- Proven effectiveness

### Quantum Amplitude Encoding
```
"Python programming" → [sin(phase₁), sin(phase₂), ..., sin(phaseₙ)]
```
- Wave-based representation
- Derived from text properties (hash, length)
- Experimental approach

### Hybrid Approach (Used Here)
```
85% * [0.23, -0.15, ...] + 15% * [sin(phase₁), sin(phase₂), ...]
```
- Combines best of both
- Stable base with quantum enhancement
- Best of both worlds

---

## When to Use Amplitude Encoding

### ✅ Good For:
- **Research/Experimentation**: Testing quantum-inspired methods
- **Pattern Recognition**: Tasks requiring subtle pattern detection
- **Novel Applications**: Where classical methods struggle
- **Hybrid Approaches**: Blending with classical methods

### ❌ Not Ideal For:
- **Production Systems**: Classical methods are more proven
- **Speed-Critical**: More computationally expensive
- **Simple Tasks**: Overkill for basic similarity
- **Large-Scale**: May not scale as well

---

## Real Quantum Computing Context

### In Real Quantum Hardware

If run on actual quantum computers, amplitude encoding could:
- **Exponential Compression**: Store 2ⁿ values in n qubits
- **Quantum Speedup**: Faster processing for certain algorithms
- **Quantum Interference**: Real interference effects
- **Quantum Advantage**: Potential speedup over classical

### Current Implementation

This is a **classical simulation**:
- ✅ Demonstrates quantum concepts
- ✅ Can be tested without quantum hardware
- ✅ Educational and experimental
- ❌ No quantum advantage (running on classical computer)
- ❌ Limited to classical computational complexity

---

## Summary

**Amplitude encoding** is a quantum-inspired technique that:
1. **Represents data as wave patterns** (sinusoidal functions)
2. **Uses phase and amplitude** to encode information
3. **Creates superposition effects** from multiple inputs
4. **Provides smooth, continuous representations**

In this codebase:
- Used as an **enhancement** to classical embeddings (15% blend)
- Creates **wave patterns** from text properties
- **Experimental** - benefits are subtle
- **Classical simulation** - not using real quantum hardware

**Think of it as:** Representing text as waves instead of numbers, where wave patterns capture relationships through interference effects.

---

## Further Reading

- **Quantum Computing**: Nielsen & Chuang, "Quantum Computation and Quantum Information"
- **Amplitude Encoding**: Quantum machine learning papers
- **Quantum Kernels**: Quantum support vector machines (QSVM)
- **This Implementation**: `quantum_kernel/kernel.py` lines 109-141
