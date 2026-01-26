# Building a Better RAG System

## 🎯 Current RAG System Issues

### **What's Wrong:**
- ❌ Simple TF-IDF-like embeddings (not semantic)
- ❌ Vocabulary-based (doesn't understand meaning)
- ❌ Poor similarity matching
- ❌ No proper vector storage
- ❌ Limited to exact word matches

### **Impact:**
- Answers often miss relevant content
- Can't find semantically similar content
- Poor retrieval quality
- Generic responses

---

## 🚀 Better RAG System Design

### **Key Components:**

1. **Proper Embeddings** - Use sentence-transformers or similar
2. **Vector Database** - Efficient similarity search
3. **Chunking Strategy** - Smart document splitting
4. **Hybrid Search** - Combine semantic + keyword search
5. **Re-ranking** - Improve result quality

---

## 📝 Implementation Guide

### **Option 1: Sentence-Transformers (Recommended)**

**Why:** Free, local, good quality, easy to use

**Installation:**
```bash
pip install sentence-transformers
```

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import pickle
from pathlib import Path

class BetterRAGSystem:
    """
    Improved RAG System with proper embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize better RAG system
        
        Args:
            model_name: Sentence transformer model name
                       Options: 'all-MiniLM-L6-v2' (fast, good)
                               'all-mpnet-base-v2' (better, slower)
        """
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
    def add_document(self, content: str, doc_id: str = None, metadata: Dict = None):
        """Add document with proper embedding"""
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}"
        
        # Create embedding
        embedding = self.encoder.encode(content, convert_to_numpy=True)
        
        # Store
        self.documents.append({
            'id': doc_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {}
        })
        
        # Update embeddings matrix
        self._update_embeddings_matrix()
    
    def _update_embeddings_matrix(self):
        """Update embeddings matrix for batch search"""
        if not self.documents:
            self.embeddings = None
            return
        
        embeddings_list = [doc['embedding'] for doc in self.documents]
        self.embeddings = np.vstack(embeddings_list)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents using semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        # Compute similarities (cosine similarity)
        if self.embeddings is not None:
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
            
            # Cosine similarity
            similarities = np.dot(doc_norms, query_norm)
        else:
            similarities = np.array([0.0])
        
        # Get top_k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'score': float(similarities[idx]),
                'metadata': doc['metadata']
            })
        
        return results
    
    def batch_add(self, documents: List[Dict]):
        """Add multiple documents efficiently"""
        contents = [doc.get('content', '') for doc in documents]
        
        # Batch encode (much faster)
        embeddings = self.encoder.encode(contents, convert_to_numpy=True, show_progress_bar=True)
        
        for i, doc in enumerate(documents):
            self.documents.append({
                'id': doc.get('id', f"doc_{len(self.documents)}"),
                'content': doc.get('content', ''),
                'embedding': embeddings[i],
                'metadata': doc.get('metadata', {})
            })
        
        self._update_embeddings_matrix()
    
    def save(self, filepath: str):
        """Save RAG system to disk"""
        data = {
            'documents': self.documents,
            'model_name': self.encoder.get_sentence_embedding_dimension()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load RAG system from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self._update_embeddings_matrix()
```

---

### **Option 2: Vector Database (Best for Scale)**

**Why:** Much faster for large knowledge bases, better search

**Options:**
- **Chroma** (easiest, local)
- **FAISS** (fastest, Facebook)
- **Pinecone** (cloud, managed)
- **Qdrant** (self-hosted, powerful)

**Chroma Implementation (Recommended):**
```python
import chromadb
from chromadb.config import Settings

class VectorDBRAGSystem:
    """
    RAG System using Chroma vector database
    """
    
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str = "./chroma_db"):
        """Initialize Chroma-based RAG"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_document(self, content: str, doc_id: str = None, metadata: Dict = None):
        """Add document to vector DB"""
        if doc_id is None:
            doc_id = f"doc_{len(self.collection.get()['ids'])}"
        
        # Create embedding
        embedding = self.encoder.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using vector DB"""
        # Encode query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        
        return formatted
```

---

### **Option 3: Hybrid Search (Best Quality)**

**Why:** Combines semantic search + keyword search for best results

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

class HybridRAGSystem:
    """
    Hybrid RAG: Semantic + Keyword search
    """
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.semantic_embeddings = None
        self.keyword_vectors = None
    
    def add_document(self, content: str, doc_id: str = None, metadata: Dict = None):
        """Add document with both semantic and keyword vectors"""
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}"
        
        # Semantic embedding
        semantic_emb = self.encoder.encode(content, convert_to_numpy=True)
        
        # Store
        self.documents.append({
            'id': doc_id,
            'content': content,
            'semantic_embedding': semantic_emb,
            'metadata': metadata or {}
        })
        
        # Update matrices
        self._update_matrices()
    
    def _update_matrices(self):
        """Update both embedding matrices"""
        if not self.documents:
            return
        
        # Semantic embeddings
        semantic_list = [doc['semantic_embedding'] for doc in self.documents]
        self.semantic_embeddings = np.vstack(semantic_list)
        
        # Keyword vectors (TF-IDF)
        contents = [doc['content'] for doc in self.documents]
        self.keyword_vectors = self.tfidf.fit_transform(contents)
    
    def retrieve(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid retrieval: semantic + keyword
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
                           Higher = more semantic, lower = more keyword
        """
        if not self.documents:
            return []
        
        # Semantic search
        query_semantic = self.encoder.encode(query, convert_to_numpy=True)
        query_semantic_norm = query_semantic / (np.linalg.norm(query_semantic) + 1e-10)
        doc_semantic_norm = self.semantic_embeddings / (np.linalg.norm(self.semantic_embeddings, axis=1, keepdims=True) + 1e-10)
        semantic_scores = np.dot(doc_semantic_norm, query_semantic_norm)
        
        # Keyword search
        query_keyword = self.tfidf.transform([query])
        keyword_scores = (self.keyword_vectors * query_keyword.T).toarray().flatten()
        keyword_scores = keyword_scores / (np.max(keyword_scores) + 1e-10)  # Normalize
        
        # Combine scores
        combined_scores = (semantic_weight * semantic_scores) + ((1 - semantic_weight) * keyword_scores)
        
        # Get top_k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores[idx]),
                'metadata': doc['metadata']
            })
        
        return results
```

---

## 🔧 Integration with LLM Twin

### **Replace Current RAG:**

```python
# In llm_twin_learning_companion.py

from better_rag_system import BetterRAGSystem  # Your new RAG

class LLMTwinLearningCompanion:
    def __init__(self, ...):
        # Replace old RAG with better one
        self.rag = RAGSystem(
            knowledge_retriever=BetterRAGSystem()  # Use better RAG
        )
```

---

## 📊 Comparison

| Feature | Current RAG | Better RAG (Sentence-Transformers) | Vector DB (Chroma) | Hybrid |
|---------|-------------|-----------------------------------|-------------------|--------|
| **Semantic Understanding** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Search Quality** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed (small KB)** | Fast | Fast | Fast | Medium |
| **Speed (large KB)** | Slow | Medium | ⚡ Very Fast | Medium |
| **Setup Complexity** | Low | Low | Medium | Medium |
| **Storage** | Memory | Memory/Disk | Disk | Memory/Disk |
| **Best For** | Testing | < 10K docs | > 10K docs | Best quality |

---

## 🎯 Recommended Approach

### **For Your Use Case:**

**Start with Sentence-Transformers** (Option 1):
- ✅ Easy to implement
- ✅ Huge improvement over current
- ✅ Good for most use cases
- ✅ Free and local

**Upgrade to Chroma** (Option 2) if:
- You have > 10,000 documents
- Need faster search
- Want persistence

**Use Hybrid** (Option 3) if:
- You want the best quality
- Have mixed content types
- Need both semantic and exact matches

---

## 🚀 Quick Implementation

### **Step 1: Install**
```bash
pip install sentence-transformers
```

### **Step 2: Create Better RAG**
```python
# better_rag_system.py
from sentence_transformers import SentenceTransformer
import numpy as np

class BetterRAGSystem:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
    
    def add_document(self, content: str, doc_id: str = None):
        embedding = self.encoder.encode(content, convert_to_numpy=True)
        self.documents.append({
            'id': doc_id or f"doc_{len(self.documents)}",
            'content': content,
            'embedding': embedding
        })
        self._update_embeddings()
    
    def _update_embeddings(self):
        if self.documents:
            self.embeddings = np.vstack([d['embedding'] for d in self.documents])
    
    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.encoder.encode(query, convert_to_numpy=True)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(doc_norms, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
```

### **Step 3: Integrate**
```python
# Update rag_system.py to use BetterRAGSystem
```

---

## 💡 Advanced Features

### **1. Chunking Strategy**
```python
def chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50):
    """Smart chunking with overlap"""
    words = content.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### **2. Re-ranking**
```python
def rerank(self, query: str, candidates: List[Dict], top_k: int = 3):
    """Re-rank results for better quality"""
    # Use cross-encoder for re-ranking
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    pairs = [[query, c['content']] for c in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]
```

### **3. Metadata Filtering**
```python
def retrieve_with_filter(self, query: str, filters: Dict, top_k: int = 5):
    """Retrieve with metadata filters"""
    results = self.retrieve(query, top_k * 2)  # Get more
    filtered = [r for r in results if self._matches_filter(r['metadata'], filters)]
    return filtered[:top_k]
```

---

## 📈 Expected Improvements

### **Before (Current RAG):**
- ❌ "What is machine learning?" → Returns random content
- ❌ Can't find semantically similar content
- ❌ Poor relevance

### **After (Better RAG):**
- ✅ "What is machine learning?" → Returns actual ML content
- ✅ Finds semantically similar content
- ✅ High relevance scores
- ✅ Better answer quality

---

## 🎯 Next Steps

1. **Install sentence-transformers:**
   ```bash
   pip install sentence-transformers
   ```

2. **Create better_rag_system.py** (use code above)

3. **Test it:**
   ```python
   rag = BetterRAGSystem()
   rag.add_document("Machine learning is...")
   results = rag.retrieve("What is ML?")
   ```

4. **Integrate with LLM Twin**

---

**The biggest improvement: Use proper semantic embeddings instead of TF-IDF!**
