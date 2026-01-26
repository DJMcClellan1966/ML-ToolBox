"""
Better RAG System with Sentence Transformers

Replaces the simple TF-IDF RAG with proper semantic embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class BetterKnowledgeRetriever:
    """
    Improved Knowledge Retriever with proper semantic embeddings
    
    Uses sentence-transformers for semantic understanding instead of
    simple TF-IDF-like embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize better knowledge retriever
        
        Args:
            model_name: Sentence transformer model name
                       'all-MiniLM-L6-v2' - Fast, good quality (default)
                       'all-mpnet-base-v2' - Better quality, slower
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        logger.info(f"Better Knowledge Retriever initialized with model: {model_name}")
    
    def add_document(self, doc_id: str, content: str, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """
        Add document with proper semantic embedding
        
        Args:
            doc_id: Document identifier
            content: Document content
            embedding: Optional pre-computed embedding
            metadata: Optional metadata
        """
        # Create embedding if not provided
        if embedding is None:
            embedding = self.encoder.encode(content, convert_to_numpy=True, show_progress_bar=False)
        else:
            embedding = np.array(embedding)
        
        # Store document
        self.documents.append({
            'id': doc_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {}
        })
        
        # Update embeddings matrix for efficient batch search
        self._update_embeddings_matrix()
    
    def _update_embeddings_matrix(self):
        """Update embeddings matrix for batch similarity computation"""
        if not self.documents:
            self.embeddings = None
            return
        
        embeddings_list = [doc['embedding'] for doc in self.documents]
        self.embeddings = np.vstack(embeddings_list)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        query_embedding = self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute cosine similarities
        if self.embeddings is not None:
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
            
            # Cosine similarity (dot product of normalized vectors)
            similarities = np.dot(doc_norms, query_norm)
        else:
            similarities = np.array([0.0])
        
        # Get top_k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'doc_id': doc['id'],
                'content': doc['content'],
                'score': float(similarities[idx]),
                'metadata': doc['metadata']
            })
        
        return results
    
    def batch_add_documents(self, documents: List[Dict]):
        """
        Add multiple documents efficiently (batch encoding)
        
        Args:
            documents: List of dicts with 'id', 'content', optional 'metadata'
        """
        if not documents:
            return
        
        contents = [doc.get('content', '') for doc in documents]
        
        # Batch encode (much faster than individual encoding)
        embeddings = self.encoder.encode(
            contents,
            convert_to_numpy=True,
            show_progress_bar=len(contents) > 10
        )
        
        # Add all documents
        for i, doc in enumerate(documents):
            self.documents.append({
                'id': doc.get('id', f"doc_{len(self.documents)}"),
                'content': doc.get('content', ''),
                'embedding': embeddings[i],
                'metadata': doc.get('metadata', {})
            })
        
        self._update_embeddings_matrix()
        logger.info(f"Added {len(documents)} documents via batch encoding")
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.encoder.get_sentence_embedding_dimension()
        }


class BetterRAGSystem:
    """
    Improved RAG System with better retrieval
    
    Drop-in replacement for RAGSystem with semantic embeddings
    """
    
    def __init__(self, knowledge_retriever: Optional[BetterKnowledgeRetriever] = None, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize better RAG system
        
        Args:
            knowledge_retriever: Optional BetterKnowledgeRetriever instance
            model_name: Sentence transformer model name
        """
        if knowledge_retriever is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available. Falling back to simple RAG.")
                from ml_toolbox.llm_engineering.rag_system import RAGSystem, KnowledgeRetriever
                self.retriever = KnowledgeRetriever()
                self.is_better = False
            else:
                self.retriever = BetterKnowledgeRetriever(model_name=model_name)
                self.is_better = True
        else:
            self.retriever = knowledge_retriever
            self.is_better = True
        
        self.retrieval_history = []
    
    def add_knowledge(self, doc_id: str, content: str, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """
        Add knowledge to the system
        
        Args:
            doc_id: Document identifier
            content: Document content
            embedding: Optional pre-computed embedding
            metadata: Optional metadata
        """
        if self.is_better:
            self.retriever.add_document(doc_id, content, embedding, metadata)
        else:
            # Fallback to simple RAG
            self.retriever.add_document(doc_id, content, embedding)
    
    def augment_prompt(self, prompt: str, query: str, top_k: int = 3) -> str:
        """
        Augment prompt with retrieved context
        
        Args:
            prompt: Original prompt
            query: Query for retrieval
            top_k: Number of relevant documents to include
        
        Returns:
            Augmented prompt with context
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query, top_k=top_k)
        
        if not relevant_docs:
            return prompt
        
        # Build context section
        context_section = "\n\nRelevant Context:\n"
        for i, doc in enumerate(relevant_docs, 1):
            score = doc.get('score', 0.0)
            content = doc.get('content', '')[:300]  # Limit length
            context_section += f"\n[{i}] {content}... (relevance: {score:.3f})\n"
        
        # Augment prompt
        augmented = f"{prompt}{context_section}\n\nUse the above context to inform your response."
        
        # Store retrieval
        self.retrieval_history.append({
            'query': query,
            'retrieved_docs': len(relevant_docs),
            'top_score': relevant_docs[0].get('score', 0.0) if relevant_docs else 0.0
        })
        
        return augmented
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        if not self.retrieval_history:
            return {
                'total_retrievals': 0,
                'total_documents': len(self.retriever.documents) if hasattr(self.retriever, 'documents') else 0
            }
        
        avg_score = sum(h['top_score'] for h in self.retrieval_history) / len(self.retrieval_history)
        
        return {
            'total_retrievals': len(self.retrieval_history),
            'avg_top_score': avg_score,
            'total_documents': len(self.retriever.documents) if hasattr(self.retriever, 'documents') else 0,
            'is_better_rag': self.is_better
        }


# Compatibility wrapper - can be used as drop-in replacement
def create_better_rag(model_name: str = 'all-MiniLM-L6-v2') -> BetterRAGSystem:
    """
    Create a better RAG system
    
    Args:
        model_name: Sentence transformer model name
    
    Returns:
        BetterRAGSystem instance
    """
    return BetterRAGSystem(model_name=model_name)


if __name__ == "__main__":
    # Test the better RAG system
    print("Testing Better RAG System...")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        exit(1)
    
    # Create RAG system
    rag = BetterRAGSystem()
    
    # Add some test documents
    test_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Deep learning uses neural networks with many layers to learn complex patterns in data."
    ]
    
    print("\nAdding documents...")
    for i, doc in enumerate(test_docs):
        rag.add_knowledge(f"doc_{i}", doc, metadata={'source': 'test'})
    
    # Test retrieval
    print("\nTesting retrieval...")
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about Python programming"
    ]
    
    for query in queries:
        results = rag.retriever.retrieve(query, top_k=2)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"  Score: {result['score']:.3f} - {result['content'][:60]}...")
    
    # Get stats
    stats = rag.get_retrieval_stats()
    print(f"\nStats: {stats}")
