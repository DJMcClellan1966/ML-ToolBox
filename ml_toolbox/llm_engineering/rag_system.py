"""
RAG (Retrieval Augmented Generation) System

Implements:
- Knowledge retrieval
- Context augmentation
- Semantic search
- Document embedding
"""
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from knowledge base
    
    Uses semantic search to find relevant information
    """
    
    def __init__(self, knowledge_base: Optional[Dict] = None):
        """
        Initialize knowledge retriever
        
        Parameters
        ----------
        knowledge_base : dict, optional
            Knowledge base with documents and embeddings
        """
        self.knowledge_base = knowledge_base or {}
        self.embeddings = {}
        self.documents = []
        self._vocabulary = set()  # Shared vocabulary for consistent embeddings
    
    def add_document(self, doc_id: str, content: str, embedding: Optional[np.ndarray] = None):
        """
        Add document to knowledge base
        
        Parameters
        ----------
        doc_id : str
            Document identifier
        content : str
            Document content
        embedding : array-like, optional
            Document embedding (auto-generated if not provided)
        """
        self.documents.append({'id': doc_id, 'content': content})
        
        if embedding is not None:
            self.embeddings[doc_id] = embedding
        else:
            # Simple embedding (can be enhanced with actual embedding model)
            self.embeddings[doc_id] = self._simple_embedding(content)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple embedding (TF-IDF-like)"""
        # In production, use sentence-transformers or similar
        words = text.lower().split()
        
        # Use fixed vocabulary size for consistency
        if not hasattr(self, '_vocabulary'):
            self._vocabulary = set()
        
        # Build vocabulary from all documents
        self._vocabulary.update(words)
        
        # Create embedding with fixed size
        if not self._vocabulary:
            return np.zeros(100)  # Default size
        
        vocab_list = sorted(list(self._vocabulary))
        embedding = np.zeros(len(vocab_list))
        
        for word in words:
            if word in vocab_list:
                embedding[vocab_list.index(word)] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-10)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query
        
        Parameters
        ----------
        query : str
            Query string
        top_k : int
            Number of top results to return
            
        Returns
        -------
        results : list of dict
            Relevant documents with scores
        """
        if not self.documents:
            return []
        
        # Build vocabulary from all documents first
        for doc in self.documents:
            words = doc['content'].lower().split()
            self._vocabulary.update(words)
        
        # Embed query with consistent vocabulary
        query_embedding = self._simple_embedding(query)
        
        # Compute similarities
        similarities = []
        for doc in self.documents:
            doc_id = doc['id']
            if doc_id in self.embeddings:
                doc_embedding = self.embeddings[doc_id]
                
                # Ensure same dimension
                min_dim = min(len(query_embedding), len(doc_embedding))
                if min_dim > 0:
                    query_vec = query_embedding[:min_dim]
                    doc_vec = doc_embedding[:min_dim]
                    similarity = np.dot(query_vec, doc_vec)
                else:
                    similarity = 0.0
                
                similarities.append({
                    'doc_id': doc_id,
                    'content': doc['content'],
                    'score': float(similarity)
                })
        
        # Sort by score and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]


class RAGSystem:
    """
    RAG System - Retrieval Augmented Generation
    
    Combines retrieval with generation for better LLM responses
    """
    
    def __init__(self, knowledge_retriever: Optional[KnowledgeRetriever] = None):
        """
        Initialize RAG system
        
        Parameters
        ----------
        knowledge_retriever : KnowledgeRetriever, optional
            Knowledge retriever instance
        """
        self.retriever = knowledge_retriever or KnowledgeRetriever()
        self.retrieval_history = []
        
        # Initialize vocabulary for consistent embeddings
        if not hasattr(self.retriever, '_vocabulary'):
            self.retriever._vocabulary = set()
    
    def augment_prompt(self, prompt: str, query: str, top_k: int = 3) -> str:
        """
        Augment prompt with retrieved context
        
        Parameters
        ----------
        prompt : str
            Original prompt
        query : str
            Query for retrieval
        top_k : int
            Number of relevant documents to include
            
        Returns
        -------
        augmented_prompt : str
            Prompt augmented with relevant context
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query, top_k=top_k)
        
        if not relevant_docs:
            return prompt
        
        # Build context section
        context_section = "\n\nRelevant Context:\n"
        for i, doc in enumerate(relevant_docs, 1):
            context_section += f"\n[{i}] {doc['content'][:200]}... (relevance: {doc['score']:.2f})\n"
        
        # Augment prompt
        augmented = f"{prompt}{context_section}\n\nUse the above context to inform your response."
        
        # Store retrieval
        self.retrieval_history.append({
            'query': query,
            'retrieved_docs': len(relevant_docs),
            'top_score': relevant_docs[0]['score'] if relevant_docs else 0.0
        })
        
        return augmented
    
    def add_knowledge(self, doc_id: str, content: str, embedding: Optional[np.ndarray] = None):
        """Add knowledge to the system"""
        self.retriever.add_document(doc_id, content, embedding)
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        if not self.retrieval_history:
            return {'total_retrievals': 0}
        
        avg_score = sum(h['top_score'] for h in self.retrieval_history) / len(self.retrieval_history)
        
        return {
            'total_retrievals': len(self.retrieval_history),
            'avg_top_score': avg_score,
            'total_documents': len(self.retriever.documents)
        }
