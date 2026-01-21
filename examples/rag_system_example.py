"""
RAG (Retrieval-Augmented Generation) System Example
Combines quantum kernel + AI system + LLM for intelligent Q&A
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from ai import CompleteAISystem
from llm.quantum_llm_standalone import StandaloneQuantumLLM


class RAGSystem:
    """
    Retrieval-Augmented Generation System
    
    Combines:
    - Semantic search (quantum kernel)
    - Knowledge graphs (AI system)  
    - Text generation (LLM)
    
    Result: AI that answers questions from your knowledge base
    """
    
    def __init__(self):
        """Initialize RAG system"""
        print("[+] Initializing RAG System...")
        
        # Initialize components
        config = KernelConfig(use_sentence_transformers=True)
        self.kernel = get_kernel(config)
        self.ai = CompleteAISystem(config=config, use_llm=True)
        self.llm = StandaloneQuantumLLM(kernel=self.kernel)
        
        self.knowledge_base = []
        print("[+] RAG System ready!")
    
    def add_knowledge(self, documents):
        """
        Add documents to knowledge base
        
        Args:
            documents: List of text documents
        """
        print(f"\n[+] Adding {len(documents)} documents to knowledge base...")
        
        for doc in documents:
            self.knowledge_base.append(doc)
            # Add to AI system knowledge graph
            self.ai.knowledge_graph.add_document(doc)
        
        print(f"[+] Knowledge base now has {len(self.knowledge_base)} documents")
    
    def query(self, question, top_k=5):
        """
        Answer question using RAG
        
        Process:
        1. Retrieve relevant documents (semantic search)
        2. Extract context
        3. Generate answer using LLM with context
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and confidence
        """
        print(f"\n[?] Question: {question}")
        
        # Step 1: Retrieve relevant documents
        print("[1/3] Retrieving relevant documents...")
        results = self.kernel.find_similar(
            question, 
            self.knowledge_base, 
            top_k=top_k
        )
        
        if not results:
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'confidence': 0.0
            }
        
        print(f"    Found {len(results)} relevant documents")
        
        # Step 2: Build context from retrieved documents
        print("[2/3] Building context...")
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i} (relevance: {score:.2f})]: {doc[:200]}...")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer with context
        print("[3/3] Generating answer...")
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so."""
        
        response = self.llm.generate_grounded(prompt, max_length=300)
        
        answer = response.get('generated') or response.get('text', '')
        confidence = response.get('confidence', 0.0)
        
        print(f"[✓] Generated answer (confidence: {confidence:.2f})")
        
        return {
            'answer': answer,
            'sources': [doc for doc, score in results],
            'source_scores': [score for doc, score in results],
            'confidence': confidence,
            'num_sources': len(results)
        }


def demo_rag_system():
    """Demonstrate RAG system"""
    print("="*70)
    print("RAG SYSTEM DEMONSTRATION")
    print("Retrieval-Augmented Generation")
    print("="*70)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Add knowledge base
    knowledge = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Natural language processing (NLP) enables computers to understand and process human language.",
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement.",
        "The quantum kernel uses semantic embeddings to understand text meaning.",
        "Vector databases store embeddings for efficient similarity search.",
        "RAG systems combine retrieval and generation for accurate, grounded responses."
    ]
    
    rag.add_knowledge(knowledge)
    
    # Ask questions
    questions = [
        "What is Python?",
        "How does machine learning relate to AI?",
        "What is RAG and how does it work?",
        "What are quantum computers used for?"
    ]
    
    print("\n" + "="*70)
    print("ASKING QUESTIONS")
    print("="*70)
    
    for question in questions:
        result = rag.query(question)
        
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print(f"{'='*70}")
        print(f"\nA: {result['answer']}")
        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"Sources used: {result['num_sources']}")
        print(f"\nTop source: {result['sources'][0][:100]}...")
        print("="*70)


if __name__ == "__main__":
    try:
        demo_rag_system()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")
