"""
Kernel Comparison Tests
Compares Quantum Kernel (Python) vs PocketFence Kernel (C#)
Note: These serve different purposes, so we test their respective capabilities
"""
import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig


class KernelComparison:
    """Compare Quantum Kernel and PocketFence Kernel capabilities"""
    
    def __init__(self):
        """Initialize comparison test"""
        print("="*70)
        print("KERNEL COMPARISON: Quantum Kernel vs PocketFence Kernel")
        print("="*70)
        print("\nNote: These kernels serve different purposes:")
        print("  - Quantum Kernel: Semantic AI (embeddings, similarity, relationships)")
        print("  - PocketFence Kernel: Content filtering (safety, URL checking)")
        print("\nWe'll test each in their respective domains.")
        print("="*70 + "\n")
        
        # Initialize Quantum Kernel
        self.quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    
    def test_quantum_kernel_semantic_search(self):
        """Test Quantum Kernel: Semantic search capabilities"""
        print("\n" + "="*70)
        print("TEST 1: Quantum Kernel - Semantic Search")
        print("="*70)
        
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing enables computers to understand text",
            "Quantum computing uses quantum mechanical phenomena",
            "Python is a popular programming language for data science"
        ]
        
        query = "AI and neural networks"
        
        start_time = time.time()
        results = self.quantum_kernel.find_similar(query, documents, top_k=3)
        elapsed = time.time() - start_time
        
        print(f"\nQuery: '{query}'")
        print(f"Time: {elapsed*1000:.2f}ms")
        print(f"\nTop Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {doc}")
        
        return results
    
    def test_quantum_kernel_similarity(self):
        """Test Quantum Kernel: Similarity computation"""
        print("\n" + "="*70)
        print("TEST 2: Quantum Kernel - Similarity Computation")
        print("="*70)
        
        pairs = [
            ("machine learning", "artificial intelligence"),
            ("Python programming", "coding in Python"),
            ("quantum computing", "classical computing"),
            ("deep learning", "neural networks")
        ]
        
        print("\nSimilarity Scores:")
        for text1, text2 in pairs:
            similarity = self.quantum_kernel.similarity(text1, text2)
            print(f"  '{text1}' <-> '{text2}': {similarity:.3f}")
    
    def test_quantum_kernel_relationships(self):
        """Test Quantum Kernel: Relationship discovery"""
        print("\n" + "="*70)
        print("TEST 3: Quantum Kernel - Relationship Discovery")
        print("="*70)
        
        texts = [
            "Machine learning algorithms",
            "Neural network architectures",
            "Deep learning models",
            "Artificial intelligence systems"
        ]
        
        print("\nBuilding relationship graph...")
        graph = self.quantum_kernel.build_relationship_graph(texts, threshold=0.5)
        
        print(f"\nFound {len(graph)} relationships:")
        for text, related in list(graph.items())[:5]:
            print(f"\n  '{text[:40]}...'")
            for rel_text, score in related[:3]:
                print(f"    -> '{rel_text[:40]}...' [{score:.3f}]")
    
    def test_quantum_kernel_performance(self):
        """Test Quantum Kernel: Performance metrics"""
        print("\n" + "="*70)
        print("TEST 4: Quantum Kernel - Performance")
        print("="*70)
        
        # Test embedding speed
        test_texts = ["Test text " + str(i) for i in range(100)]
        
        start = time.time()
        embeddings = [self.quantum_kernel.embed(text) for text in test_texts]
        embed_time = time.time() - start
        
        # Test similarity search speed
        query = "test query"
        start = time.time()
        results = self.quantum_kernel.find_similar(query, test_texts, top_k=10)
        search_time = time.time() - start
        
        stats = self.quantum_kernel.get_stats()
        
        print(f"\nPerformance Metrics:")
        print(f"  Embeddings (100 texts): {embed_time*1000:.2f}ms ({embed_time/100*1000:.2f}ms per text)")
        print(f"  Similarity search: {search_time*1000:.2f}ms")
        print(f"  Cache hits: {stats.get('cache_hits', 0)}")
        print(f"  Cache size: {stats.get('cache_size', 0)}")
        print(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
    
    def test_pocketfence_kernel_info(self):
        """Test PocketFence Kernel: Information about capabilities"""
        print("\n" + "="*70)
        print("TEST 5: PocketFence Kernel - Capabilities")
        print("="*70)
        
        print("\nPocketFence Kernel (C#) Capabilities:")
        print("  - Content filtering (text analysis)")
        print("  - URL safety checking")
        print("  - Threat detection")
        print("  - Plugin system for custom filters")
        print("  - REST API for application integration")
        print("  - Batch processing")
        print("  - Statistics and monitoring")
        
        print("\nNote: PocketFence Kernel is a C# service that requires:")
        print("  - .NET 8.0 runtime")
        print("  - Running as background service or API")
        print("  - Different testing approach (API calls or service integration)")
        
        print("\nTo test PocketFence Kernel:")
        print("  1. Start the service: cd PocketFenceKernel && dotnet run -- --kernel")
        print("  2. Make API calls to http://localhost:5000/api/filter/")
        print("  3. Test URL checking, content filtering, etc.")
    
    def test_hybrid_approach(self):
        """Test: How both kernels could work together"""
        print("\n" + "="*70)
        print("TEST 6: Hybrid Approach - Both Kernels Together")
        print("="*70)
        
        print("\nConceptual Integration:")
        print("  1. Quantum Kernel: Understand semantic meaning")
        print("  2. PocketFence Kernel: Check safety/filtering")
        print("  3. Combined: Safe, intelligent content processing")
        
        print("\nExample Workflow:")
        print("  Step 1: User query -> Quantum Kernel (semantic understanding)")
        print("  Step 2: Retrieved content -> PocketFence Kernel (safety check)")
        print("  Step 3: Safe, relevant results -> User")
        
        # Simulate hybrid workflow
        user_query = "Tell me about machine learning"
        documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Python is great for ML"
        ]
        
        print(f"\nSimulated Hybrid Workflow:")
        print(f"  Query: '{user_query}'")
        
        # Step 1: Quantum Kernel - semantic search
        results = self.quantum_kernel.find_similar(user_query, documents, top_k=3)
        print(f"\n  [Quantum Kernel] Found {len(results)} relevant documents:")
        for doc, score in results:
            print(f"    - [{score:.3f}] {doc}")
        
        # Step 2: PocketFence Kernel - safety check (simulated)
        print(f"\n  [PocketFence Kernel] Safety check (simulated):")
        for doc, score in results:
            # In real scenario, would call PocketFence API
            is_safe = True  # Simulated
            print(f"    - '{doc[:40]}...' -> Safe: {is_safe}")
        
        print(f"\n  [Result] Safe, relevant content delivered to user")


def run_all_tests():
    """Run all comparison tests"""
    comparison = KernelComparison()
    
    try:
        # Quantum Kernel tests
        comparison.test_quantum_kernel_semantic_search()
        comparison.test_quantum_kernel_similarity()
        comparison.test_quantum_kernel_relationships()
        comparison.test_quantum_kernel_performance()
        
        # PocketFence Kernel info
        comparison.test_pocketfence_kernel_info()
        
        # Hybrid approach
        comparison.test_hybrid_approach()
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\nQuantum Kernel Strengths:")
        print("  [+] Semantic understanding")
        print("  [+] Similarity computation")
        print("  [+] Relationship discovery")
        print("  [+] Knowledge graph building")
        print("  [+] Fast embeddings and search")
        
        print("\nPocketFence Kernel Strengths:")
        print("  [+] Content filtering")
        print("  [+] URL safety checking")
        print("  [+] Threat detection")
        print("  [+] Plugin extensibility")
        print("  [+] Production service architecture")
        
        print("\nBest Use:")
        print("  - Quantum Kernel: AI/ML applications, semantic search")
        print("  - PocketFence Kernel: Safety, content filtering, child protection")
        print("  - Together: Safe, intelligent content processing")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
