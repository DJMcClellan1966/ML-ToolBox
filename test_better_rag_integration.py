"""
Test script to verify Better RAG System integration with LLM Twin
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_better_rag_integration():
    """Test that Better RAG is properly integrated"""
    print("Testing Better RAG Integration with LLM Twin...")
    print("=" * 60)
    
    # Test 1: Check if Better RAG can be imported
    print("\n1. Testing Better RAG import...")
    try:
        from better_rag_system import BetterRAGSystem, BetterKnowledgeRetriever
        print("   [OK] Better RAG System imported successfully")
    except ImportError as e:
        print(f"   [FAIL] Failed to import Better RAG: {e}")
        print("   [INFO] Install with: pip install sentence-transformers")
        return False
    
    # Test 2: Check if sentence-transformers is available
    print("\n2. Testing sentence-transformers availability...")
    try:
        from sentence_transformers import SentenceTransformer
        print("   [OK] sentence-transformers is available")
    except ImportError:
        print("   [WARN] sentence-transformers not installed")
        print("   [INFO] Install with: pip install sentence-transformers")
        print("   [WARN] Will fall back to simple RAG")
    
    # Test 3: Test LLM Twin initialization with Better RAG
    print("\n3. Testing LLM Twin initialization...")
    try:
        from llm_twin_learning_companion import LLMTwinLearningCompanion
        
        companion = LLMTwinLearningCompanion(user_id="test_user")
        
        if companion.rag is None:
            print("   [WARN] RAG system is None (not available)")
        elif hasattr(companion.rag, 'is_better') and companion.rag.is_better:
            print("   [OK] LLM Twin using Better RAG System")
        else:
            print("   [WARN] LLM Twin using simple RAG System")
            print("   [INFO] Install sentence-transformers to use Better RAG")
        
        print(f"   [OK] LLM Twin initialized successfully")
        
    except Exception as e:
        print(f"   [FAIL] Failed to initialize LLM Twin: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test adding knowledge
    print("\n4. Testing knowledge ingestion...")
    try:
        if companion.rag:
            companion.ingest_text(
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                source="test"
            )
            print("   [OK] Knowledge added successfully")
            
            # Test retrieval
            results = companion.rag.retriever.retrieve("What is machine learning?", top_k=1)
            if results:
                print(f"   [OK] Retrieval works! Top result score: {results[0]['score']:.3f}")
            else:
                print("   [WARN] No results retrieved")
        else:
            print("   [WARN] RAG not available, skipping test")
    except Exception as e:
        print(f"   [FAIL] Failed to test knowledge ingestion: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test knowledge stats
    print("\n5. Testing knowledge stats...")
    try:
        stats = companion.get_knowledge_stats()
        print(f"   [OK] Knowledge stats retrieved: {stats}")
    except Exception as e:
        print(f"   ❌ Failed to get knowledge stats: {e}")
    
    print("\n" + "=" * 60)
    print("Integration test complete!")
    
    return True


if __name__ == "__main__":
    success = test_better_rag_integration()
    sys.exit(0 if success else 1)
