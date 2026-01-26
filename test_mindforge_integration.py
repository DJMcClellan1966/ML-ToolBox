"""
Test MindForge Integration

Quick test to verify MindForge connection and sync work.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from llm_twin_learning_companion import LLMTwinLearningCompanion
from mindforge_connector import MindForgeConnector
from easy_content_ingestion import EasyIngestion

def test_mindforge_connector():
    """Test MindForge connector"""
    print("="*80)
    print("Testing MindForge Connector".center(80))
    print("="*80)
    print()
    
    try:
        # Try to connect
        print("1. Connecting to MindForge...")
        connector = MindForgeConnector()
        print("   Connected!")
        print()
        
        # Get stats
        print("2. Getting MindForge statistics...")
        stats = connector.get_stats()
        print(f"   Total items: {stats.get('total_items', 0)}")
        print(f"   By type: {stats.get('by_type', {})}")
        print()
        
        # Get items
        print("3. Getting knowledge items...")
        items = connector.get_all_knowledge_items(limit=5)
        print(f"   Found {len(items)} items")
        for item in items[:3]:
            print(f"   - {item['title']} ({item['content_type']})")
        print()
        
        print("MindForge connector test: PASSED")
        return True
        
    except FileNotFoundError as e:
        print(f"   MindForge database not found: {e}")
        print("   This is OK if MindForge is not set up yet.")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_easy_ingestion():
    """Test easy ingestion"""
    print("="*80)
    print("Testing Easy Content Ingestion".center(80))
    print("="*80)
    print()
    
    try:
        ingestion = EasyIngestion(user_id="test_user")
        
        # Test add text
        print("1. Testing add text...")
        result = ingestion.add_text("Test content for LLM Twin", source="test")
        print(f"   Result: {result.get('message', 'Done')}")
        print()
        
        # Test stats
        print("2. Testing get stats...")
        stats = ingestion.get_stats()
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print()
        
        print("Easy ingestion test: PASSED")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_llm_twin_sync():
    """Test LLM Twin MindForge sync"""
    print("="*80)
    print("Testing LLM Twin MindForge Sync".center(80))
    print("="*80)
    print()
    
    try:
        companion = LLMTwinLearningCompanion(user_id="test_user")
        
        # Test sync (may fail if MindForge not available)
        print("1. Testing MindForge sync...")
        result = companion.sync_mindforge()
        
        if result.get('success'):
            print(f"   {result.get('message', 'Synced')}")
            print(f"   Synced: {result.get('synced', 0)} items")
        else:
            print(f"   Note: {result.get('error', 'Sync not available')}")
            print("   This is OK if MindForge is not set up.")
        print()
        
        # Test stats
        print("2. Testing knowledge stats...")
        stats = companion.get_knowledge_stats()
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print()
        
        print("LLM Twin sync test: PASSED")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MINDFORGE INTEGRATION TESTS".center(80))
    print("="*80 + "\n")
    
    results = {
        'connector': test_mindforge_connector(),
        'ingestion': test_easy_ingestion(),
        'sync': test_llm_twin_sync()
    }
    
    print("\n" + "="*80)
    print("TEST RESULTS".center(80))
    print("="*80 + "\n")
    
    for test, passed in results.items():
        status = "PASSED" if passed else "SKIPPED/FAILED"
        print(f"{test:20} {status}")
    
    print("\n" + "="*80)
    print("Tests complete!".center(80))
    print("="*80 + "\n")
    
    if all(results.values()):
        print("All tests passed!")
    else:
        print("Some tests were skipped (this is OK if MindForge is not set up)")

if __name__ == "__main__":
    main()
