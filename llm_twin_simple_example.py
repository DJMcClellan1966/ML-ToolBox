"""
LLM Twin Learning Companion - Simple Example

This is a minimal example showing how to use the LLM Twin Learning Companion.
Run this to see it in action!
"""

from llm_twin_learning_companion import LLMTwinLearningCompanion

def main():
    print("="*80)
    print("LLM TWIN LEARNING COMPANION - SIMPLE EXAMPLE".center(80))
    print("="*80)
    print()
    
    # Step 1: Create companion
    print("Step 1: Creating companion...")
    companion = LLMTwinLearningCompanion(user_id="example_user")
    print("Companion created!")
    print()
    
    # Step 2: Greet user
    print("Step 2: Getting greeting...")
    greeting = companion.greet_user()
    print(greeting)
    print()
    
    # Step 3: Add some content
    print("Step 3: Adding content to knowledge base...")
    content = """
    Machine learning is a subset of artificial intelligence that focuses
    on algorithms that can learn from data without being explicitly programmed.
    Neural networks are a type of machine learning model inspired by the brain.
    """
    result = companion.ingest_text(content, source="example_notes")
    print(f"{result['message']}")
    print()
    
    # Step 4: Learn a concept
    print("Step 4: Learning a concept...")
    result = companion.learn_concept_twin("machine learning")
    print(f"Concept: {result['concept']}")
    print(f"Explanation: {result['explanation'][:200]}...")
    print()
    
    # Step 5: Ask a question
    print("Step 5: Asking a question...")
    result = companion.answer_question_twin("What is machine learning?")
    print(f"Question: What is machine learning?")
    print(f"Answer: {result['answer'][:200]}...")
    print()
    
    # Step 6: Continue conversation
    print("Step 6: Continuing conversation...")
    result = companion.continue_conversation("Can you tell me more about neural networks?")
    print(f"You: Can you tell me more about neural networks?")
    print(f"Companion: {result['answer'][:200]}...")
    print()
    
    # Step 7: Get profile
    print("Step 7: Getting user profile...")
    profile = companion.get_user_profile()
    print(f"User ID: {profile['user_id']}")
    print(f"Topics learned: {profile['conversation_stats']['topics_learned']}")
    print(f"Total interactions: {profile['conversation_stats']['total_interactions']}")
    print()
    
    # Step 8: Get knowledge stats
    print("Step 8: Getting knowledge base statistics...")
    stats = companion.get_knowledge_stats()
    print(f"Total documents: {stats['total_documents']}")
    if stats.get('sources'):
        print("Sources:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count} document(s)")
    print()
    
    # Step 9: Save session
    print("Step 9: Saving session...")
    companion.save_session()
    print("Session saved! The companion will remember you next time.")
    print()
    
    print("="*80)
    print("EXAMPLE COMPLETE!".center(80))
    print("="*80)
    print()
    print("Next steps:")
    print("  - Try the web UI: python llm_twin_web_ui.py")
    print("  - Read examples: LLM_TWIN_EXAMPLES.md")
    print("  - See integration guide: LLM_TWIN_INTEGRATION.md")
    print("  - Check API reference: LLM_TWIN_API.md")
    print()

if __name__ == "__main__":
    main()
