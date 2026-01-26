"""
AI Learning Companion Demo

Shows how the learning companion works without requiring interactive input.
"""

from ai_learning_companion import LearningCompanion


def demo():
    """Demo the AI Learning Companion"""
    print("\n" + "="*80)
    print("AI LEARNING COMPANION - Demo".center(80))
    print("="*80)
    print("\nThis is your personal AI learning companion built on the ML Organism.")
    print("It helps you learn ML and AI concepts, answers questions, and guides your learning.")
    print("\n" + "="*80 + "\n")
    
    # Initialize companion
    companion = LearningCompanion()
    
    # Demo 1: Learn a concept
    print("="*80)
    print("DEMO 1: Learning a Concept".center(80))
    print("="*80)
    result = companion.learn_concept('classification')
    companion._print_learning_result(result)
    
    # Demo 2: Ask a question
    print("="*80)
    print("DEMO 2: Asking a Question".center(80))
    print("="*80)
    result = companion.answer_question("What is machine learning?")
    companion._print_answer(result)
    
    # Demo 3: Get learning path
    print("="*80)
    print("DEMO 3: Learning Path".center(80))
    print("="*80)
    result = companion.suggest_learning_path('ml_fundamentals')
    companion._print_path(result)
    
    # Demo 4: Check progress
    print("="*80)
    print("DEMO 4: Learning Progress".center(80))
    print("="*80)
    result = companion.assess_progress()
    companion._print_progress(result)
    
    print("\n" + "="*80)
    print("HOW TO USE".center(80))
    print("="*80)
    print("\nTo use interactively, run:")
    print("  python ai_learning_companion.py")
    print("\nCommands:")
    print("  learn <concept>  - Learn a concept")
    print("  ask <question>   - Ask a question")
    print("  path <goal>      - Get a learning path")
    print("  progress         - See your progress")
    print("  quit             - Exit")
    print("\n" + "="*80)
    print("\nThe companion uses the ML Organism to:")
    print("  • Remember what you've learned")
    print("  • Discover related concepts")
    print("  • Reason about your questions")
    print("  • Learn from your interactions")
    print("  • Adapt to your learning style")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    demo()
