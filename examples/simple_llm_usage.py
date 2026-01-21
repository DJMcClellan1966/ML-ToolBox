"""
Simple LLM Usage Examples
Practical examples of using the LLM system for your projects
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from ai import CompleteAISystem


def example_1_simple_generation():
    """Example 1: Simple text generation"""
    print("="*60)
    print("EXAMPLE 1: Simple Text Generation")
    print("="*60)
    
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    llm = StandaloneQuantumLLM(kernel=kernel)
    
    prompt = "Explain what machine learning is in simple terms"
    result = llm.generate_grounded(
        prompt=prompt,
        max_length=150,
        temperature=0.7
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated:\n{result['text']}\n")


def example_2_code_documentation():
    """Example 2: Generate code documentation"""
    print("="*60)
    print("EXAMPLE 2: Code Documentation Generator")
    print("="*60)
    
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    llm = StandaloneQuantumLLM(kernel=kernel)
    
    code = """
def calculate_total(items):
    return sum(item.price * item.quantity for item in items)
"""
    
    prompt = f"Write clear documentation for this Python function:\n{code}"
    try:
        result = llm.generate_grounded(prompt, max_length=200)
        text = result.get('text') or result.get('generated_text') or str(result)
        
        print(f"\nCode:\n{code}")
        print(f"\nDocumentation:\n{text}\n")
    except Exception as e:
        print(f"Error: {e}")


def example_3_semantic_search():
    """Example 3: Semantic search through your code/documents"""
    print("="*60)
    print("EXAMPLE 3: Semantic Search")
    print("="*60)
    
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    
    # Your project files/notes
    documents = [
        "The authentication system uses JWT tokens stored in HTTP-only cookies",
        "The database is PostgreSQL with SQLAlchemy ORM for data access",
        "API endpoints are defined using FastAPI decorators",
        "Frontend components are built with React and TypeScript",
        "Error handling is centralized in a custom exception handler"
    ]
    
    query = "How does user authentication work?"
    results = kernel.find_similar(query, documents, top_k=3)
    
    print(f"\nQuery: {query}\n")
    print("Top Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.3f}] {doc}\n")


def example_4_knowledge_base():
    """Example 4: Build and query a knowledge base"""
    print("="*60)
    print("EXAMPLE 4: Knowledge Base")
    print("="*60)
    
    ai = CompleteAISystem(use_llm=True)
    
    # Add your project knowledge
    project_knowledge = [
        "The backend API is built with FastAPI framework",
        "Authentication uses JWT tokens with 24-hour expiration",
        "Database uses PostgreSQL with connection pooling",
        "Frontend is a React SPA with React Router for navigation",
        "Deployment uses Docker containers on AWS ECS"
    ]
    
    print("\nAdding project knowledge...")
    for knowledge in project_knowledge:
        ai.knowledge_graph.add_document(knowledge)
        print(f"  + {knowledge}")
    
    # Query the knowledge base
    query = "What is the deployment setup?"
    results = ai.search.semantic_search(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print("\nResults:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}\n")


def example_5_conversational_ai():
    """Example 5: Conversational AI with context"""
    print("="*60)
    print("EXAMPLE 5: Conversational AI")
    print("="*60)
    
    ai = CompleteAISystem(use_llm=True)
    
    # Set up context
    context = """
    This project is a task management application.
    It uses FastAPI backend, React frontend, and PostgreSQL database.
    Features include user authentication, task CRUD operations, and notifications.
    """
    
    ai.knowledge_graph.add_document(context)
    
    # Have a conversation
    questions = [
        "What technologies are used in this project?",
        "What are the main features?"
    ]
    
    for question in questions:
        response = ai.conversation.chat(question, context=[context])
        print(f"\nQ: {question}")
        print(f"A: {response}\n")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("PRACTICAL LLM USAGE EXAMPLES")
    print("How to use this as your own LLM system")
    print("="*60)
    
    try:
        example_1_simple_generation()
        example_2_code_documentation()
        example_3_semantic_search()
        example_4_knowledge_base()
        example_5_conversational_ai()
        
        print("="*60)
        print("All examples completed!")
        print("="*60)
        print("\nYou can use this system for:")
        print("  - Code documentation generation")
        print("  - Semantic search through your codebase")
        print("  - Project knowledge bases")
        print("  - Conversational AI for your projects")
        print("  - Text generation and summarization")
        print("\nAll without external APIs or costs!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")


if __name__ == "__main__":
    main()
