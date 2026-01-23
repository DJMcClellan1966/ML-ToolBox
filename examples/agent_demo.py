"""
AI Agent Demo
Demonstrates the AI agent generating ML code
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox.ai_agent import MLCodeAgent

def demo_agent():
    """Demonstrate AI Agent capabilities"""
    print("="*80)
    print("AI AGENT DEMO - Pattern-Based Code Generation")
    print("="*80)
    print()
    
    # Initialize agent
    print("Initializing AI Agent...")
    agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)
    print("✅ Agent ready!")
    print()
    
    # Demo 1: Simple classification
    print("="*80)
    print("DEMO 1: Simple Classification")
    print("="*80)
    print("Task: 'Classify data into 2 classes'")
    print()
    
    result1 = agent.build("Classify data into 2 classes")
    
    if result1['success']:
        print("✅ Code generated successfully!")
        print(f"Iterations: {result1['iterations']}")
        print("\nGenerated Code:")
        print("-" * 80)
        print(result1['code'])
        print("-" * 80)
        if result1.get('output'):
            print("\nExecution Output:")
            print(result1['output'])
    else:
        print(f"❌ Generation failed: {result1.get('error', 'Unknown')}")
    
    print()
    
    # Demo 2: Regression
    print("="*80)
    print("DEMO 2: Regression")
    print("="*80)
    print("Task: 'Predict continuous values'")
    print()
    
    result2 = agent.build("Predict continuous values")
    
    if result2['success']:
        print("✅ Code generated successfully!")
        print(f"Iterations: {result2['iterations']}")
        print("\nGenerated Code:")
        print("-" * 80)
        print(result2['code'][:500] + "..." if len(result2['code']) > 500 else result2['code'])
        print("-" * 80)
    else:
        print(f"❌ Generation failed: {result2.get('error', 'Unknown')}")
    
    print()
    
    # Show agent statistics
    print("="*80)
    print("AGENT STATISTICS")
    print("="*80)
    
    history = agent.get_history()
    print(f"Tasks attempted: {len(history)}")
    successful = sum(1 for h in history if h.get('success', False))
    print(f"Successful: {successful}")
    print(f"Failed: {len(history) - successful}")
    
    graph_stats = agent.graph.get_statistics()
    print(f"\nPattern Graph:")
    print(f"  Patterns: {graph_stats['total_patterns']}")
    print(f"  Relationships: {graph_stats['total_relationships']}")
    print(f"  Compositions: {graph_stats['total_compositions']}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nThe agent uses pattern composition instead of training on")
    print("billions of examples - innovative approach!")


if __name__ == '__main__':
    demo_agent()
