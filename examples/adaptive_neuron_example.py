"""
Adaptive Learning Neuron Example
Demonstrates how the adaptive neuron learns and adapts over time
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from ai.adaptive_neuron import AdaptiveNeuron, AdaptiveNeuralNetwork


def demo_single_neuron():
    """Demonstrate single adaptive neuron learning"""
    print("="*70)
    print("ADAPTIVE NEURON DEMONSTRATION")
    print("="*70)
    
    # Initialize
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    neuron = AdaptiveNeuron(kernel, name="LearningNeuron")
    
    print("\n[+] Neuron initialized")
    print(f"    Learning rate: {neuron.learning_rate}")
    print(f"    Momentum: {neuron.momentum}")
    
    # Training phase
    print("\n" + "="*70)
    print("PHASE 1: LEARNING")
    print("="*70)
    
    training_examples = [
        ("machine learning", "artificial intelligence", 1.0),
        ("deep learning", "neural networks", 1.0),
        ("Python programming", "data science", 1.0),
        ("quantum computing", "quantum mechanics", 1.0),
        ("natural language processing", "text analysis", 1.0),
        ("machine learning", "data science", 1.0),  # Reinforce connection
        ("deep learning", "artificial intelligence", 1.0),  # Reinforce connection
    ]
    
    print("\nTraining neuron on examples...")
    for i, (input_text, output, reward) in enumerate(training_examples, 1):
        neuron.learn(input_text, output, reward)
        print(f"  [{i}/{len(training_examples)}] Learned: '{input_text}' -> '{output}'")
    
    # Testing phase
    print("\n" + "="*70)
    print("PHASE 2: TESTING")
    print("="*70)
    
    test_inputs = [
        "machine learning",
        "deep learning",
        "Python",
        "quantum",
        "unknown concept"
    ]
    
    print("\nTesting neuron responses...")
    for test_input in test_inputs:
        result = neuron.activate(test_input)
        print(f"\n  Input: '{test_input}'")
        print(f"  Response: {result['response']}")
        print(f"  Activation: {result['activation_strength']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        if result['related_concepts']:
            top_concepts = list(result['related_concepts'].items())[:3]
            print(f"  Related: {', '.join([c for c, _ in top_concepts])}")
    
    # Reinforcement learning
    print("\n" + "="*70)
    print("PHASE 3: REINFORCEMENT")
    print("="*70)
    
    print("\nReinforcing correct responses...")
    neuron.reinforce("machine learning", was_correct=True)
    neuron.reinforce("deep learning", was_correct=True)
    neuron.reinforce("unknown concept", was_correct=False)
    
    # Show statistics
    print("\n" + "="*70)
    print("NEURON STATISTICS")
    print("="*70)
    
    stats = neuron.get_stats()
    print(f"\n  Total activations: {stats['total_activations']}")
    print(f"  Successful predictions: {stats['successful_predictions']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Learning epochs: {stats['learning_epochs']}")
    print(f"  Learned concepts: {stats['learned_concepts']}")
    print(f"  Total weights: {stats['total_weights']}")
    print(f"  Current learning rate: {stats['learning_rate']:.4f}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


def demo_neural_network():
    """Demonstrate network of adaptive neurons"""
    print("="*70)
    print("ADAPTIVE NEURAL NETWORK DEMONSTRATION")
    print("="*70)
    
    # Initialize network
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    network = AdaptiveNeuralNetwork(kernel)
    
    # Create specialized neurons
    ml_neuron = network.add_neuron("MLNeuron")
    coding_neuron = network.add_neuron("CodingNeuron")
    science_neuron = network.add_neuron("ScienceNeuron")
    
    print("\n[+] Created neural network with 3 specialized neurons:")
    print("    - MLNeuron (machine learning)")
    print("    - CodingNeuron (programming)")
    print("    - ScienceNeuron (science topics)")
    
    # Train each neuron on specialized data
    print("\n" + "="*70)
    print("TRAINING SPECIALIZED NEURONS")
    print("="*70)
    
    # ML Neuron training
    ml_examples = [
        ("machine learning", "AI", 1.0),
        ("neural networks", "deep learning", 1.0),
        ("supervised learning", "training data", 1.0),
    ]
    for inp, out, reward in ml_examples:
        ml_neuron.learn(inp, out, reward)
    
    # Coding Neuron training
    coding_examples = [
        ("Python", "programming language", 1.0),
        ("functions", "code reuse", 1.0),
        ("debugging", "fixing errors", 1.0),
    ]
    for inp, out, reward in coding_examples:
        coding_neuron.learn(inp, out, reward)
    
    # Science Neuron training
    science_examples = [
        ("quantum", "physics", 1.0),
        ("atoms", "molecules", 1.0),
        ("experiments", "hypothesis", 1.0),
    ]
    for inp, out, reward in science_examples:
        science_neuron.learn(inp, out, reward)
    
    print("\n[+] All neurons trained")
    
    # Connect neurons
    network.connect("MLNeuron", "CodingNeuron", weight=0.6)  # ML uses coding
    network.connect("CodingNeuron", "ScienceNeuron", weight=0.3)  # Some overlap
    
    print("\n[+] Neurons connected")
    
    # Test network
    print("\n" + "="*70)
    print("TESTING NETWORK")
    print("="*70)
    
    test_inputs = [
        "machine learning algorithms",
        "Python functions",
        "quantum physics",
        "neural network programming"
    ]
    
    for test_input in test_inputs:
        print(f"\n  Input: '{test_input}'")
        results = network.activate_network(test_input)
        
        for neuron_name, result in results.items():
            if result['activation_strength'] > 0.3:
                print(f"    {neuron_name}: {result['activation_strength']:.3f} - {result['response'][:60]}...")
    
    # Network statistics
    print("\n" + "="*70)
    print("NETWORK STATISTICS")
    print("="*70)
    
    net_stats = network.get_network_stats()
    print(f"\n  Neurons: {net_stats['num_neurons']}")
    print(f"  Connections: {net_stats['num_connections']}")
    
    for neuron_name, neuron_stats in net_stats['neurons'].items():
        print(f"\n  {neuron_name}:")
        print(f"    Activations: {neuron_stats['total_activations']}")
        print(f"    Learned concepts: {neuron_stats['learned_concepts']}")
        print(f"    Success rate: {neuron_stats['success_rate']:.2%}")
    
    print("\n" + "="*70)
    print("NETWORK DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


def demo_adaptive_learning():
    """Demonstrate adaptive learning over time"""
    print("="*70)
    print("ADAPTIVE LEARNING OVER TIME")
    print("="*70)
    
    kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    neuron = AdaptiveNeuron(kernel, name="AdaptiveLearner")
    
    # Initial state
    print("\n[Initial State]")
    stats = neuron.get_stats()
    print(f"  Learning rate: {stats['learning_rate']:.4f}")
    print(f"  Learned concepts: {stats['learned_concepts']}")
    
    # Learning phase 1
    print("\n[Learning Phase 1: Basic concepts]")
    for i in range(5):
        neuron.learn(f"concept{i}", f"related{i}", reward=1.0)
    stats = neuron.get_stats()
    print(f"  Learned concepts: {stats['learned_concepts']}")
    
    # Test
    result = neuron.activate("concept0")
    print(f"  Test 'concept0': activation={result['activation_strength']:.3f}")
    
    # Adaptive adjustment based on feedback
    print("\n[Adaptive Adjustment]")
    neuron.adapt({"concept0": 0.8, "concept1": -0.2})  # Positive and negative feedback
    stats = neuron.get_stats()
    print(f"  Learning rate adjusted: {stats['learning_rate']:.4f}")
    
    # Reinforcement
    print("\n[Reinforcement Learning]")
    neuron.reinforce("concept0", was_correct=True)
    neuron.reinforce("concept1", was_correct=False)
    
    # Final state
    print("\n[Final State]")
    stats = neuron.get_stats()
    print(f"  Total activations: {stats['total_activations']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Learning epochs: {stats['learning_epochs']}")
    
    print("\n" + "="*70)
    print("ADAPTIVE LEARNING DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


def main():
    """Run all demonstrations"""
    try:
        demo_single_neuron()
        demo_neural_network()
        demo_adaptive_learning()
        
        print("="*70)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  [+] Single neuron learning")
        print("  [+] Neural network of multiple neurons")
        print("  [+] Adaptive learning over time")
        print("  [+] Reinforcement learning")
        print("  [+] Weight adaptation")
        print("  [+] Performance tracking")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")


if __name__ == "__main__":
    main()
