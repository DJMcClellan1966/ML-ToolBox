"""
Adaptive Learning Neuron
A neural-like component that learns and adapts using quantum kernel semantics
Combines quantum kernel + AI system + LLM for adaptive learning
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import time
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class AdaptiveNeuron:
    """
    Adaptive Learning Neuron
    
    A neural-like component that:
    - Learns from input-output pairs
    - Adapts weights based on feedback
    - Uses quantum kernel for semantic understanding
    - Builds associations between concepts
    - Improves over time with experience
    """
    
    def __init__(self, kernel, name: str = "AdaptiveNeuron"):
        """
        Initialize adaptive neuron
        
        Args:
            kernel: Quantum kernel instance for semantic operations
            name: Name identifier for this neuron
        """
        self.kernel = kernel
        self.name = name
        
        # Neural weights (semantic associations)
        self.weights = {}  # concept -> {related_concept: weight}
        self.bias = defaultdict(float)  # concept -> bias value
        
        # Learning parameters
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay_rate = 0.95  # Weight decay over time
        
        # Memory and experience
        self.experience = []  # List of (input, output, reward) tuples
        self.activation_history = defaultdict(list)  # Track activations
        
        # Performance tracking
        self.total_activations = 0
        self.successful_predictions = 0
        self.learning_epochs = 0
        
        # Semantic associations (learned patterns)
        self.semantic_associations = defaultdict(dict)
        
        logger.info(f"AdaptiveNeuron '{name}' initialized")
    
    def activate(self, input_text: str, context: Optional[List[str]] = None) -> Dict:
        """
        Activate neuron with input
        
        Process:
        1. Understand input semantically
        2. Find related concepts from learned associations
        3. Compute weighted response
        4. Return activation result
        
        Args:
            input_text: Input to process
            context: Optional context for better understanding
            
        Returns:
            Dictionary with activation result, confidence, and related concepts
        """
        self.total_activations += 1
        
        # Step 1: Semantic embedding
        input_embedding = self.kernel.embed(input_text)
        
        # Step 2: Find related concepts from learned weights
        related_concepts = self._find_related_concepts(input_text)
        
        # Step 3: Compute activation strength
        activation_strength = self._compute_activation(input_embedding, related_concepts)
        
        # Step 4: Generate response based on learned patterns
        response = self._generate_response(input_text, related_concepts, activation_strength)
        
        # Track activation
        self.activation_history[input_text].append({
            'timestamp': datetime.utcnow().isoformat(),
            'strength': activation_strength,
            'related_concepts': list(related_concepts.keys())
        })
        
        return {
            'input': input_text,
            'response': response,
            'activation_strength': activation_strength,
            'related_concepts': related_concepts,
            'confidence': min(activation_strength, 1.0),
            'neuron_name': self.name
        }
    
    def _find_related_concepts(self, input_text: str) -> Dict[str, float]:
        """Find concepts related to input based on learned weights"""
        related = {}
        
        # Check learned semantic associations
        for concept, associations in self.semantic_associations.items():
            # Compute similarity to input
            similarity = self.kernel.similarity(input_text, concept)
            
            if similarity > 0.5:  # Threshold for relevance
                # Weight by learned associations
                for related_concept, weight in associations.items():
                    if related_concept not in related:
                        related[related_concept] = 0.0
                    related[related_concept] += similarity * weight
        
        # Sort by strength
        return dict(sorted(related.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _compute_activation(self, embedding: np.ndarray, related_concepts: Dict) -> float:
        """Compute activation strength"""
        if not related_concepts:
            return 0.0
        
        # Base activation from related concepts
        base_activation = sum(related_concepts.values()) / len(related_concepts)
        
        # Add bias
        bias_value = sum(self.bias.values()) / max(len(self.bias), 1)
        
        # Combine
        activation = base_activation * 0.7 + bias_value * 0.3
        
        return float(np.clip(activation, 0.0, 1.0))
    
    def _generate_response(self, input_text: str, related_concepts: Dict, 
                          activation_strength: float) -> str:
        """Generate response based on learned patterns"""
        if not related_concepts:
            return f"I'm learning about '{input_text}'. Can you provide more context?"
        
        # Get top related concepts
        top_concepts = list(related_concepts.keys())[:3]
        
        # Generate response based on learned associations
        if activation_strength > 0.7:
            response = f"Based on what I've learned, '{input_text}' relates to: {', '.join(top_concepts[:2])}"
        elif activation_strength > 0.4:
            response = f"I see some connection between '{input_text}' and: {top_concepts[0]}"
        else:
            response = f"I'm still learning about '{input_text}'. It might relate to: {top_concepts[0] if top_concepts else 'unknown concepts'}"
        
        return response
    
    def learn(self, input_text: str, expected_output: str, reward: float = 1.0):
        """
        Learn from input-output pair with reward
        
        Args:
            input_text: Input example
            expected_output: Expected/desired output
            reward: Reward signal (1.0 = correct, 0.0 = incorrect, 0.5 = partial)
        """
        self.learning_epochs += 1
        
        # Store experience
        self.experience.append({
            'input': input_text,
            'output': expected_output,
            'reward': reward,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Extract concepts from input and output
        input_concepts = self._extract_concepts(input_text)
        output_concepts = self._extract_concepts(expected_output)
        
        # Update weights based on reward
        for input_concept in input_concepts:
            for output_concept in output_concepts:
                # Initialize if needed
                if input_concept not in self.semantic_associations:
                    self.semantic_associations[input_concept] = {}
                
                # Update weight (reinforcement learning style)
                current_weight = self.semantic_associations[input_concept].get(output_concept, 0.0)
                
                # Weight update: w = w + learning_rate * reward * (target - current)
                weight_update = self.learning_rate * reward * (1.0 - current_weight)
                new_weight = current_weight + weight_update
                
                # Apply momentum
                if input_concept in self.weights and output_concept in self.weights[input_concept]:
                    momentum_update = self.momentum * (new_weight - current_weight)
                    new_weight += momentum_update
                
                # Store updated weight
                self.semantic_associations[input_concept][output_concept] = np.clip(new_weight, 0.0, 1.0)
        
        # Update bias based on reward
        for concept in input_concepts:
            self.bias[concept] += self.learning_rate * reward * 0.1
            self.bias[concept] = np.clip(self.bias[concept], -1.0, 1.0)
        
        # Track successful learning
        if reward > 0.5:
            self.successful_predictions += 1
        
        logger.info(f"Neuron '{self.name}' learned from example (reward: {reward:.2f})")
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction (could be enhanced)
        words = text.lower().split()
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 
                     'with', 'by', 'from', 'as', 'this', 'that', 'these', 'those'}
        
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also consider phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 5:
                phrases.append(phrase)
        
        return concepts + phrases[:3]  # Limit phrases
    
    def adapt(self, feedback: Dict[str, float]):
        """
        Adapt based on feedback
        
        Args:
            feedback: Dictionary of {concept: feedback_score}
        """
        for concept, score in feedback.items():
            if concept in self.bias:
                # Adjust bias based on feedback
                self.bias[concept] += self.learning_rate * score
                self.bias[concept] = np.clip(self.bias[concept], -1.0, 1.0)
            
            # Adjust learning rate based on performance
            if score < 0:
                # Decrease learning rate if getting negative feedback
                self.learning_rate *= 0.99
            elif score > 0:
                # Slight increase for positive feedback
                self.learning_rate = min(self.learning_rate * 1.01, 0.5)
    
    def reinforce(self, input_text: str, was_correct: bool):
        """
        Reinforce learning based on correctness
        
        Args:
            input_text: Input that was processed
            was_correct: Whether the response was correct
        """
        reward = 1.0 if was_correct else -0.5
        
        # Find recent activation for this input
        if input_text in self.activation_history:
            recent_activation = self.activation_history[input_text][-1] if self.activation_history[input_text] else None
            
            if recent_activation:
                # Reinforce the concepts that were activated
                for concept in recent_activation.get('related_concepts', []):
                    if concept in self.bias:
                        self.bias[concept] += self.learning_rate * reward * 0.2
                        self.bias[concept] = np.clip(self.bias[concept], -1.0, 1.0)
    
    def decay_weights(self):
        """Apply weight decay to prevent overfitting"""
        for concept in self.semantic_associations:
            for related_concept in self.semantic_associations[concept]:
                self.semantic_associations[concept][related_concept] *= self.decay_rate
    
    def get_stats(self) -> Dict:
        """Get neuron statistics"""
        total_weights = sum(len(assoc) for assoc in self.semantic_associations.values())
        
        return {
            'name': self.name,
            'total_activations': self.total_activations,
            'successful_predictions': self.successful_predictions,
            'success_rate': self.successful_predictions / max(self.total_activations, 1),
            'learning_epochs': self.learning_epochs,
            'total_weights': total_weights,
            'learned_concepts': len(self.semantic_associations),
            'learning_rate': self.learning_rate,
            'experience_count': len(self.experience)
        }
    
    def save_state(self, filepath: str):
        """Save neuron state to file"""
        state = {
            'name': self.name,
            'semantic_associations': {k: dict(v) for k, v in self.semantic_associations.items()},
            'bias': dict(self.bias),
            'learning_rate': self.learning_rate,
            'stats': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Neuron state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load neuron state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.name = state.get('name', self.name)
        self.semantic_associations = defaultdict(dict, 
            {k: defaultdict(float, v) for k, v in state.get('semantic_associations', {}).items()})
        self.bias = defaultdict(float, state.get('bias', {}))
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        
        logger.info(f"Neuron state loaded from {filepath}")


class AdaptiveNeuralNetwork:
    """
    Network of adaptive neurons
    
    Multiple neurons working together for complex learning
    """
    
    def __init__(self, kernel):
        """
        Initialize neural network
        
        Args:
            kernel: Quantum kernel instance
        """
        self.kernel = kernel
        self.neurons = {}
        self.connections = defaultdict(dict)  # neuron1 -> {neuron2: weight}
    
    def add_neuron(self, name: str) -> AdaptiveNeuron:
        """Add a new neuron to the network"""
        neuron = AdaptiveNeuron(self.kernel, name=name)
        self.neurons[name] = neuron
        return neuron
    
    def connect(self, neuron1_name: str, neuron2_name: str, weight: float = 0.5):
        """Connect two neurons"""
        if neuron1_name in self.neurons and neuron2_name in self.neurons:
            self.connections[neuron1_name][neuron2_name] = weight
    
    def activate_network(self, input_text: str) -> Dict:
        """Activate network with input"""
        results = {}
        
        # Activate all neurons
        for name, neuron in self.neurons.items():
            results[name] = neuron.activate(input_text)
        
        # Propagate through connections
        for neuron1_name, connections in self.connections.items():
            neuron1_result = results[neuron1_name]
            for neuron2_name, weight in connections.items():
                if neuron2_name in results:
                    # Influence neuron2 based on neuron1's activation
                    influence = neuron1_result['activation_strength'] * weight
                    # Could modify neuron2's result here
        
        return results
    
    def get_network_stats(self) -> Dict:
        """Get statistics for entire network"""
        return {
            'num_neurons': len(self.neurons),
            'num_connections': sum(len(conns) for conns in self.connections.values()),
            'neurons': {name: neuron.get_stats() for name, neuron in self.neurons.items()}
        }
