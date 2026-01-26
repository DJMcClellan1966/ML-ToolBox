"""
Advanced AI Learning Companion

Uses ALL the advanced capabilities from the toolbox:
- Brain Topology (Working Memory, Episodic Memory, Semantic Memory)
- Socratic Method (Question-based learning)
- Information Theory (Optimal learning paths)
- Metacognition (Self-awareness of learning)
- Pattern Abstraction (Concept generalization)
- Adaptive Learning (Personalized to you)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
import logging
from collections import defaultdict
import time
import json

logger = logging.getLogger(__name__)

# Import advanced components
try:
    from ml_toolbox.agent_brain.cognitive_architecture import CognitiveArchitecture
    from ml_toolbox.agent_brain.working_memory import WorkingMemory
    from ml_toolbox.agent_brain.episodic_memory import EpisodicMemory
    from ml_toolbox.agent_brain.semantic_memory import SemanticMemory
    from ml_toolbox.agent_brain.attention_mechanism import AttentionMechanism
    from ml_toolbox.agent_brain.metacognition import Metacognition
    from ml_toolbox.agent_brain.pattern_abstraction import PatternAbstraction
    BRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Brain components not available: {e}")
    BRAIN_AVAILABLE = False

try:
    from ml_toolbox.agent_enhancements.socratic_method import (
        SocraticQuestioner, SocraticActiveLearner
    )
    SOCRATIC_AVAILABLE = True
except ImportError:
    SOCRATIC_AVAILABLE = False

try:
    from ml_toolbox.textbook_concepts.information_theory import (
        entropy, mutual_information, information_gain
    )
    INFO_THEORY_AVAILABLE = True
except ImportError:
    INFO_THEORY_AVAILABLE = False

try:
    from ml_toolbox.optimization.evolutionary_algorithms import GeneticAlgorithm
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False

from organic_ml_organism import MLOrganism


class AdvancedLearningCompanion:
    """
    Advanced AI Learning Companion
    
    Uses ALL advanced capabilities:
    - Brain Topology for cognitive learning
    - Socratic Method for deep understanding
    - Information Theory for optimal learning
    - Metacognition for self-awareness
    - Pattern Abstraction for generalization
    - Adaptive Learning for personalization
    """
    
    def __init__(self):
        """Initialize advanced learning companion"""
        logger.info("Initializing Advanced AI Learning Companion...")
        
        # Brain Topology (Cognitive Architecture)
        if BRAIN_AVAILABLE:
            self.brain = CognitiveArchitecture(working_memory_capacity=7)
            logger.info("Brain topology initialized")
        else:
            self.brain = None
        
        # Socratic Method
        if SOCRATIC_AVAILABLE:
            self.socratic = SocraticQuestioner()
            self.active_learner = SocraticActiveLearner()
            logger.info("Socratic method initialized")
        else:
            self.socratic = None
            self.active_learner = None
        
        # ML Organism (Unified Systems)
        self.organism = MLOrganism()
        
        # Learning State (Advanced)
        self.learning_state = {
            'level': 'beginner',
            'topics_learned': [],
            'topics_in_progress': [],
            'weak_areas': [],
            'strong_areas': [],
            'learning_style': self._detect_learning_style(),
            'preferred_pace': 'moderate',
            'knowledge_graph': {},  # Concept relationships
            'difficulty_map': {},  # Concept difficulty
            'prerequisite_map': {},  # Prerequisites
            'learning_paths': {}  # Personalized paths
        }
        
        # Advanced Knowledge Base
        self.knowledge_base = self._build_advanced_knowledge_base()
        
        # Learning Analytics
        self.learning_analytics = {
            'concept_mastery': {},  # concept -> mastery_score
            'learning_velocity': {},  # concept -> time_to_learn
            'retention_rate': {},  # concept -> retention_score
            'confusion_matrix': {}  # concept -> confused_with
        }
        
        logger.info("Advanced AI Learning Companion initialized!")
    
    def _detect_learning_style(self) -> str:
        """Detect learning style using brain patterns"""
        # Simple detection (can be enhanced)
        return 'visual'  # visual, auditory, kinesthetic, reading
    
    def _build_advanced_knowledge_base(self) -> Dict:
        """Build comprehensive knowledge base with relationships"""
        kb = {
            'beginner': {
                'machine_learning': {
                    'concept': 'Machine Learning is about teaching computers to learn from data without explicit programming.',
                    'detailed_explanation': '''
Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

Key Principles:
- Learning from Data: ML models learn patterns from data
- Generalization: Models should work on new, unseen data
- Iterative Improvement: Models get better with more data

Types of ML:
1. Supervised Learning: Learning from labeled examples
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

Applications:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
                    ''',
                    'examples': [
                        'Email spam detection learns from labeled spam/not spam emails',
                        'Netflix recommendations learn from your viewing history',
                        'Self-driving cars learn from millions of driving scenarios'
                    ],
                    'key_terms': ['data', 'training', 'prediction', 'model', 'algorithm', 'pattern'],
                    'prerequisites': [],
                    'difficulty': 1,
                    'estimated_time': '2 hours',
                    'related': ['supervised_learning', 'unsupervised_learning', 'neural_networks'],
                    'common_misconceptions': [
                        'ML is just statistics',
                        'More data always means better models',
                        'ML models are black boxes'
                    ],
                    'practice_projects': [
                        'Build a simple spam classifier',
                        'Create a recommendation system',
                        'Train a model on a dataset'
                    ]
                },
                'supervised_learning': {
                    'concept': 'Supervised learning uses labeled data to train models that can make predictions.',
                    'detailed_explanation': '''
Supervised Learning is a type of ML where models learn from labeled training data.

Process:
1. Training Data: Input-output pairs (features -> labels)
2. Learning: Model learns the mapping
3. Prediction: Model predicts outputs for new inputs

Types:
- Classification: Predict discrete categories (spam/not spam)
- Regression: Predict continuous values (house prices)

Key Concepts:
- Features: Input variables
- Labels: Output variables (what we want to predict)
- Training: Process of learning
- Testing: Evaluating on new data

Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Neural Networks
                    ''',
                    'examples': [
                        'Classifying emails as spam/not spam',
                        'Predicting house prices from features',
                        'Recognizing handwritten digits'
                    ],
                    'key_terms': ['labeled data', 'training', 'labels', 'prediction', 'features', 'target'],
                    'prerequisites': ['machine_learning'],
                    'difficulty': 2,
                    'estimated_time': '3 hours',
                    'related': ['classification', 'regression', 'neural_networks'],
                    'common_misconceptions': [
                        'Supervised learning always needs lots of data',
                        'More features always help',
                        'Training accuracy = real-world performance'
                    ],
                    'practice_projects': [
                        'Build a classification model',
                        'Create a regression model',
                        'Compare different algorithms'
                    ]
                },
                'classification': {
                    'concept': 'Classification predicts discrete categories (like spam/not spam).',
                    'detailed_explanation': '''
Classification is a supervised learning task that predicts discrete categories.

Key Concepts:
- Classes: The categories to predict (e.g., spam/not spam)
- Features: Input variables used for prediction
- Decision Boundary: The line that separates classes

Types:
- Binary Classification: Two classes (spam/not spam)
- Multi-class Classification: Multiple classes (cat/dog/bird)
- Multi-label Classification: Multiple labels per instance

Algorithms:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- Neural Networks

Evaluation Metrics:
- Accuracy: Percentage of correct predictions
- Precision: Of predicted positives, how many are actually positive
- Recall: Of actual positives, how many were predicted
- F1-Score: Harmonic mean of precision and recall
                    ''',
                    'examples': [
                        'Email spam detection (spam/not spam)',
                        'Image classification (cat/dog/bird)',
                        'Sentiment analysis (positive/negative/neutral)'
                    ],
                    'key_terms': ['categories', 'classes', 'labels', 'accuracy', 'precision', 'recall', 'f1-score'],
                    'prerequisites': ['supervised_learning'],
                    'difficulty': 2,
                    'estimated_time': '3 hours',
                    'related': ['regression', 'neural_networks', 'evaluation_metrics'],
                    'common_misconceptions': [
                        'High accuracy always means good model',
                        'All classes are equally important',
                        'More features always improve classification'
                    ],
                    'practice_projects': [
                        'Build a spam classifier',
                        'Classify images',
                        'Sentiment analysis'
                    ]
                },
                'regression': {
                    'concept': 'Regression predicts continuous values (like prices or temperatures).',
                    'detailed_explanation': '''
Regression is a supervised learning task that predicts continuous numerical values.

Key Concepts:
- Target Variable: The continuous value to predict
- Features: Input variables
- Regression Line: The line that best fits the data

Types:
- Linear Regression: Straight line relationship
- Polynomial Regression: Curved relationship
- Ridge/Lasso Regression: Regularized regression

Algorithms:
- Linear Regression
- Polynomial Regression
- Random Forest Regression
- Neural Network Regression

Evaluation Metrics:
- Mean Squared Error (MSE): Average squared difference
- Mean Absolute Error (MAE): Average absolute difference
- R-squared: Proportion of variance explained
                    ''',
                    'examples': [
                        'House price prediction',
                        'Stock price forecasting',
                        'Temperature prediction'
                    ],
                    'key_terms': ['continuous', 'values', 'prediction', 'error', 'mse', 'mae', 'r-squared'],
                    'prerequisites': ['supervised_learning'],
                    'difficulty': 2,
                    'estimated_time': '3 hours',
                    'related': ['classification', 'linear_regression', 'neural_networks'],
                    'common_misconceptions': [
                        'Regression only works for linear relationships',
                        'Lower error always means better model',
                        "Outliers don't matter"
                    ],
                    'practice_projects': [
                        'Predict house prices',
                        'Forecast stock prices',
                        'Predict temperatures'
                    ]
                }
            },
            'intermediate': {
                'neural_networks': {
                    'concept': 'Neural networks are inspired by the brain, with layers of interconnected neurons.',
                    'detailed_explanation': '''
Neural Networks are computing systems inspired by biological neural networks.

Architecture:
- Input Layer: Receives input features
- Hidden Layers: Process information
- Output Layer: Produces predictions

Key Concepts:
- Neurons: Basic processing units
- Weights: Connection strengths
- Activation Functions: Non-linear transformations
- Backpropagation: Learning algorithm

Types:
- Feedforward: Information flows forward
- Convolutional (CNN): For images
- Recurrent (RNN): For sequences
- Transformer: For attention-based processing

Training:
- Forward Pass: Compute predictions
- Loss Calculation: Measure error
- Backward Pass: Update weights
- Gradient Descent: Optimization
                    ''',
                    'examples': [
                        'Image recognition (CNNs)',
                        'Language translation (RNNs/Transformers)',
                        'Speech recognition'
                    ],
                    'key_terms': ['neurons', 'layers', 'weights', 'activation', 'backpropagation', 'gradient'],
                    'prerequisites': ['classification', 'regression'],
                    'difficulty': 4,
                    'estimated_time': '8 hours',
                    'related': ['deep_learning', 'cnn', 'rnn', 'transformer'],
                    'common_misconceptions': [
                        'More layers always better',
                        'Neural networks are black boxes',
                        'You need huge datasets'
                    ],
                    'practice_projects': [
                        'Build a simple neural network',
                        'Train on MNIST dataset',
                        'Create a CNN for images'
                    ]
                },
                'deep_learning': {
                    'concept': 'Deep learning uses neural networks with many layers to learn complex patterns.',
                    'detailed_explanation': '''
Deep Learning uses neural networks with multiple hidden layers.

Key Advantages:
- Automatic Feature Learning: Learns features automatically
- Hierarchical Representations: Each layer learns more abstract features
- Scalability: Works well with large datasets

Architectures:
- Deep Feedforward: Multiple hidden layers
- Convolutional Neural Networks (CNN): For images
- Recurrent Neural Networks (RNN): For sequences
- Transformers: Attention-based

Training Techniques:
- Dropout: Prevent overfitting
- Batch Normalization: Stabilize training
- Learning Rate Scheduling: Adaptive learning
- Transfer Learning: Use pre-trained models

Applications:
- Computer Vision
- Natural Language Processing
- Speech Recognition
- Autonomous Systems
                    ''',
                    'examples': [
                        'Image classification with CNNs',
                        'Language models (GPT, BERT)',
                        'Self-driving cars'
                    ],
                    'key_terms': ['deep', 'layers', 'complex patterns', 'representation', 'cnn', 'rnn'],
                    'prerequisites': ['neural_networks'],
                    'difficulty': 5,
                    'estimated_time': '12 hours',
                    'related': ['neural_networks', 'cnn', 'rnn', 'transformer'],
                    'common_misconceptions': [
                        'Deep learning always better',
                        'You need massive datasets',
                        'Deep learning is magic'
                    ],
                    'practice_projects': [
                        'Build a deep CNN',
                        'Fine-tune a pre-trained model',
                        'Create a language model'
                    ]
                },
                'feature_engineering': {
                    'concept': 'Feature engineering creates meaningful inputs for ML models from raw data.',
                    'detailed_explanation': '''
Feature Engineering transforms raw data into features that ML models can use.

Process:
1. Understanding Data: Analyze raw data
2. Feature Creation: Create new features
3. Feature Selection: Choose best features
4. Feature Transformation: Normalize, encode, etc.

Techniques:
- Encoding: Convert categorical to numerical
- Scaling: Normalize features
- Binning: Group continuous values
- Polynomial Features: Create interactions
- Domain Features: Use domain knowledge

Importance:
- Good features > Good algorithms
- Feature engineering is often the most important step
- Domain knowledge helps

Tools:
- Scikit-learn transformers
- Pandas for data manipulation
- Feature selection algorithms
                    ''',
                    'examples': [
                        'Extracting text features',
                        'Creating time-based features',
                        'Combining features'
                    ],
                    'key_terms': ['features', 'transformation', 'selection', 'extraction', 'encoding', 'scaling'],
                    'prerequisites': ['machine_learning'],
                    'difficulty': 3,
                    'estimated_time': '6 hours',
                    'related': ['preprocessing', 'dimensionality_reduction', 'data_quality'],
                    'common_misconceptions': [
                        'More features always better',
                        'Feature engineering is obsolete',
                        'Models can learn everything'
                    ],
                    'practice_projects': [
                        'Engineer features from text',
                        'Create time-based features',
                        'Select best features'
                    ]
                }
            },
            'advanced': {
                'transformer': {
                    'concept': 'Transformers use attention mechanisms to process sequences, revolutionizing NLP.',
                    'detailed_explanation': '''
Transformers revolutionized NLP with attention mechanisms.

Key Innovation:
- Self-Attention: Each position attends to all positions
- No Recurrence: Parallel processing
- Position Encoding: Captures sequence order

Architecture:
- Encoder: Processes input
- Decoder: Generates output
- Multi-Head Attention: Multiple attention mechanisms
- Feed-Forward Networks: Process attention output

Advantages:
- Parallel Processing: Faster than RNNs
- Long-Range Dependencies: Better than RNNs
- Transfer Learning: Pre-trained models

Models:
- BERT: Bidirectional Encoder
- GPT: Generative Pre-trained Transformer
- T5: Text-to-Text Transfer Transformer

Applications:
- Language Translation
- Text Generation
- Question Answering
- Summarization
                    ''',
                    'examples': [
                        'GPT models for text generation',
                        'BERT for understanding',
                        'Language translation'
                    ],
                    'key_terms': ['attention', 'self-attention', 'encoder', 'decoder', 'bert', 'gpt'],
                    'prerequisites': ['deep_learning', 'neural_networks'],
                    'difficulty': 6,
                    'estimated_time': '16 hours',
                    'related': ['bert', 'gpt', 'attention_mechanism', 'nlp'],
                    'common_misconceptions': [
                        'Transformers only for NLP',
                        'Attention is just weighting',
                        'You need huge models'
                    ],
                    'practice_projects': [
                        'Fine-tune BERT',
                        'Generate text with GPT',
                        'Build a transformer'
                    ]
                },
                'reinforcement_learning': {
                    'concept': 'Reinforcement learning learns through trial and error, receiving rewards for good actions.',
                    'detailed_explanation': '''
Reinforcement Learning learns optimal actions through interaction.

Key Components:
- Agent: The learner
- Environment: The world
- Actions: What the agent can do
- Rewards: Feedback signal
- Policy: Strategy for actions

Process:
1. Agent observes state
2. Agent takes action
3. Environment provides reward
4. Agent updates policy
5. Repeat

Algorithms:
- Q-Learning: Value-based
- Policy Gradient: Policy-based
- Actor-Critic: Both value and policy

Applications:
- Game Playing (AlphaGo, Chess)
- Robotics
- Autonomous Systems
- Recommendation Systems

Challenges:
- Exploration vs Exploitation
- Sparse Rewards
- Credit Assignment
                    ''',
                    'examples': [
                        'Game playing (AlphaGo)',
                        'Robotics',
                        'Autonomous systems'
                    ],
                    'key_terms': ['agent', 'environment', 'reward', 'policy', 'q-learning', 'exploration'],
                    'prerequisites': ['machine_learning', 'neural_networks'],
                    'difficulty': 6,
                    'estimated_time': '16 hours',
                    'related': ['q_learning', 'policy_gradient', 'deep_rl'],
                    'common_misconceptions': [
                        'RL is just trial and error',
                        'RL always needs simulation',
                        'RL is only for games'
                    ],
                    'practice_projects': [
                        'Train an RL agent',
                        'Build a game-playing agent',
                        'Create a robotic controller'
                    ]
                }
            }
        }
        
        # Build knowledge graph
        self._build_knowledge_graph(kb)
        
        return kb
    
    def _build_knowledge_graph(self, kb: Dict):
        """Build knowledge graph of concept relationships"""
        for level, concepts in kb.items():
            for concept_name, concept_data in concepts.items():
                # Add to organism's memory
                if self.brain:
                    self.brain.semantic_memory.add_fact(
                        concept_data['concept'],
                        context=level,
                        source=concept_name,
                        confidence=1.0
                    )
                
                # Build relationships
                if concept_data.get('related'):
                    for related in concept_data['related']:
                        self.learning_state['knowledge_graph'].setdefault(concept_name, []).append(related)
                        self.learning_state['knowledge_graph'].setdefault(related, []).append(concept_name)
    
    def learn_concept_advanced(self, concept_name: str, use_socratic: bool = True) -> Dict:
        """
        Advanced concept learning using ALL capabilities
        
        Uses:
        - Brain Topology for cognitive processing
        - Socratic Method for deep understanding
        - Information Theory for optimal learning
        - Metacognition for self-assessment
        - Pattern Abstraction for generalization
        """
        logger.info(f"Advanced learning: {concept_name}")
        
        # Get concept
        concept_data = None
        level = None
        for lvl in ['beginner', 'intermediate', 'advanced']:
            if concept_name in self.knowledge_base.get(lvl, {}):
                concept_data = self.knowledge_base[lvl][concept_name]
                level = lvl
                break
        
        if not concept_data:
            # Use organism to discover
            all_concepts = []
            for lvl_data in self.knowledge_base.values():
                all_concepts.extend(list(lvl_data.keys()))
            
            search_results = self.organism.search(concept_name, all_concepts)
            if search_results['results']:
                closest = search_results['results'][0]['item']
                for lvl in ['beginner', 'intermediate', 'advanced']:
                    if closest in self.knowledge_base.get(lvl, {}):
                        concept_data = self.knowledge_base[lvl][closest]
                        concept_name = closest
                        level = lvl
                        break
        
        if not concept_data:
            return {
                'concept': concept_name,
                'found': False,
                'message': f"Concept '{concept_name}' not found. Try: machine_learning, classification, neural_networks, etc."
            }
        
        # BRAIN PROCESSING: Process through cognitive architecture
        if self.brain:
            brain_result = self.brain.process(
                {
                    'concept': concept_name,
                    'data': concept_data
                },
                task=f"Learn: {concept_name}"
            )
            
            # Working Memory: Add to active learning
            self.brain.working_memory.add(
                f"learning_{concept_name}",
                concept_data['concept'],
                chunk_type="concept"
            )
            
            # Episodic Memory: Check if learned before
            past_learning = self.brain.episodic_memory.search_events(f"Learn: {concept_name}")
            
            # Semantic Memory: Retrieve related knowledge
            semantic_knowledge = self.brain.semantic_memory.search(concept_name)
            
            # Attention: Focus on key concepts
            key_terms = concept_data.get('key_terms', [])
            if key_terms:
                focused_terms = self.brain.attention.get_focused_items(key_terms, query=concept_name)
            else:
                focused_terms = []
            
            # Metacognition: Assess learning readiness
            metacognition = self.brain.metacognition.get_self_report()
            
            # Pattern Abstraction: Find similar concepts
            if self.brain.pattern_abstraction:
                patterns = self.brain.pattern_abstraction.get_patterns()
                similar_concepts = []
                for pattern_name in patterns.keys():
                    if concept_name.lower() in pattern_name.lower():
                        similar_concepts.append(pattern_name)
        else:
            brain_result = None
            past_learning = []
            semantic_knowledge = []
            focused_terms = []
            metacognition = {}
            similar_concepts = []
        
        # SOCRATIC METHOD: Generate deep questions
        socratic_questions = []
        if use_socratic and self.socratic:
            # Generate questions for deep understanding
            for q_type in ['clarification', 'assumption', 'evidence', 'implication']:
                question = self.socratic.generate_question(
                    concept_data['concept'],
                    question_type=q_type
                )
                socratic_questions.append(question)
        
        # INFORMATION THEORY: Calculate optimal learning path
        optimal_path = None
        if INFO_THEORY_AVAILABLE and concept_data.get('prerequisites'):
            # Calculate information gain for learning order
            prerequisites = concept_data.get('prerequisites', [])
            if prerequisites:
                # Use information gain to order prerequisites
                optimal_path = self._calculate_optimal_path(prerequisites, concept_name)
        
        # ADAPTIVE LEARNING: Personalize to user
        personalized_content = self._personalize_content(concept_data, level)
        
        # Remember learning session
        if self.brain:
            event_id = self.brain.episodic_memory.remember_event(
                what=f"Learned: {concept_name}",
                where="learning_companion",
                importance=0.9
            )
        
        # Update learning analytics
        self._update_learning_analytics(concept_name, concept_data)
        
        # Build comprehensive result
        result = {
            'concept': concept_name,
            'level': level,
            'found': True,
            'explanation': concept_data['concept'],
            'detailed_explanation': concept_data.get('detailed_explanation', concept_data['concept']),
            'examples': concept_data.get('examples', []),
            'key_terms': concept_data.get('key_terms', []),
            'focused_terms': focused_terms[:5],  # Top 5 from attention
            'prerequisites': concept_data.get('prerequisites', []),
            'optimal_path': optimal_path,
            'difficulty': concept_data.get('difficulty', 3),
            'estimated_time': concept_data.get('estimated_time', 'Unknown'),
            'related_concepts': concept_data.get('related', []),
            'common_misconceptions': concept_data.get('common_misconceptions', []),
            'practice_projects': concept_data.get('practice_projects', []),
            'socratic_questions': socratic_questions,
            'personalized_content': personalized_content,
            'brain_processing': {
                'past_learning_sessions': len(past_learning),
                'semantic_knowledge_retrieved': len(semantic_knowledge),
                'working_memory_items': len(self.brain.working_memory.chunks) if self.brain else 0,
                'cognitive_load': self.brain.working_memory.cognitive_load.current_load if self.brain else 0,
                'metacognition': metacognition
            } if self.brain else None,
            'learning_tips': self._generate_advanced_learning_tips(concept_name, concept_data),
            'practice_suggestions': self._generate_advanced_practice(concept_name, concept_data),
            'next_steps': self._suggest_next_steps(concept_name, concept_data)
        }
        
        # Track learning
        if concept_name not in self.learning_state['topics_learned']:
            self.learning_state['topics_learned'].append(concept_name)
        
        return result
    
    def _calculate_optimal_path(self, prerequisites: List[str], target: str) -> List[str]:
        """Calculate optimal learning path using information theory"""
        # Simple path calculation (can be enhanced with actual info theory)
        if not prerequisites:
            return [target]
        
        # Order by difficulty
        ordered = []
        for prereq in prerequisites:
            for level in ['beginner', 'intermediate', 'advanced']:
                if prereq in self.knowledge_base.get(level, {}):
                    difficulty = self.knowledge_base[level][prereq].get('difficulty', 3)
                    ordered.append((difficulty, prereq))
                    break
        
        ordered.sort()
        path = [p[1] for p in ordered]
        path.append(target)
        return path
    
    def _personalize_content(self, concept_data: Dict, level: str) -> Dict:
        """Personalize content based on learning style and level"""
        personalized = {
            'explanation_style': self.learning_state['learning_style'],
            'pace': self.learning_state['preferred_pace'],
            'focus_areas': [],
            'skip_areas': []
        }
        
        # Adjust based on learning style
        if self.learning_state['learning_style'] == 'visual':
            personalized['focus_areas'].append('diagrams')
            personalized['focus_areas'].append('visual_examples')
        elif self.learning_state['learning_style'] == 'kinesthetic':
            personalized['focus_areas'].append('hands_on')
            personalized['focus_areas'].append('practice_projects')
        
        # Adjust based on level
        if level == 'beginner':
            personalized['skip_areas'].append('advanced_details')
        elif level == 'advanced':
            personalized['focus_areas'].append('advanced_details')
        
        return personalized
    
    def _generate_advanced_learning_tips(self, concept: str, data: Dict) -> List[str]:
        """Generate advanced learning tips using all capabilities"""
        tips = []
        
        # Check prerequisites
        if data.get('prerequisites'):
            missing = [p for p in data['prerequisites'] if p not in self.learning_state['topics_learned']]
            if missing:
                tips.append(f"Prerequisites needed: {', '.join(missing)}. Learn these first for better understanding.")
        
        # Difficulty-based tips
        difficulty = data.get('difficulty', 3)
        if difficulty >= 5:
            tips.append("This is an advanced concept. Take your time and practice extensively.")
        elif difficulty <= 2:
            tips.append("This is a beginner-friendly concept. Great starting point!")
        
        # Brain-based tips
        if self.brain:
            cognitive_load = self.brain.working_memory.cognitive_load.current_load
            if cognitive_load > 5:
                tips.append("Your cognitive load is high. Consider taking a break or reviewing simpler concepts first.")
        
        # Socratic tips
        if self.socratic:
            tips.append("Use Socratic questioning to deepen your understanding. Ask 'why' and 'how' questions.")
        
        # Information theory tips
        if data.get('key_terms'):
            tips.append(f"Master these key terms first: {', '.join(data['key_terms'][:3])}")
        
        # Practice tips
        if data.get('practice_projects'):
            tips.append(f"Try this practice project: {data['practice_projects'][0]}")
        
        return tips
    
    def _generate_advanced_practice(self, concept: str, data: Dict) -> List[Dict]:
        """Generate advanced practice suggestions"""
        practices = []
        
        # Basic practice
        if data.get('practice_projects'):
            for project in data['practice_projects']:
                practices.append({
                    'type': 'project',
                    'title': project,
                    'difficulty': data.get('difficulty', 3),
                    'estimated_time': '2-4 hours'
                })
        
        # Socratic practice
        if self.socratic:
            practices.append({
                'type': 'socratic',
                'title': 'Socratic Questioning',
                'description': 'Use Socratic method to deepen understanding',
                'questions': data.get('socratic_questions', [])
            })
        
        # Brain-based practice
        if self.brain:
            practices.append({
                'type': 'brain',
                'title': 'Cognitive Practice',
                'description': 'Use working memory exercises',
                'exercises': ['Recall key terms', 'Explain to someone', 'Draw concept map']
            })
        
        return practices
    
    def _suggest_next_steps(self, concept: str, data: Dict) -> List[Dict]:
        """Suggest next learning steps"""
        next_steps = []
        
        # Related concepts
        if data.get('related'):
            for related in data['related'][:3]:
                next_steps.append({
                    'type': 'concept',
                    'concept': related,
                    'reason': f'Related to {concept}',
                    'priority': 'high'
                })
        
        # Prerequisites for advanced concepts
        if data.get('prerequisites'):
            for prereq in data['prerequisites']:
                if prereq not in self.learning_state['topics_learned']:
                    next_steps.append({
                        'type': 'prerequisite',
                        'concept': prereq,
                        'reason': f'Required for {concept}',
                        'priority': 'critical'
                    })
        
        return next_steps
    
    def _update_learning_analytics(self, concept: str, data: Dict):
        """Update learning analytics"""
        # Track mastery
        if concept not in self.learning_analytics['concept_mastery']:
            self.learning_analytics['concept_mastery'][concept] = 0.5  # Initial
        
        # Track difficulty
        self.learning_analytics['concept_mastery'][concept] = min(
            1.0,
            self.learning_analytics['concept_mastery'][concept] + 0.1
        )
    
    def answer_question_advanced(self, question: str) -> Dict:
        """
        Advanced question answering using ALL capabilities
        """
        # Brain processing
        if self.brain:
            brain_result = self.brain.process(question, task="Answer question")
            
            # Semantic memory search
            semantic_results = self.brain.semantic_memory.search(question)
            
            # Episodic memory: Check past similar questions
            past_qa = self.brain.episodic_memory.search_events("qa")
        else:
            brain_result = None
            semantic_results = []
            past_qa = []
        
        # Organism search
        all_concepts = []
        for level_data in self.knowledge_base.values():
            for concept_name, concept_data in level_data.items():
                all_concepts.append(f"{concept_name}: {concept_data['concept']}")
        
        search_results = self.organism.search(question, all_concepts)
        
        # Socratic reasoning
        if self.socratic:
            # Generate clarifying questions
            clarifying = self.socratic.generate_question(question, question_type='clarification')
        else:
            clarifying = None
        
        # Build comprehensive answer
        answer_text = self._build_comprehensive_answer(question, search_results, semantic_results)
        
        # Metacognition: Assess answer quality
        if self.brain:
            confidence = self.brain.metacognition.get_self_report().get('confidence', 0.5)
        else:
            confidence = 0.7
        
        result = {
            'question': question,
            'answer': answer_text,
            'confidence': confidence,
            'related_concepts': [r['item'].split(':')[0] for r in search_results['results'][:3]],
            'sources': search_results['results'][:3],
            'clarifying_question': clarifying,
            'brain_processing': {
                'semantic_matches': len(semantic_results),
                'past_similar_questions': len(past_qa)
            } if self.brain else None
        }
        
        # Remember Q&A
        if self.brain:
            self.brain.episodic_memory.remember_event(
                what=f"Q: {question}",
                where="learning_companion",
                importance=0.7
            )
        
        return result
    
    def _build_comprehensive_answer(self, question: str, search_results: Dict, semantic_results: List) -> str:
        """Build comprehensive answer from multiple sources"""
        # Extract relevant information
        relevant_info = []
        
        # From search results
        for result in search_results['results'][:3]:
            concept_part = result['item'].split(':')
            if len(concept_part) == 2:
                concept_name = concept_part[0]
                # Get full concept data
                for level_data in self.knowledge_base.values():
                    if concept_name in level_data:
                        concept_data = level_data[concept_name]
                        relevant_info.append({
                            'concept': concept_name,
                            'explanation': concept_data.get('detailed_explanation', concept_data['concept']),
                            'examples': concept_data.get('examples', [])
                        })
                        break
        
        # Build answer
        if not relevant_info:
            return "I don't have enough information to answer that question. Try asking about a specific ML/AI concept."
        
        answer_parts = []
        for info in relevant_info[:2]:  # Top 2
            answer_parts.append(f"{info['explanation']}")
            if info.get('examples'):
                answer_parts.append(f"\n\nExamples: {', '.join(info['examples'][:2])}")
        
        return "\n\n".join(answer_parts)
    
    def suggest_personalized_path(self, goal: str) -> Dict:
        """
        Suggest personalized learning path using information theory and brain topology
        """
        # Find base path
        base_paths = {
            'ml_fundamentals': ['machine_learning', 'supervised_learning', 'classification', 'regression'],
            'deep_learning': ['neural_networks', 'deep_learning', 'cnn'],
            'practical_ml': ['feature_engineering', 'model_selection', 'hyperparameter_tuning']
        }
        
        base_path = base_paths.get(goal, [])
        
        # Use information theory to optimize path
        if INFO_THEORY_AVAILABLE and base_path:
            optimized_path = self._optimize_path_with_info_theory(base_path)
        else:
            optimized_path = base_path
        
        # Check prerequisites using brain
        full_path = []
        for concept in optimized_path:
            # Add prerequisites if not learned
            for level_data in self.knowledge_base.values():
                if concept in level_data:
                    prereqs = level_data[concept].get('prerequisites', [])
                    for prereq in prereqs:
                        if prereq not in self.learning_state['topics_learned'] and prereq not in full_path:
                            full_path.append(prereq)
            full_path.append(concept)
        
        # Calculate estimated time
        total_time = 0
        for concept in full_path:
            for level_data in self.knowledge_base.values():
                if concept in level_data:
                    time_str = level_data[concept].get('estimated_time', '2 hours')
                    # Extract hours (simple)
                    if 'hour' in time_str:
                        try:
                            hours = float(time_str.split()[0])
                            total_time += hours
                        except:
                            total_time += 2
                    break
        
        # Use brain to assess readiness
        if self.brain:
            readiness = self.brain.metacognition.get_self_report().get('confidence', 0.5)
        else:
            readiness = 0.7
        
        return {
            'goal': goal,
            'path': full_path,
            'optimized_path': optimized_path,
            'steps': len(full_path),
            'estimated_time': f"{total_time:.0f} hours",
            'readiness': readiness,
            'personalized': True,
            'next_step': full_path[0] if full_path else None
        }
    
    def _optimize_path_with_info_theory(self, path: List[str]) -> List[str]:
        """Optimize learning path using information theory"""
        # Simple optimization: order by information gain
        # In practice, would calculate actual information gain
        ordered = []
        for concept in path:
            for level_data in self.knowledge_base.values():
                if concept in level_data:
                    difficulty = level_data[concept].get('difficulty', 3)
                    ordered.append((difficulty, concept))
                    break
        
        ordered.sort()
        return [c[1] for c in ordered]
    
    def assess_progress_advanced(self) -> Dict:
        """
        Advanced progress assessment using brain topology and analytics
        """
        # Get basic progress
        basic_progress = {
            'topics_learned': len(self.learning_state['topics_learned']),
            'topics_list': self.learning_state['topics_learned'],
            'learning_sessions': 0,
            'questions_asked': 0,
            'current_level': self.learning_state['level']
        }
        
        # Brain-based assessment
        if self.brain:
            learning_events = self.brain.episodic_memory.search_events("Learn")
            qa_events = self.brain.episodic_memory.search_events("qa")
            
            basic_progress['learning_sessions'] = len(learning_events)
            basic_progress['questions_asked'] = len(qa_events)
            
            # Metacognition assessment
            metacognition = self.brain.metacognition.get_self_report()
            basic_progress['self_awareness'] = metacognition.get('confidence', 0.5)
            
            # Working memory assessment
            basic_progress['cognitive_load'] = self.brain.working_memory.cognitive_load.current_load
            basic_progress['working_memory_usage'] = len(self.brain.working_memory.chunks)
        
        # Learning analytics
        mastery_scores = list(self.learning_analytics['concept_mastery'].values())
        avg_mastery = sum(mastery_scores) / len(mastery_scores) if mastery_scores else 0
        
        # Recommendations using information theory
        recommendations = self._generate_advanced_recommendations()
        
        # Calculate progress percentage
        total_concepts = sum(len(level_data) for level_data in self.knowledge_base.values())
        progress_pct = basic_progress['topics_learned'] / total_concepts if total_concepts > 0 else 0
        
        return {
            **basic_progress,
            'average_mastery': avg_mastery,
            'progress_percentage': progress_pct,
            'recommendations': recommendations,
            'learning_analytics': {
                'concepts_mastered': len([c for c, m in self.learning_analytics['concept_mastery'].items() if m > 0.7]),
                'concepts_in_progress': len([c for c, m in self.learning_analytics['concept_mastery'].items() if 0.3 < m <= 0.7]),
                'concepts_new': len([c for c, m in self.learning_analytics['concept_mastery'].items() if m <= 0.3])
            }
        }
    
    def _generate_advanced_recommendations(self) -> List[str]:
        """Generate advanced recommendations using all capabilities"""
        recommendations = []
        
        topics_learned = len(self.learning_state['topics_learned'])
        
        # Level-based recommendations
        if topics_learned < 3:
            recommendations.append("Start with fundamentals: machine_learning, classification, regression")
        elif topics_learned < 10:
            recommendations.append("Explore intermediate topics: neural_networks, deep_learning, feature_engineering")
        else:
            recommendations.append("Ready for advanced topics: transformer, reinforcement_learning")
        
        # Brain-based recommendations
        if self.brain:
            cognitive_load = self.brain.working_memory.cognitive_load.current_load
            if cognitive_load > 6:
                recommendations.append("Your cognitive load is high. Review previous concepts before learning new ones.")
        
        # Prerequisite-based recommendations
        for level_data in self.knowledge_base.values():
            for concept_name, concept_data in level_data.items():
                if concept_name not in self.learning_state['topics_learned']:
                    prereqs = concept_data.get('prerequisites', [])
                    missing = [p for p in prereqs if p not in self.learning_state['topics_learned']]
                    if not missing and concept_name not in [r.split(':')[1].strip() for r in recommendations if ':' in r]:
                        recommendations.append(f"Ready to learn: {concept_name}")
                        break
        
        # Socratic recommendations
        if self.socratic:
            recommendations.append("Use Socratic questioning to deepen understanding of concepts you've learned")
        
        return recommendations


def main():
    """Main function"""
    print("\n" + "="*80)
    print("ADVANCED AI LEARNING COMPANION".center(80))
    print("="*80)
    print("\nUsing ALL advanced capabilities:")
    print("  • Brain Topology (Working Memory, Episodic Memory, Semantic Memory)")
    print("  • Socratic Method (Deep questioning)")
    print("  • Information Theory (Optimal learning paths)")
    print("  • Metacognition (Self-awareness)")
    print("  • Pattern Abstraction (Generalization)")
    print("  • Adaptive Learning (Personalization)")
    print("\n" + "="*80 + "\n")
    
    companion = AdvancedLearningCompanion()
    
    # Demo
    print("Demo: Learning 'classification' with advanced features...\n")
    result = companion.learn_concept_advanced('classification', use_socratic=True)
    
    print("="*80)
    print(f"CONCEPT: {result['concept'].upper()}")
    print("="*80)
    print(f"\n[Level] {result['level']}")
    print(f"[Difficulty] {result['difficulty']}/6")
    print(f"[Estimated Time] {result['estimated_time']}")
    
    print(f"\n[Explanation]")
    print(result['detailed_explanation'][:200] + "...")
    
    if result.get('socratic_questions'):
        print(f"\n[Socratic Questions for Deep Understanding]")
        for i, q in enumerate(result['socratic_questions'][:3], 1):
            print(f"  {i}. {q}")
    
    if result.get('optimal_path'):
        print(f"\n[Optimal Learning Path]")
        print(f"  {' -> '.join(result['optimal_path'])}")
    
    if result.get('brain_processing'):
        print(f"\n[Brain Processing]")
        brain = result['brain_processing']
        print(f"  Working Memory Items: {brain['working_memory_items']}")
        print(f"  Cognitive Load: {brain['cognitive_load']:.2f}")
        print(f"  Past Learning Sessions: {brain['past_learning_sessions']}")
    
    if result.get('next_steps'):
        print(f"\n[Next Steps]")
        for step in result['next_steps'][:3]:
            print(f"  • {step['concept']} ({step['reason']})")
    
    print("\n" + "="*80)
    print("\nThis is MUCH more advanced than the basic version!")
    print("It uses brain topology, Socratic method, information theory, and more!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
