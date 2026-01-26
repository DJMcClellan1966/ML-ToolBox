"""
AI Learning Companion - Built on ML Organism

A personal AI learning companion that helps you learn ML and AI.
Uses the organic ML organism to provide personalized learning experiences.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
import logging
from collections import defaultdict
import time
import json

logger = logging.getLogger(__name__)

# Import the organism
from organic_ml_organism import MLOrganism, UnifiedMemory, UnifiedLearning, UnifiedDiscovery, UnifiedReasoning


class LearningCompanion:
    """
    AI Learning Companion
    
    Uses the ML Organism to help you learn ML and AI concepts.
    Adapts to your learning style and level.
    """
    
    def __init__(self):
        """Initialize the learning companion"""
        logger.info("Initializing AI Learning Companion...")
        
        # The organism (your learning partner)
        self.organism = MLOrganism()
        
        # Learning state
        self.learning_state = {
            'level': 'beginner',  # beginner, intermediate, advanced
            'topics_learned': [],
            'topics_in_progress': [],
            'weak_areas': [],
            'strong_areas': [],
            'learning_style': 'visual',  # visual, auditory, kinesthetic, reading
            'preferred_pace': 'moderate'  # slow, moderate, fast
        }
        
        # Knowledge base (ML/AI concepts)
        self.knowledge_base = self._build_knowledge_base()
        
        # Learning paths
        self.learning_paths = self._build_learning_paths()
        
        logger.info("AI Learning Companion ready!")
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base of ML/AI concepts"""
        return {
            'beginner': {
                'machine_learning': {
                    'concept': 'Machine Learning is about teaching computers to learn from data without explicit programming.',
                    'examples': [
                        'Email spam detection',
                        'Image recognition',
                        'Recommendation systems'
                    ],
                    'key_terms': ['data', 'training', 'prediction', 'model'],
                    'related': ['supervised_learning', 'unsupervised_learning']
                },
                'supervised_learning': {
                    'concept': 'Supervised learning uses labeled data to train models that can make predictions.',
                    'examples': [
                        'Classifying emails as spam/not spam',
                        'Predicting house prices',
                        'Recognizing handwritten digits'
                    ],
                    'key_terms': ['labeled data', 'training', 'labels', 'prediction'],
                    'related': ['classification', 'regression']
                },
                'classification': {
                    'concept': 'Classification predicts discrete categories (like spam/not spam).',
                    'examples': [
                        'Email spam detection',
                        'Image classification',
                        'Sentiment analysis'
                    ],
                    'key_terms': ['categories', 'classes', 'labels', 'accuracy'],
                    'related': ['regression', 'neural_networks']
                },
                'regression': {
                    'concept': 'Regression predicts continuous values (like prices or temperatures).',
                    'examples': [
                        'House price prediction',
                        'Stock price forecasting',
                        'Temperature prediction'
                    ],
                    'key_terms': ['continuous', 'values', 'prediction', 'error'],
                    'related': ['classification', 'linear_regression']
                }
            },
            'intermediate': {
                'neural_networks': {
                    'concept': 'Neural networks are inspired by the brain, with layers of interconnected neurons.',
                    'examples': [
                        'Image recognition',
                        'Natural language processing',
                        'Speech recognition'
                    ],
                    'key_terms': ['neurons', 'layers', 'weights', 'activation'],
                    'related': ['deep_learning', 'backpropagation']
                },
                'deep_learning': {
                    'concept': 'Deep learning uses neural networks with many layers to learn complex patterns.',
                    'examples': [
                        'Image classification',
                        'Language translation',
                        'Autonomous vehicles'
                    ],
                    'key_terms': ['deep', 'layers', 'complex patterns', 'representation'],
                    'related': ['neural_networks', 'cnn', 'rnn']
                },
                'feature_engineering': {
                    'concept': 'Feature engineering creates meaningful inputs for ML models from raw data.',
                    'examples': [
                        'Extracting text features',
                        'Creating time-based features',
                        'Combining features'
                    ],
                    'key_terms': ['features', 'transformation', 'selection', 'extraction'],
                    'related': ['preprocessing', 'dimensionality_reduction']
                }
            },
            'advanced': {
                'transformer': {
                    'concept': 'Transformers use attention mechanisms to process sequences, revolutionizing NLP.',
                    'examples': [
                        'GPT models',
                        'BERT',
                        'Language translation'
                    ],
                    'key_terms': ['attention', 'self-attention', 'encoder', 'decoder'],
                    'related': ['bert', 'gpt', 'attention_mechanism']
                },
                'reinforcement_learning': {
                    'concept': 'Reinforcement learning learns through trial and error, receiving rewards for good actions.',
                    'examples': [
                        'Game playing (AlphaGo)',
                        'Robotics',
                        'Autonomous systems'
                    ],
                    'key_terms': ['agent', 'environment', 'reward', 'policy'],
                    'related': ['q_learning', 'policy_gradient']
                }
            }
        }
    
    def _build_learning_paths(self) -> Dict:
        """Build learning paths for different goals"""
        return {
            'ml_fundamentals': [
                'machine_learning',
                'supervised_learning',
                'classification',
                'regression',
                'evaluation_metrics'
            ],
            'deep_learning': [
                'neural_networks',
                'deep_learning',
                'cnn',
                'rnn',
                'transformer'
            ],
            'practical_ml': [
                'data_preprocessing',
                'feature_engineering',
                'model_selection',
                'hyperparameter_tuning',
                'deployment'
            ]
        }
    
    def learn_concept(self, concept_name: str, level: Optional[str] = None) -> Dict:
        """
        Learn a concept - uses organism to help you learn
        
        Args:
            concept_name: Name of concept to learn
            level: Learning level (beginner/intermediate/advanced)
            
        Returns:
            Learning experience with explanations, examples, and practice
        """
        level = level or self.learning_state['level']
        
        # Get concept from knowledge base
        concept_data = None
        if level in self.knowledge_base:
            if concept_name in self.knowledge_base[level]:
                concept_data = self.knowledge_base[level][concept_name]
        
        if not concept_data:
            # Search for concept using organism
            all_concepts = []
            for lvl in self.knowledge_base.values():
                all_concepts.extend(list(lvl.keys()))
            
            # Use organism's discovery to find related concepts
            search_results = self.organism.search(concept_name, all_concepts)
            if search_results['results']:
                closest = search_results['results'][0]['item']
                # Try to find in any level
                for lvl in ['beginner', 'intermediate', 'advanced']:
                    if closest in self.knowledge_base.get(lvl, {}):
                        concept_data = self.knowledge_base[lvl][closest]
                        concept_name = closest
                        break
        
        if not concept_data:
            return {
                'concept': concept_name,
                'found': False,
                'message': f"Concept '{concept_name}' not found. Try asking about: machine_learning, classification, regression, neural_networks, etc."
            }
        
        # Use organism to process learning
        learning_result = self.organism.process(
            {
                'concept': concept_name,
                'data': concept_data
            },
            task=f"Learn: {concept_name}"
        )
        
        # Remember this learning session
        self.organism.memory.remember(
            {
                'concept': concept_name,
                'level': level,
                'timestamp': time.time()
            },
            'event',
            type='learning'
        )
        
        # Track learning progress
        if concept_name not in self.learning_state['topics_learned']:
            self.learning_state['topics_learned'].append(concept_name)
        
        # Build learning experience
        experience = {
            'concept': concept_name,
            'level': level,
            'explanation': concept_data['concept'],
            'examples': concept_data.get('examples', []),
            'key_terms': concept_data.get('key_terms', []),
            'related_concepts': concept_data.get('related', []),
            'learning_tips': self._generate_learning_tips(concept_name, concept_data),
            'practice_suggestions': self._generate_practice_suggestions(concept_name),
            'organism_insights': learning_result
        }
        
        return experience
    
    def _generate_learning_tips(self, concept: str, data: Dict) -> List[str]:
        """Generate personalized learning tips"""
        tips = []
        
        # General tips
        tips.append(f"Focus on understanding '{data['concept'][:50]}...'")
        
        # Key terms tip
        if data.get('key_terms'):
            tips.append(f"Master these key terms: {', '.join(data['key_terms'][:3])}")
        
        # Examples tip
        if data.get('examples'):
            tips.append(f"Think about real-world examples: {data['examples'][0]}")
        
        # Related concepts tip
        if data.get('related'):
            tips.append(f"Next, explore: {data['related'][0]}")
        
        return tips
    
    def _generate_practice_suggestions(self, concept: str) -> List[str]:
        """Generate practice suggestions"""
        suggestions = []
        
        if 'classification' in concept:
            suggestions.append("Try building a spam email classifier")
            suggestions.append("Practice with the Iris dataset")
        elif 'regression' in concept:
            suggestions.append("Try predicting house prices")
            suggestions.append("Practice with linear regression")
        elif 'neural' in concept:
            suggestions.append("Build a simple neural network")
            suggestions.append("Try the MNIST digit recognition")
        elif 'feature' in concept:
            suggestions.append("Practice feature extraction from text")
            suggestions.append("Try feature selection techniques")
        
        return suggestions
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer your question about ML/AI
        
        Uses organism's reasoning and discovery to answer
        """
        # Use organism to search knowledge
        all_concepts = []
        for level_data in self.knowledge_base.values():
            for concept_name, concept_data in level_data.items():
                all_concepts.append(f"{concept_name}: {concept_data['concept']}")
        
        # Search for relevant information
        search_results = self.organism.search(question, all_concepts)
        
        # Use organism's reasoning to formulate answer
        premises = [r['item'] for r in search_results['results'][:3]]
        reasoning = self.organism.reasoning.reason(premises, question)
        
        # Build answer
        answer = {
            'question': question,
            'answer': self._formulate_answer(question, search_results, reasoning),
            'related_concepts': [r['item'].split(':')[0] for r in search_results['results'][:3]],
            'confidence': reasoning.get('confidence', 0.5),
            'sources': search_results['results'][:3]
        }
        
        # Remember this Q&A
        self.organism.memory.remember(
            {
                'question': question,
                'answer': answer['answer'],
                'timestamp': time.time()
            },
            'event',
            type='qa'
        )
        
        return answer
    
    def _formulate_answer(self, question: str, search_results: Dict, reasoning: Dict) -> str:
        """Formulate a natural language answer"""
        if not search_results['results']:
            return "I don't have enough information to answer that question. Try asking about a specific ML/AI concept."
        
        # Extract relevant information
        relevant = search_results['results'][0]['item']
        concept_name = relevant.split(':')[0] if ':' in relevant else relevant
        
        # Get concept details
        concept_data = None
        for level_data in self.knowledge_base.values():
            if concept_name in level_data:
                concept_data = level_data[concept_name]
                break
        
        if concept_data:
            answer = f"Based on {concept_name}, {concept_data['concept']}"
            if concept_data.get('examples'):
                answer += f"\n\nExamples include: {', '.join(concept_data['examples'][:2])}"
        else:
            answer = relevant
        
        return answer
    
    def suggest_learning_path(self, goal: str) -> Dict:
        """
        Suggest a learning path based on your goal
        
        Uses organism to discover the best path
        """
        # Find matching learning path
        path = None
        if goal in self.learning_paths:
            path = self.learning_paths[goal]
        else:
            # Use organism to discover path
            all_paths = list(self.learning_paths.keys())
            search_results = self.organism.search(goal, all_paths)
            if search_results['results']:
                closest_path = search_results['results'][0]['item']
                if closest_path in self.learning_paths:
                    path = self.learning_paths[closest_path]
        
        if not path:
            # Create custom path
            path = self._create_custom_path(goal)
        
        # Use organism to reason about path
        reasoning = self.organism.reasoning.reason(
            [f"Goal: {goal}", f"Path: {path}"],
            "Is this a good learning path?"
        )
        
        return {
            'goal': goal,
            'path': path,
            'steps': len(path),
            'reasoning': reasoning,
            'estimated_time': f"{len(path) * 2} hours",
            'next_step': path[0] if path else None
        }
    
    def _create_custom_path(self, goal: str) -> List[str]:
        """Create a custom learning path"""
        # Simple path creation
        if 'beginner' in goal.lower() or 'start' in goal.lower():
            return ['machine_learning', 'supervised_learning', 'classification']
        elif 'deep' in goal.lower() or 'neural' in goal.lower():
            return ['neural_networks', 'deep_learning', 'cnn']
        else:
            return ['machine_learning', 'classification', 'regression']
    
    def assess_progress(self) -> Dict:
        """
        Assess your learning progress
        
        Uses organism's metacognition to assess
        """
        # Get learning history from organism
        learning_events = self.organism.memory.recall('learning', 'events')
        qa_events = self.organism.memory.recall('qa', 'events')
        
        # Assess progress
        progress = {
            'topics_learned': len(self.learning_state['topics_learned']),
            'topics_list': self.learning_state['topics_learned'],
            'learning_sessions': len(learning_events),
            'questions_asked': len(qa_events),
            'current_level': self.learning_state['level'],
            'recommendations': []
        }
        
        # Generate recommendations
        if progress['topics_learned'] < 3:
            progress['recommendations'].append("Start with fundamentals: machine_learning, classification, regression")
        elif progress['topics_learned'] < 10:
            progress['recommendations'].append("Explore intermediate topics: neural_networks, deep_learning")
        else:
            progress['recommendations'].append("Ready for advanced topics: transformers, reinforcement_learning")
        
        return progress
    
    def interactive_learn(self):
        """Interactive learning session"""
        print("\n" + "="*80)
        print("AI LEARNING COMPANION - Interactive Learning".center(80))
        print("="*80)
        print("\nI'm your AI learning companion! I'll help you learn ML and AI.")
        print("Commands:")
        print("  learn <concept>  - Learn a concept (e.g., 'learn classification')")
        print("  ask <question>   - Ask a question (e.g., 'ask what is machine learning')")
        print("  path <goal>      - Get a learning path (e.g., 'path ml_fundamentals')")
        print("  progress         - See your learning progress")
        print("  quit             - Exit")
        print("\n" + "="*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input or user_input.lower() == 'quit':
                    break
                
                # Parse command
                if user_input.startswith('learn '):
                    concept = user_input[6:].strip()
                    result = self.learn_concept(concept)
                    self._print_learning_result(result)
                
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    result = self.answer_question(question)
                    self._print_answer(result)
                
                elif user_input.startswith('path '):
                    goal = user_input[5:].strip()
                    result = self.suggest_learning_path(goal)
                    self._print_path(result)
                
                elif user_input.lower() == 'progress':
                    result = self.assess_progress()
                    self._print_progress(result)
                
                else:
                    # Try as a question
                    result = self.answer_question(user_input)
                    self._print_answer(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Keep learning!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.exception("Error in interactive learning")
    
    def _print_learning_result(self, result: Dict):
        """Print learning result"""
        if not result.get('found', True):
            print(f"\n{result['message']}\n")
            return
        
        print("\n" + "="*80)
        print(f"LEARNING: {result['concept'].upper()}".center(80))
        print("="*80)
        print(f"\n[Explanation]")
        print(f"   {result['explanation']}")
        
        if result.get('examples'):
            print(f"\n[Examples]")
            for i, ex in enumerate(result['examples'], 1):
                print(f"   {i}. {ex}")
        
        if result.get('key_terms'):
            print(f"\n[Key Terms]")
            print(f"   {', '.join(result['key_terms'])}")
        
        if result.get('learning_tips'):
            print(f"\n[Learning Tips]")
            for tip in result['learning_tips']:
                print(f"   • {tip}")
        
        if result.get('practice_suggestions'):
            print(f"\n[Practice Suggestions]")
            for suggestion in result['practice_suggestions']:
                print(f"   • {suggestion}")
        
        if result.get('related_concepts'):
            print(f"\n[Related Concepts]")
            print(f"   {', '.join(result['related_concepts'])}")
        
        print("\n" + "="*80 + "\n")
    
    def _print_answer(self, result: Dict):
        """Print answer"""
        print("\n" + "="*80)
        print("ANSWER".center(80))
        print("="*80)
        print(f"\n[Question] {result['question']}")
        print(f"\n[Answer]")
        print(f"   {result['answer']}")
        
        if result.get('related_concepts'):
            print(f"\n[Related] {', '.join(result['related_concepts'])}")
        
        print("\n" + "="*80 + "\n")
    
    def _print_path(self, result: Dict):
        """Print learning path"""
        print("\n" + "="*80)
        print(f"LEARNING PATH: {result['goal'].upper()}".center(80))
        print("="*80)
        print(f"\n[Goal] {result['goal']}")
        print(f"[Steps] {result['steps']}")
        print(f"[Estimated Time] {result['estimated_time']}")
        print(f"\n[Path]")
        for i, step in enumerate(result['path'], 1):
            marker = "->" if i == 1 else "  "
            print(f"   {marker} {i}. {step}")
        
        if result.get('next_step'):
            print(f"\n[Next Step] Learn '{result['next_step']}'")
        
        print("\n" + "="*80 + "\n")
    
    def _print_progress(self, result: Dict):
        """Print progress"""
        print("\n" + "="*80)
        print("YOUR LEARNING PROGRESS".center(80))
        print("="*80)
        print(f"\n[Topics Learned] {result['topics_learned']}")
        print(f"[Learning Sessions] {result['learning_sessions']}")
        print(f"[Questions Asked] {result['questions_asked']}")
        print(f"[Current Level] {result['current_level']}")
        
        if result.get('recommendations'):
            print(f"\n[Recommendations]")
            for rec in result['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    companion = LearningCompanion()
    companion.interactive_learn()


if __name__ == "__main__":
    main()
