"""
ML Organism - Organic ML System

Created by removing all barriers and divisions, then reconstructing
based on natural fits and patterns.

This is what emerges when you take all the toolbox layers, mix them
together, and let natural patterns emerge.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


# ============================================================================
# CORE: Essential Functions (The Heart)
# ============================================================================

def flow_data(data: Any, transformations: List[Callable]) -> Any:
    """Natural data flow - no compartments, just flow"""
    result = data
    for transform in transformations:
        result = transform(result)
    return result


def transform(data: Any, operation: str, **kwargs) -> Any:
    """Natural transformation - no types, just operations"""
    if operation == "normalize":
        if isinstance(data, np.ndarray):
            return (data - data.mean()) / (data.std() + 1e-10)
    elif operation == "embed":
        # Simple embedding
        if isinstance(data, str):
            words = data.lower().split()
            return np.array([hash(w) % 1000 for w in words[:100]])
    elif operation == "similarity":
        if len(data) == 2:
            vec1, vec2 = data
            if isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
                # Ensure same dimension
                min_dim = min(len(vec1), len(vec2))
                if min_dim > 0:
                    vec1 = vec1[:min_dim]
                    vec2 = vec2[:min_dim]
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        return np.dot(vec1, vec2) / (norm1 * norm2)
                return 0.0
    return data


# ============================================================================
# MEMORY: Unified Memory System (The Brain)
# ============================================================================

class UnifiedMemory:
    """
    Unified Memory - All memory types work together naturally
    No separation between working, episodic, semantic - just memory
    """
    
    def __init__(self):
        # All memory unified
        self.memory = {
            'active': [],  # Working memory (active items)
            'events': [],  # Episodic memory (events)
            'facts': {},   # Semantic memory (facts)
            'cache': {}    # Cache memory (fast access)
        }
        self.relationships = defaultdict(set)  # Natural relationships
    
    def remember(self, item: Any, memory_type: str = 'active', **metadata):
        """Remember - no distinction, just remember"""
        if memory_type == 'active':
            self.memory['active'].append({
                'item': item,
                'timestamp': time.time(),
                **metadata
            })
            # Limit active memory (natural capacity)
            if len(self.memory['active']) > 7:
                # Move oldest to episodic
                oldest = self.memory['active'].pop(0)
                self.memory['events'].append(oldest)
        elif memory_type == 'event':
            self.memory['events'].append({
                'item': item,
                'timestamp': time.time(),
                **metadata
            })
        elif memory_type == 'fact':
            fact_id = f"fact_{len(self.memory['facts'])}"
            self.memory['facts'][fact_id] = {
                'fact': item,
                'timestamp': time.time(),
                **metadata
            }
        elif memory_type == 'cache':
            cache_key = metadata.get('key', str(item))
            self.memory['cache'][cache_key] = {
                'value': item,
                'timestamp': time.time()
            }
    
    def recall(self, query: Any, memory_type: Optional[str] = None) -> List[Any]:
        """Recall - search all memory naturally"""
        results = []
        
        if memory_type is None or memory_type == 'active':
            # Search active memory
            for item in self.memory['active']:
                if self._matches(query, item['item']):
                    results.append(item)
        
        if memory_type is None or memory_type == 'events':
            # Search events
            for event in self.memory['events']:
                if self._matches(query, event['item']):
                    results.append(event)
        
        if memory_type is None or memory_type == 'facts':
            # Search facts
            for fact in self.memory['facts'].values():
                if self._matches(query, fact['fact']):
                    results.append(fact)
        
        if memory_type == 'cache':
            # Check cache
            cache_key = str(query)
            if cache_key in self.memory['cache']:
                results.append(self.memory['cache'][cache_key])
        
        return results
    
    def _matches(self, query: Any, item: Any) -> bool:
        """Natural matching - no types, just similarity"""
        if isinstance(query, str) and isinstance(item, str):
            return query.lower() in item.lower()
        return False
    
    def connect(self, item1: Any, item2: Any, relationship: str = "related"):
        """Connect items - natural relationships"""
        key1 = str(item1)
        key2 = str(item2)
        self.relationships[key1].add((key2, relationship))
        self.relationships[key2].add((key1, relationship))


# ============================================================================
# LEARNING: Unified Learning System (Growth)
# ============================================================================

class UnifiedLearning:
    """
    Unified Learning - All learning works together
    Training, optimization, evolution, adaptation - all unified
    """
    
    def __init__(self):
        self.patterns = {}
        self.adaptations = {}
        self.performance_history = defaultdict(list)
    
    def learn_pattern(self, data: Any, pattern_name: str):
        """Learn pattern - natural pattern recognition"""
        if pattern_name not in self.patterns:
            self.patterns[pattern_name] = []
        
        # Extract pattern (simplified)
        if isinstance(data, list):
            pattern = {
                'length': len(data),
                'type': type(data[0]).__name__ if data else 'empty'
            }
        elif isinstance(data, dict):
            pattern = {
                'keys': list(data.keys()),
                'size': len(data)
            }
        else:
            pattern = {'value': str(data)}
        
        self.patterns[pattern_name].append(pattern)
    
    def adapt(self, situation: str, behavior: Callable, performance: float):
        """Adapt - natural adaptation"""
        if situation not in self.adaptations:
            self.adaptations[situation] = []
        
        self.adaptations[situation].append({
            'behavior': behavior,
            'performance': performance,
            'timestamp': time.time()
        })
        
        self.performance_history[situation].append(performance)
    
    def optimize(self, objective: Callable, constraints: List[Callable] = None):
        """Optimize - natural optimization"""
        # Simple optimization (can be enhanced)
        best_value = None
        best_params = None
        
        # Try different parameters
        for i in range(10):
            params = np.random.rand(5)  # Random params
            value = objective(params)
            
            # Check constraints
            if constraints:
                valid = all(c(params) for c in constraints)
                if not valid:
                    continue
            
            if best_value is None or value > best_value:
                best_value = value
                best_params = params
        
        return best_params, best_value


# ============================================================================
# DISCOVERY: Unified Discovery System (Exploration)
# ============================================================================

class UnifiedDiscovery:
    """
    Unified Discovery - Search, graphs, patterns all unified
    No separation - just discovery
    """
    
    def __init__(self, memory: UnifiedMemory):
        self.memory = memory
        self.knowledge_network = defaultdict(set)
    
    def search(self, query: Any, corpus: List[Any]) -> List[Dict]:
        """Search - natural search"""
        results = []
        
        # Embed query
        query_embed = transform(query, "embed")
        
        for item in corpus:
            # Embed item
            item_embed = transform(item, "embed")
            
            # Similarity
            similarity = transform((query_embed, item_embed), "similarity")
            
            if similarity > 0.1:  # Threshold
                results.append({
                    'item': item,
                    'similarity': similarity
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def discover_relationships(self, items: List[Any]) -> Dict:
        """Discover relationships - natural relationship discovery"""
        relationships = {}
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                # Check if related
                embed1 = transform(item1, "embed")
                embed2 = transform(item2, "embed")
                similarity = transform((embed1, embed2), "similarity")
                
                # Ensure similarity is a number
                if isinstance(similarity, (int, float)) and similarity > 0.3:
                    key1 = str(item1)
                    key2 = str(item2)
                    if key1 not in relationships:
                        relationships[key1] = []
                    relationships[key1].append({
                        'item': item2,
                        'similarity': similarity
                    })
        
        return relationships
    
    def build_network(self, items: List[Any]):
        """Build knowledge network - natural network building"""
        for item in items:
            key = str(item)
            self.knowledge_network[key] = set()
        
        # Find relationships
        relationships = self.discover_relationships(items)
        for key, related in relationships.items():
            for rel in related:
                self.knowledge_network[key].add(str(rel['item']))


# ============================================================================
# REASONING: Unified Reasoning System (Thinking)
# ============================================================================

class UnifiedReasoning:
    """
    Unified Reasoning - All reasoning unified
    Game theory, ethics, logic, multi-objective - all together
    """
    
    def __init__(self):
        self.reasoning_history = []
    
    def reason(self, premises: List[str], question: str) -> Dict:
        """Reason - natural reasoning"""
        # Simple reasoning (can be enhanced)
        reasoning = {
            'premises': premises,
            'question': question,
            'conclusion': None,
            'confidence': 0.5
        }
        
        # Check if premises support question
        if premises:
            # Simple pattern matching
            if any(q.lower() in p.lower() for p in premises for q in question.split()):
                reasoning['conclusion'] = "Premises support the question"
                reasoning['confidence'] = 0.7
        
        self.reasoning_history.append(reasoning)
        return reasoning
    
    def decide(self, options: List[Dict], objectives: List[Callable]) -> Dict:
        """Decide - natural decision making"""
        best_option = None
        best_score = None
        
        for option in options:
            # Score option on all objectives
            scores = [obj(option) for obj in objectives]
            total_score = sum(scores) / len(scores)  # Average
            
            if best_score is None or total_score > best_score:
                best_score = total_score
                best_option = option
        
        return {
            'decision': best_option,
            'score': best_score,
            'all_scores': [obj(option) for option in options for obj in objectives]
        }
    
    def ethical_check(self, action: str, context: Dict) -> Dict:
        """Ethical check - natural ethical reasoning"""
        concerns = []
        
        # Simple ethical checks
        if 'harm' in action.lower() or 'violence' in action.lower():
            concerns.append("Potential harm")
        
        if 'privacy' in context:
            concerns.append("Privacy consideration")
        
        return {
            'action': action,
            'concerns': concerns,
            'approved': len(concerns) == 0
        }


# ============================================================================
# THE ORGANISM: All Systems Unified
# ============================================================================

class MLOrganism:
    """
    ML Organism - Living ML System
    
    Created by removing all barriers and reconstructing based on
    natural fits and patterns. No compartments, just natural systems.
    """
    
    def __init__(self):
        """Initialize the organism"""
        logger.info("Growing ML Organism...")
        
        # Core systems (natural fits)
        self.memory = UnifiedMemory()
        self.learning = UnifiedLearning()
        self.discovery = UnifiedDiscovery(self.memory)
        self.reasoning = UnifiedReasoning()
        
        # Natural flows
        self.flows = {
            'data': [],
            'learning': [],
            'discovery': [],
            'reasoning': []
        }
        
        logger.info("ML Organism grown!")
    
    def process(self, input_data: Any, task: str = "") -> Dict:
        """
        Process - natural processing flow
        No compartments, just natural flow
        """
        result = {
            'input': input_data,
            'task': task,
            'output': None,
            'memory': {},
            'learning': {},
            'discovery': {},
            'reasoning': {}
        }
        
        # 1. Remember input
        self.memory.remember(input_data, 'active', task=task)
        
        # 2. Discover patterns
        if isinstance(input_data, list):
            patterns = self.discovery.discover_relationships(input_data)
            result['discovery']['patterns'] = patterns
        
        # 3. Learn
        self.learning.learn_pattern(input_data, task)
        result['learning']['pattern_learned'] = task
        
        # 4. Reason
        if task:
            reasoning = self.reasoning.reason([str(input_data)], task)
            result['reasoning'] = reasoning
        
        # 5. Transform (natural flow)
        output = flow_data(input_data, [
            lambda x: transform(x, "normalize") if isinstance(x, np.ndarray) else x
        ])
        result['output'] = output
        
        # 6. Remember output
        self.memory.remember(output, 'event', task=task)
        
        # 7. Connect input and output
        self.memory.connect(input_data, output, "processed")
        
        return result
    
    def search(self, query: Any, corpus: List[Any]) -> Dict:
        """Search - natural search flow"""
        # Remember query
        self.memory.remember(query, 'active', type='query')
        
        # Search
        results = self.discovery.search(query, corpus)
        
        # Remember results
        for result in results[:3]:  # Top 3
            self.memory.remember(result['item'], 'fact', similarity=result['similarity'])
            self.memory.connect(query, result['item'], "found")
        
        return {
            'query': query,
            'results': results,
            'count': len(results)
        }
    
    def learn_and_adapt(self, experience: Dict, performance: float):
        """Learn and adapt - natural learning flow"""
        situation = experience.get('situation', 'default')
        
        # Adapt
        self.learning.adapt(situation, lambda x: x, performance)
        
        # Remember experience
        self.memory.remember(experience, 'event', performance=performance)
        
        # Learn pattern
        self.learning.learn_pattern(experience, situation)
    
    def reason_and_decide(self, situation: Dict, options: List[Dict]) -> Dict:
        """Reason and decide - natural decision flow"""
        # Reason about situation
        premises = [str(v) for v in situation.values()]
        reasoning = self.reasoning.reason(premises, "What should I do?")
        
        # Ethical check
        ethical_checks = []
        for option in options:
            check = self.reasoning.ethical_check(str(option), situation)
            ethical_checks.append(check)
        
        # Decide
        objectives = [
            lambda opt: 1.0 if self.reasoning.ethical_check(str(opt), situation)['approved'] else 0.0
        ]
        decision = self.reasoning.decide(options, objectives)
        
        # Remember decision
        self.memory.remember(decision, 'event', situation=situation)
        
        return {
            'reasoning': reasoning,
            'ethical_checks': ethical_checks,
            'decision': decision
        }


# ============================================================================
# DEMO
# ============================================================================

def main():
    """Demo the ML Organism"""
    print("\n" + "="*80)
    print("ML ORGANISM - Living ML System".center(80))
    print("="*80)
    print("\nCreated by removing all barriers and reconstructing based on")
    print("natural fits and patterns. No compartments, just natural systems.")
    print("\n" + "="*80 + "\n")
    
    # Grow the organism
    organism = MLOrganism()
    
    # Demo 1: Natural Processing
    print("Demo 1: Natural Processing Flow")
    print("-" * 80)
    data = np.array([1, 2, 3, 4, 5])
    result = organism.process(data, task="normalize")
    print(f"Input: {data}")
    print(f"Output: {result['output']}")
    print(f"Memory: {len(organism.memory.memory['active'])} active items")
    print(f"Learning: {len(organism.learning.patterns)} patterns learned")
    
    # Demo 2: Natural Search
    print("\nDemo 2: Natural Search Flow")
    print("-" * 80)
    corpus = [
        "Machine learning is about patterns",
        "Deep learning uses neural networks",
        "Natural language processing understands text"
    ]
    results = organism.search("machine learning", corpus)
    print(f"Query: machine learning")
    print(f"Found: {results['count']} results")
    for i, r in enumerate(results['results'][:2], 1):
        print(f"  {i}. {r['item'][:50]}... (similarity: {r['similarity']:.3f})")
    
    # Demo 3: Natural Learning
    print("\nDemo 3: Natural Learning Flow")
    print("-" * 80)
    organism.learn_and_adapt(
        {'situation': 'classification', 'accuracy': 0.95},
        performance=0.95
    )
    print(f"Adaptations: {len(organism.learning.adaptations)}")
    print(f"Performance history: {len(organism.learning.performance_history)} situations")
    
    # Demo 4: Natural Reasoning
    print("\nDemo 4: Natural Reasoning Flow")
    print("-" * 80)
    decision = organism.reason_and_decide(
        {'context': 'research', 'privacy': True},
        [
            {'action': 'analyze data', 'method': 'statistical'},
            {'action': 'share results', 'method': 'public'}
        ]
    )
    print(f"Reasoning: {decision['reasoning']['conclusion']}")
    print(f"Decision: {decision['decision']['decision']}")
    print(f"Ethical checks: {len(decision['ethical_checks'])}")
    
    # Show organism state
    print("\n" + "="*80)
    print("ORGANISM STATE".center(80))
    print("="*80)
    print(f"Active Memory: {len(organism.memory.memory['active'])} items")
    print(f"Events: {len(organism.memory.memory['events'])} events")
    print(f"Facts: {len(organism.memory.memory['facts'])} facts")
    print(f"Patterns Learned: {len(organism.learning.patterns)}")
    print(f"Knowledge Network: {len(organism.discovery.knowledge_network)} nodes")
    print(f"Reasoning History: {len(organism.reasoning.reasoning_history)}")
    print("\n" + "="*80)
    print("\nThe organism is alive and growing!")
    print("No compartments, no boundaries - just natural systems working together.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
