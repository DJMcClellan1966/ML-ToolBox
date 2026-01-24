"""
Cognitive Architecture - Complete Brain System

Integrates all brain-like features into unified cognitive architecture
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .attention_mechanism import AttentionMechanism
from .metacognition import Metacognition
from .pattern_abstraction import PatternAbstraction


class CognitiveArchitecture:
    """
    Cognitive Architecture
    
    Complete brain-like cognitive system
    """
    
    def __init__(self, working_memory_capacity: int = 7):
        """
        Initialize cognitive architecture
        
        Parameters
        ----------
        working_memory_capacity : int
            Working memory capacity
        """
        # Memory systems
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Cognitive processes
        self.attention = AttentionMechanism()
        self.metacognition = Metacognition()
        self.pattern_abstraction = PatternAbstraction()
        
        logger.info("[CognitiveArchitecture] Initialized brain system")
    
    def process(self, input_data: Any, task: str = "") -> Dict[str, Any]:
        """
        Process input through cognitive architecture
        
        Parameters
        ----------
        input_data : any
            Input data
        task : str
            Task description
            
        Returns
        -------
        result : dict
            Processing result
        """
        # 1. Attention: Focus on relevant information
        if isinstance(input_data, list):
            focused = self.attention.get_focused_items(input_data, query=task)
        else:
            focused = [input_data]
        
        # 2. Working memory: Add to active processing
        for i, item in enumerate(focused[:3]):  # Top 3
            self.working_memory.add(f"chunk_{i}", item, chunk_type="input")
        
        # 3. Retrieve relevant memories
        episodic_context = self.episodic_memory.search_events(task)
        semantic_context = self.semantic_memory.search(task)
        
        # 4. Metacognition: Self-assess
        self_awareness = self.metacognition.get_self_report()
        
        # 5. Pattern matching
        if isinstance(input_data, dict):
            patterns = self.pattern_abstraction.get_patterns()
            pattern_matches = {
                name: self.pattern_abstraction.match_pattern(input_data, name)
                for name in patterns.keys()
            }
        else:
            pattern_matches = {}
        
        # 6. Remember this processing event
        event_id = self.episodic_memory.remember_event(
            what=f"Processed: {task}",
            where="cognitive_architecture",
            importance=0.7
        )
        
        return {
            'focused_input': focused,
            'working_memory_chunks': len(self.working_memory.chunks),
            'episodic_context': len(episodic_context),
            'semantic_context': len(semantic_context),
            'self_awareness': self_awareness,
            'pattern_matches': pattern_matches,
            'event_id': event_id
        }
    
    def think(self, problem: str) -> Dict[str, Any]:
        """
        Think about problem (cognitive processing)
        
        Parameters
        ----------
        problem : str
            Problem description
            
        Returns
        -------
        thinking : dict
            Thinking process result
        """
        # Add problem to working memory
        self.working_memory.add("current_problem", problem, chunk_type="goal")
        
        # Retrieve relevant knowledge
        relevant_episodes = self.episodic_memory.search_events(problem)
        relevant_semantic = self.semantic_memory.search(problem)
        
        # Apply attention
        all_context = [e.what for e in relevant_episodes] + relevant_semantic
        focused_context = self.attention.get_focused_items(all_context, query=problem)
        
        # Metacognitive monitoring
        monitoring = self.metacognition.monitor_thinking(problem, {
            'context_retrieved': len(focused_context),
            'episodes_found': len(relevant_episodes)
        })
        
        return {
            'problem': problem,
            'focused_context': focused_context,
            'relevant_episodes': len(relevant_episodes),
            'monitoring': monitoring,
            'working_memory_load': self.working_memory.cognitive_load.current_load
        }
    
    def learn(self, experience: Dict[str, Any]):
        """
        Learn from experience
        
        Parameters
        ----------
        experience : dict
            Experience data
        """
        # Remember as episode
        event_id = self.episodic_memory.remember_event(
            what=experience.get('what', 'experience'),
            outcome=experience.get('outcome'),
            importance=experience.get('importance', 0.5)
        )
        
        # Consolidate to semantic if important
        event = self.episodic_memory.recall_event(event_id)
        if event and self.episodic_memory.consolidation.should_consolidate(event):
            semantic_knowledge = self.episodic_memory.consolidation.consolidate(event)
            self.semantic_memory.add_fact(
                semantic_knowledge['fact'],
                semantic_knowledge.get('context', '')
            )
        
        # Abstract patterns if multiple similar experiences
        # (simplified - would need more sophisticated pattern detection)
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state"""
        return {
            'working_memory': {
                'chunks': len(self.working_memory.chunks),
                'load': self.working_memory.cognitive_load.current_load,
                'capacity': self.working_memory.cognitive_load.max_capacity
            },
            'episodic_memory': {
                'events': len(self.episodic_memory.events),
                'consolidated': self.episodic_memory.consolidation.consolidated_count
            },
            'semantic_memory': {
                'facts': len(self.semantic_memory.facts)
            },
            'attention': self.attention.get_attention_stats(),
            'metacognition': self.metacognition.get_self_report(),
            'patterns': len(self.pattern_abstraction.get_patterns())
        }


class BrainSystem:
    """
    Brain System - Simplified interface
    
    Easy-to-use brain system for agents
    """
    
    def __init__(self, capacity: int = 7):
        """Initialize brain system"""
        self.brain = CognitiveArchitecture(working_memory_capacity=capacity)
    
    def think(self, problem: str) -> Dict[str, Any]:
        """Think about problem"""
        return self.brain.think(problem)
    
    def remember(self, what: str, importance: float = 0.5):
        """Remember something"""
        self.brain.episodic_memory.remember_event(what, importance=importance)
    
    def recall(self, query: str) -> List[str]:
        """Recall from memory"""
        events = self.brain.episodic_memory.search_events(query)
        return [e.what for e in events[:5]]  # Top 5
    
    def get_state(self) -> Dict[str, Any]:
        """Get brain state"""
        return self.brain.get_brain_state()
