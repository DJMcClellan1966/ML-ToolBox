"""
Agent Brain - Brain-Like Cognitive Features

Inspired by cognitive architectures (ACT-R, SOAR):
- Working Memory (active problem-solving)
- Episodic Memory (event-based)
- Semantic Memory (factual knowledge)
- Attention Mechanisms (focus and filtering)
- Metacognition (self-awareness)
- Pattern Abstraction (generalization)
- Cognitive Load Management
"""
try:
    from .working_memory import WorkingMemory, CognitiveLoad
    from .episodic_memory import EpisodicMemory, MemoryConsolidation
    from .semantic_memory import SemanticMemory, KnowledgeBase
    from .attention_mechanism import AttentionMechanism, FocusFilter
    from .metacognition import Metacognition, SelfAwareness
    from .pattern_abstraction import PatternAbstraction, ConceptFormation
    from .cognitive_architecture import CognitiveArchitecture, BrainSystem
    __all__ = [
        'WorkingMemory',
        'CognitiveLoad',
        'EpisodicMemory',
        'MemoryConsolidation',
        'SemanticMemory',
        'KnowledgeBase',
        'AttentionMechanism',
        'FocusFilter',
        'Metacognition',
        'SelfAwareness',
        'PatternAbstraction',
        'ConceptFormation',
        'CognitiveArchitecture',
        'BrainSystem'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Agent Brain not available: {e}")
    __all__ = []
