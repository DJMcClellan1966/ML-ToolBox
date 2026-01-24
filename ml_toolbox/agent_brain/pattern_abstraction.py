"""
Pattern Abstraction - Generalization and Concept Formation

Brain-like pattern abstraction and concept formation
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AbstractConcept:
    """Abstract concept"""
    concept_id: str
    name: str
    examples: List[Any] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    abstraction_level: int = 1  # 1=concrete, higher=more abstract
    confidence: float = 1.0


class ConceptFormation:
    """
    Concept Formation
    
    Forms abstract concepts from specific examples
    """
    
    def __init__(self):
        self.concepts: Dict[str, AbstractConcept] = {}
        self.concept_counter = 0
    
    def form_concept(self, examples: List[Any], concept_name: Optional[str] = None) -> AbstractConcept:
        """
        Form abstract concept from examples
        
        Parameters
        ----------
        examples : list
            Specific examples
        concept_name : str, optional
            Concept name
            
        Returns
        -------
        concept : AbstractConcept
            Formed concept
        """
        self.concept_counter += 1
        concept_id = f"concept_{self.concept_counter}"
        
        if not concept_name:
            concept_name = f"Concept_{self.concept_counter}"
        
        # Extract common properties
        properties = self._extract_common_properties(examples)
        
        concept = AbstractConcept(
            concept_id=concept_id,
            name=concept_name,
            examples=examples,
            properties=properties,
            abstraction_level=1,
            confidence=1.0
        )
        
        self.concepts[concept_id] = concept
        logger.info(f"[ConceptFormation] Formed concept: {concept_name}")
        return concept
    
    def _extract_common_properties(self, examples: List[Any]) -> Dict[str, Any]:
        """Extract common properties from examples"""
        if not examples:
            return {}
        
        # Simple: if all examples are strings, find common words
        if all(isinstance(ex, str) for ex in examples):
            words_list = [ex.lower().split() for ex in examples]
            if words_list:
                common_words = set(words_list[0])
                for words in words_list[1:]:
                    common_words &= set(words)
                return {'common_words': list(common_words)}
        
        return {'type': type(examples[0]).__name__}


class PatternAbstraction:
    """
    Pattern Abstraction
    
    Generalizes patterns from specific instances (like brain)
    """
    
    def __init__(self):
        self.concept_formation = ConceptFormation()
        self.patterns: Dict[str, Dict] = {}
    
    def abstract_pattern(self, instances: List[Dict[str, Any]], pattern_name: str) -> Dict[str, Any]:
        """
        Abstract pattern from instances
        
        Parameters
        ----------
        instances : list
            Specific instances
        pattern_name : str
            Pattern name
            
        Returns
        -------
        pattern : dict
            Abstracted pattern
        """
        if not instances:
            return {}
        
        # Find common structure
        common_keys = set(instances[0].keys())
        for instance in instances[1:]:
            common_keys &= set(instance.keys())
        
        # Extract common values
        pattern = {
            'name': pattern_name,
            'common_structure': list(common_keys),
            'instance_count': len(instances),
            'variations': {}
        }
        
        # Find variations
        for key in common_keys:
            values = [inst.get(key) for inst in instances]
            unique_values = list(set(values))
            if len(unique_values) == 1:
                pattern['variations'][key] = {'type': 'constant', 'value': unique_values[0]}
            else:
                pattern['variations'][key] = {'type': 'variable', 'values': unique_values}
        
        self.patterns[pattern_name] = pattern
        logger.info(f"[PatternAbstraction] Abstracted pattern: {pattern_name}")
        return pattern
    
    def match_pattern(self, instance: Dict[str, Any], pattern_name: str) -> float:
        """
        Match instance to pattern
        
        Parameters
        ----------
        instance : dict
            Instance to match
        pattern_name : str
            Pattern name
            
        Returns
        -------
        match_score : float
            Match score (0-1)
        """
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return 0.0
        
        # Check structure match
        instance_keys = set(instance.keys())
        pattern_keys = set(pattern['common_structure'])
        
        if not pattern_keys.issubset(instance_keys):
            return 0.0
        
        # Check value matches
        matches = 0
        total = len(pattern['variations'])
        
        for key, variation in pattern['variations'].items():
            if variation['type'] == 'constant':
                if instance.get(key) == variation['value']:
                    matches += 1
            else:  # variable
                if instance.get(key) in variation['values']:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def get_patterns(self) -> Dict[str, Dict]:
        """Get all patterns"""
        return dict(self.patterns)
