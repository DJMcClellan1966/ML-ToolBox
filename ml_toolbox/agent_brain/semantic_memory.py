"""
Semantic Memory - Factual Knowledge Memory

Brain-like semantic memory for storing and retrieving facts
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticFact:
    """Semantic memory fact"""
    fact: str
    context: str = ""
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    associations: List[str] = field(default_factory=list)


class KnowledgeBase:
    """
    Knowledge Base
    
    Stores semantic knowledge
    """
    
    def __init__(self):
        self.facts: Dict[str, SemanticFact] = {}
        self.fact_counter = 0
    
    def add_fact(self, fact: str, context: str = "", confidence: float = 1.0,
                source: str = "") -> str:
        """Add fact to knowledge base"""
        self.fact_counter += 1
        fact_id = f"fact_{self.fact_counter}"
        
        semantic_fact = SemanticFact(
            fact=fact,
            context=context,
            confidence=confidence,
            source=source
        )
        
        self.facts[fact_id] = semantic_fact
        logger.debug(f"[KnowledgeBase] Added fact: {fact_id}")
        return fact_id
    
    def search(self, query: str) -> List[str]:
        """Search facts by query"""
        query_lower = query.lower()
        results = []
        
        for fact in self.facts.values():
            if query_lower in fact.fact.lower() or query_lower in fact.context.lower():
                results.append(fact.fact)
        
        return results


class SemanticMemory:
    """
    Semantic Memory
    
    Factual knowledge memory (like human semantic memory)
    """
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.facts: Dict[str, SemanticFact] = {}
    
    def add_fact(self, fact: str, context: str = "", confidence: float = 1.0,
                source: str = "") -> str:
        """Add fact to semantic memory"""
        return self.knowledge_base.add_fact(fact, context, confidence, source)
    
    def search(self, query: str) -> List[str]:
        """Search semantic memory"""
        return self.knowledge_base.search(query)
    
    def get_fact(self, fact_id: str) -> Optional[SemanticFact]:
        """Get fact by ID"""
        return self.knowledge_base.facts.get(fact_id)
    
    def get_all_facts(self) -> List[str]:
        """Get all facts"""
        return [fact.fact for fact in self.knowledge_base.facts.values()]
