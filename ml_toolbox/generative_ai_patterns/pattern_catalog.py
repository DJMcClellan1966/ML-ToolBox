"""
Pattern Catalog - Reusable Pattern Library

From Generative AI Design Patterns
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PatternCategory(Enum):
    """Pattern categories"""
    PROMPT = "prompt"
    RAG = "rag"
    CHAIN = "chain"
    AGENT = "agent"
    DEPLOYMENT = "deployment"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"


@dataclass
class Pattern:
    """Reusable pattern definition"""
    id: str
    name: str
    category: PatternCategory
    description: str
    template: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'template': self.template,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'examples': self.examples,
            'version': self.version,
            'author': self.author,
            'tags': self.tags
        }


class PatternCatalog:
    """
    Pattern Catalog - Central repository for reusable patterns
    
    From Generative AI Design Patterns
    """
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.categories: Dict[PatternCategory, List[str]] = {
            cat: [] for cat in PatternCategory
        }
        self._initialize_common_patterns()
    
    def _initialize_common_patterns(self):
        """Initialize common patterns"""
        
        # Prompt Patterns
        self.add_pattern(Pattern(
            id="prompt_zero_shot",
            name="Zero-Shot Prompting",
            category=PatternCategory.PROMPT,
            description="Direct prompt without examples",
            template="Task: {task}\nInput: {input}\nOutput:",
            parameters={'task': str, 'input': str},
            tags=['prompt', 'zero-shot']
        ))
        
        self.add_pattern(Pattern(
            id="prompt_few_shot",
            name="Few-Shot Prompting",
            category=PatternCategory.PROMPT,
            description="Prompt with examples",
            template="Task: {task}\n\nExamples:\n{examples}\n\nInput: {input}\nOutput:",
            parameters={'task': str, 'examples': list, 'input': str},
            tags=['prompt', 'few-shot']
        ))
        
        self.add_pattern(Pattern(
            id="prompt_chain_of_thought",
            name="Chain-of-Thought Prompting",
            category=PatternCategory.PROMPT,
            description="Step-by-step reasoning",
            template="Task: {task}\nInput: {input}\n\nLet's think step by step:\n",
            parameters={'task': str, 'input': str},
            tags=['prompt', 'reasoning', 'cot']
        ))
        
        # RAG Patterns
        self.add_pattern(Pattern(
            id="rag_basic",
            name="Basic RAG",
            category=PatternCategory.RAG,
            description="Retrieve and augment generation",
            template="Retrieve relevant context:\n{context}\n\nQuestion: {query}\nAnswer:",
            parameters={'context': str, 'query': str},
            dependencies=['prompt_zero_shot'],
            tags=['rag', 'retrieval']
        ))
        
        self.add_pattern(Pattern(
            id="rag_rerank",
            name="RAG with Reranking",
            category=PatternCategory.RAG,
            description="RAG with result reranking",
            template="Retrieved contexts:\n{contexts}\n\nRerank by relevance:\n{reranked}\n\nQuestion: {query}\nAnswer:",
            parameters={'contexts': list, 'reranked': list, 'query': str},
            dependencies=['rag_basic'],
            tags=['rag', 'reranking']
        ))
        
        # Chain Patterns
        self.add_pattern(Pattern(
            id="chain_sequential",
            name="Sequential Chain",
            category=PatternCategory.CHAIN,
            description="Chain operations sequentially",
            template="Step 1: {step1}\nStep 2: {step2}\nFinal: {final}",
            parameters={'step1': str, 'step2': str, 'final': str},
            tags=['chain', 'sequential']
        ))
        
        self.add_pattern(Pattern(
            id="chain_parallel",
            name="Parallel Chain",
            category=PatternCategory.CHAIN,
            description="Execute operations in parallel",
            template="Parallel tasks:\n{task1}\n{task2}\nCombine: {combine}",
            parameters={'task1': str, 'task2': str, 'combine': str},
            tags=['chain', 'parallel']
        ))
        
        # Agent Patterns
        self.add_pattern(Pattern(
            id="agent_react",
            name="ReAct Agent",
            category=PatternCategory.AGENT,
            description="Reasoning and Acting agent",
            template="Thought: {thought}\nAction: {action}\nObservation: {observation}\n",
            parameters={'thought': str, 'action': str, 'observation': str},
            tags=['agent', 'react']
        ))
        
        # Deployment Patterns
        self.add_pattern(Pattern(
            id="deploy_api",
            name="API Deployment",
            category=PatternCategory.DEPLOYMENT,
            description="Deploy as API endpoint",
            template="API Endpoint: {endpoint}\nModel: {model}\nVersion: {version}",
            parameters={'endpoint': str, 'model': str, 'version': str},
            tags=['deployment', 'api']
        ))
    
    def add_pattern(self, pattern: Pattern):
        """Add pattern to catalog"""
        self.patterns[pattern.id] = pattern
        self.categories[pattern.category].append(pattern.id)
        logger.info(f"[PatternCatalog] Added pattern: {pattern.id}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def search_patterns(self, query: str, category: Optional[PatternCategory] = None) -> List[Pattern]:
        """Search patterns by query"""
        results = []
        query_lower = query.lower()
        
        for pattern in self.patterns.values():
            if category and pattern.category != category:
                continue
            
            # Search in name, description, tags
            if (query_lower in pattern.name.lower() or
                query_lower in pattern.description.lower() or
                any(query_lower in tag.lower() for tag in pattern.tags)):
                results.append(pattern)
        
        return results
    
    def get_patterns_by_category(self, category: PatternCategory) -> List[Pattern]:
        """Get all patterns in category"""
        return [self.patterns[pid] for pid in self.categories[category]]
    
    def get_dependencies(self, pattern_id: str) -> List[str]:
        """Get pattern dependencies"""
        pattern = self.get_pattern(pattern_id)
        if pattern:
            return pattern.dependencies
        return []


class PatternLibrary:
    """
    Pattern Library - Extended catalog with versioning and inheritance
    """
    
    def __init__(self):
        self.catalog = PatternCatalog()
        self.versions: Dict[str, List[str]] = {}  # pattern_id -> [versions]
        self.inheritance: Dict[str, str] = {}  # child_id -> parent_id
    
    def create_pattern_variant(self, base_pattern_id: str, variant_id: str,
                              modifications: Dict[str, Any]) -> Pattern:
        """Create pattern variant (inheritance)"""
        base = self.catalog.get_pattern(base_pattern_id)
        if not base:
            raise ValueError(f"Base pattern not found: {base_pattern_id}")
        
        # Create variant
        variant = Pattern(
            id=variant_id,
            name=modifications.get('name', f"{base.name} (Variant)"),
            category=base.category,
            description=modifications.get('description', base.description),
            template=modifications.get('template', base.template),
            parameters={**base.parameters, **modifications.get('parameters', {})},
            dependencies=[base_pattern_id] + base.dependencies,
            examples=modifications.get('examples', base.examples),
            version="1.0.0",
            author=modifications.get('author', base.author),
            tags=base.tags + modifications.get('tags', [])
        )
        
        self.catalog.add_pattern(variant)
        self.inheritance[variant_id] = base_pattern_id
        
        logger.info(f"[PatternLibrary] Created variant: {variant_id} from {base_pattern_id}")
        return variant
    
    def get_pattern_lineage(self, pattern_id: str) -> List[str]:
        """Get pattern inheritance lineage"""
        lineage = [pattern_id]
        current = pattern_id
        
        while current in self.inheritance:
            parent = self.inheritance[current]
            lineage.append(parent)
            current = parent
        
        return lineage
