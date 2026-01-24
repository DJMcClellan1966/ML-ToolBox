"""
Pattern Composition Strategies

From Generative AI Design Patterns
"""
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

from .pattern_catalog import PatternCatalog, Pattern


class CompositionStrategy(Enum):
    """Composition strategies"""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"  # Simultaneous
    CONDITIONAL = "conditional"  # Based on conditions
    LOOP = "loop"  # Iterative
    PIPELINE = "pipeline"  # Data flows through


class PatternCompositionStrategy:
    """
    Pattern Composition Strategy
    
    Composes multiple patterns into workflows
    """
    
    def __init__(self, catalog: PatternCatalog):
        self.catalog = catalog
        self.compositions: Dict[str, Dict] = {}
    
    def compose(self, pattern_ids: List[str], strategy: CompositionStrategy,
               context: Optional[Dict] = None) -> str:
        """
        Compose patterns using strategy
        
        Parameters
        ----------
        pattern_ids : list
            List of pattern IDs to compose
        strategy : CompositionStrategy
            Composition strategy
        context : dict, optional
            Context variables
            
        Returns
        -------
        composed : str
            Composed pattern code/template
        """
        patterns = [self.catalog.get_pattern(pid) for pid in pattern_ids]
        patterns = [p for p in patterns if p is not None]
        
        if not patterns:
            return ""
        
        if strategy == CompositionStrategy.SEQUENTIAL:
            return self._compose_sequential(patterns, context)
        elif strategy == CompositionStrategy.PARALLEL:
            return self._compose_parallel(patterns, context)
        elif strategy == CompositionStrategy.CONDITIONAL:
            return self._compose_conditional(patterns, context)
        elif strategy == CompositionStrategy.LOOP:
            return self._compose_loop(patterns, context)
        elif strategy == CompositionStrategy.PIPELINE:
            return self._compose_pipeline(patterns, context)
        else:
            return self._compose_sequential(patterns, context)
    
    def _compose_sequential(self, patterns: List[Pattern], context: Optional[Dict]) -> str:
        """Sequential composition"""
        parts = []
        for i, pattern in enumerate(patterns):
            part = pattern.template
            if context:
                try:
                    part = part.format(**context)
                except:
                    pass
            parts.append(f"# Step {i+1}: {pattern.name}\n{part}")
        
        return "\n\n".join(parts)
    
    def _compose_parallel(self, patterns: List[Pattern], context: Optional[Dict]) -> str:
        """Parallel composition"""
        parts = []
        for pattern in patterns:
            part = pattern.template
            if context:
                try:
                    part = part.format(**context)
                except:
                    pass
            parts.append(f"# Parallel: {pattern.name}\n{part}")
        
        return "\n\n".join(parts) + "\n\n# Combine results"
    
    def _compose_conditional(self, patterns: List[Pattern], context: Optional[Dict]) -> str:
        """Conditional composition"""
        if len(patterns) < 2:
            return self._compose_sequential(patterns, context)
        
        condition = context.get('condition', 'condition_met') if context else 'condition_met'
        true_pattern = patterns[0]
        false_pattern = patterns[1] if len(patterns) > 1 else patterns[0]
        
        true_part = true_pattern.template
        false_part = false_pattern.template
        
        return f"if {condition}:\n    {true_part}\nelse:\n    {false_part}"
    
    def _compose_loop(self, patterns: List[Pattern], context: Optional[Dict]) -> str:
        """Loop composition"""
        if not patterns:
            return ""
        
        pattern = patterns[0]
        loop_var = context.get('loop_var', 'item') if context else 'item'
        iterable = context.get('iterable', 'items') if context else 'items'
        
        part = pattern.template
        return f"for {loop_var} in {iterable}:\n    {part}"
    
    def _compose_pipeline(self, patterns: List[Pattern], context: Optional[Dict]) -> str:
        """Pipeline composition"""
        parts = []
        data_var = context.get('data_var', 'data') if context else 'data'
        
        for i, pattern in enumerate(patterns):
            part = pattern.template
            if i > 0:
                # Pass output from previous step
                part = part.replace('{input}', f'{data_var}')
            
            parts.append(f"# Pipeline Step {i+1}: {pattern.name}\n{data_var} = {part}")
        
        return "\n\n".join(parts)


class PatternOrchestrator:
    """
    Pattern Orchestrator
    
    Orchestrates pattern execution and composition
    """
    
    def __init__(self, catalog: PatternCatalog):
        self.catalog = catalog
        self.composer = PatternCompositionStrategy(catalog)
        self.execution_history: List[Dict] = []
    
    def execute_pattern(self, pattern_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single pattern
        
        Parameters
        ----------
        pattern_id : str
            Pattern ID
        inputs : dict
            Input variables
            
        Returns
        -------
        result : dict
            Execution result
        """
        pattern = self.catalog.get_pattern(pattern_id)
        if not pattern:
            return {'error': f'Pattern not found: {pattern_id}'}
        
        try:
            # Format template
            output = pattern.template.format(**inputs)
            
            result = {
                'pattern_id': pattern_id,
                'output': output,
                'success': True
            }
            
            self.execution_history.append(result)
            return result
        except Exception as e:
            result = {
                'pattern_id': pattern_id,
                'error': str(e),
                'success': False
            }
            self.execution_history.append(result)
            return result
    
    def execute_workflow(self, pattern_ids: List[str], strategy: CompositionStrategy,
                        inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow of patterns
        
        Parameters
        ----------
        pattern_ids : list
            Pattern IDs in workflow
        strategy : CompositionStrategy
            Composition strategy
        inputs : dict
            Input variables
            
        Returns
        -------
        result : dict
            Workflow execution result
        """
        # Check dependencies
        all_patterns = []
        for pid in pattern_ids:
            pattern = self.catalog.get_pattern(pid)
            if pattern:
                all_patterns.append(pattern)
                # Add dependencies
                for dep_id in pattern.dependencies:
                    dep_pattern = self.catalog.get_pattern(dep_id)
                    if dep_pattern and dep_pattern not in all_patterns:
                        all_patterns.insert(0, dep_pattern)
        
        # Compose
        composed = self.composer.compose([p.id for p in all_patterns], strategy, inputs)
        
        return {
            'workflow': pattern_ids,
            'strategy': strategy.value,
            'composed_code': composed,
            'success': True
        }
