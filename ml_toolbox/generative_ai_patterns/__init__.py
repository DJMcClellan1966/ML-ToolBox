"""
Generative AI Design Patterns

Implements reusable patterns for generative AI applications:
- Pattern Catalog
- Pattern Composition Strategies
- Pattern Reuse & Inheritance
"""
try:
    from .pattern_catalog import PatternCatalog, PatternLibrary
    from .pattern_composition import PatternCompositionStrategy, PatternOrchestrator, CompositionStrategy
    __all__ = [
        'PatternCatalog',
        'PatternLibrary',
        'PatternCompositionStrategy',
        'PatternOrchestrator',
        'CompositionStrategy'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Generative AI Patterns not available: {e}")
    __all__ = []
