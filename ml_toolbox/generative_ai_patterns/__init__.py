"""
Generative AI Design Patterns

Implements reusable patterns for generative AI applications:
- Pattern Catalog
- Pattern Composition Strategies
- Pattern Validation
- Pattern Versioning
- Pattern Reuse & Inheritance
"""
try:
    from .pattern_catalog import PatternCatalog, PatternLibrary
    from .pattern_composition import PatternCompositionStrategy, PatternOrchestrator
    from .pattern_validation import PatternValidator, PatternTester
    from .pattern_versioning import PatternVersioning, PatternRegistry
    __all__ = [
        'PatternCatalog',
        'PatternLibrary',
        'PatternCompositionStrategy',
        'PatternOrchestrator',
        'PatternValidator',
        'PatternTester',
        'PatternVersioning',
        'PatternRegistry'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Generative AI Patterns not available: {e}")
    __all__ = []
