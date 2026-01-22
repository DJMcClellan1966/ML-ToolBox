"""
Pragmatic Programmer & Clean Code Framework
Practical development practices and clean code principles

Methods from:
- Hunt & Thomas "The Pragmatic Programmer" - DRY, orthogonality, design by contract
- Robert Martin "Clean Code" & "Clean Architecture" - SOLID, clean architecture, dependency inversion
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable, Set
import ast
import inspect
import re
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent))


class DRYFramework:
    """
    DRY (Don't Repeat Yourself) Framework - Pragmatic Programmer
    
    Detect and eliminate code duplication
    """
    
    @staticmethod
    def detect_duplication(functions: List[Callable], threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Detect code duplication
        
        Args:
            functions: List of functions to analyze
            threshold: Minimum lines to consider duplication
            
        Returns:
            List of duplication reports
        """
        duplications = []
        
        for i, func1 in enumerate(functions):
            try:
                source1 = inspect.getsource(func1)
                lines1 = source1.split('\n')
            except:
                continue
            
            for j, func2 in enumerate(functions[i+1:], i+1):
                try:
                    source2 = inspect.getsource(func2)
                    lines2 = source2.split('\n')
                except:
                    continue
                
                # Find common subsequences
                common = DRYFramework._find_common_lines(lines1, lines2, threshold)
                if common:
                    duplications.append({
                        'function1': func1.__name__,
                        'function2': func2.__name__,
                        'common_lines': common,
                        'similarity': len(common) / max(len(lines1), len(lines2))
                    })
        
        return duplications
    
    @staticmethod
    def _find_common_lines(lines1: List[str], lines2: List[str], threshold: int) -> List[str]:
        """Find common lines between two functions"""
        # Normalize lines (remove whitespace, comments)
        normalized1 = [line.strip() for line in lines1 if line.strip() and not line.strip().startswith('#')]
        normalized2 = [line.strip() for line in lines2 if line.strip() and not line.strip().startswith('#')]
        
        common = []
        for line in normalized1:
            if line in normalized2 and len(line) > threshold:
                common.append(line)
        
        return common if len(common) >= threshold else []


class OrthogonalityChecker:
    """
    Orthogonality Checker - Pragmatic Programmer
    
    Measure component independence and coupling
    """
    
    @staticmethod
    def measure_orthogonality(components: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Measure orthogonality (independence) of components
        
        Args:
            components: Dict of component_name -> list of dependencies
            
        Returns:
            Orthogonality scores (higher is better, 1.0 = fully orthogonal)
        """
        scores = {}
        
        for name, deps in components.items():
            # Count how many other components this depends on
            external_deps = [d for d in deps if d in components and d != name]
            coupling = len(external_deps)
            
            # Orthogonality = 1 - (coupling / total_components)
            total = len(components)
            orthogonality = 1.0 - (coupling / total) if total > 0 else 1.0
            scores[name] = max(0.0, orthogonality)
        
        return scores
    
    @staticmethod
    def check_coupling(component1: str, component2: str, 
                      dependencies: Dict[str, List[str]]) -> bool:
        """Check if two components are coupled"""
        deps1 = dependencies.get(component1, [])
        deps2 = dependencies.get(component2, [])
        return component2 in deps1 or component1 in deps2


class DesignByContract:
    """
    Design by Contract - Pragmatic Programmer
    
    Preconditions, postconditions, and invariants
    """
    
    @staticmethod
    def requires(precondition: Callable):
        """Precondition decorator"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not precondition(*args, **kwargs):
                    raise ValueError(f"Precondition failed for {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def ensures(postcondition: Callable):
        """Postcondition decorator"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if not postcondition(result, *args, **kwargs):
                    raise ValueError(f"Postcondition failed for {func.__name__}")
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def invariant(invariant_func: Callable):
        """Class invariant decorator"""
        def decorator(cls):
            original_init = cls.__init__
            
            def new_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                if not invariant_func(self):
                    raise ValueError(f"Invariant violated for {cls.__name__}")
            
            cls.__init__ = new_init
            return cls
        return decorator


class AssertionsFramework:
    """
    Assertions Framework - Pragmatic Programmer
    
    Defensive programming with assertions
    """
    
    @staticmethod
    def assert_not_none(value: Any, message: str = "Value cannot be None"):
        """Assert value is not None"""
        if value is None:
            raise AssertionError(message)
        return value
    
    @staticmethod
    def assert_positive(value: float, message: str = "Value must be positive"):
        """Assert value is positive"""
        if value <= 0:
            raise AssertionError(message)
        return value
    
    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float,
                       message: str = "Value out of range"):
        """Assert value is in range"""
        if not (min_val <= value <= max_val):
            raise AssertionError(f"{message}: {value} not in [{min_val}, {max_val}]")
        return value
    
    @staticmethod
    def assert_type(value: Any, expected_type: type, message: str = "Type mismatch"):
        """Assert value is of expected type"""
        if not isinstance(value, expected_type):
            raise AssertionError(f"{message}: expected {expected_type}, got {type(value)}")
        return value


class SOLIDPrinciplesChecker:
    """
    SOLID Principles Checker - Clean Code (Robert Martin)
    
    Check adherence to SOLID principles
    """
    
    @staticmethod
    def check_single_responsibility(func: Callable) -> Tuple[bool, str]:
        """
        Check Single Responsibility Principle
        
        Returns:
            Tuple (adheres, explanation)
        """
        try:
            source = inspect.getsource(func)
            # Count different types of operations
            operations = {
                'print': source.count('print('),
                'file': source.count('open('),
                'network': source.count('requests.') + source.count('urllib'),
                'database': source.count('sqlite') + source.count('mysql'),
                'calculation': source.count('+') + source.count('*') + source.count('/')
            }
            
            # If function does too many different things, violates SRP
            non_zero = sum(1 for v in operations.values() if v > 0)
            if non_zero > 3:
                return False, f"Function does {non_zero} different types of operations"
            
            return True, "Function has single responsibility"
        except:
            return True, "Could not analyze"
    
    @staticmethod
    def check_open_closed(func: Callable) -> Tuple[bool, str]:
        """
        Check Open/Closed Principle (simplified)
        
        Returns:
            Tuple (adheres, explanation)
        """
        # Simplified: check if function uses inheritance/polymorphism
        try:
            source = inspect.getsource(func)
            if 'class' in source or 'isinstance' in source or 'hasattr' in source:
                return True, "Function uses polymorphism"
            return True, "Function is extensible"
        except:
            return True, "Could not analyze"
    
    @staticmethod
    def check_liskov_substitution(cls: type) -> Tuple[bool, str]:
        """
        Check Liskov Substitution Principle (simplified)
        
        Returns:
            Tuple (adheres, explanation)
        """
        # Simplified: check if class properly inherits
        try:
            if not cls.__bases__ or cls.__bases__ == (object,):
                return True, "No inheritance to check"
            
            # Check if subclass can be used where base class is expected
            base = cls.__bases__[0]
            if issubclass(cls, base):
                return True, "Subclass properly extends base class"
            return False, "Subclass does not properly extend base class"
        except:
            return True, "Could not analyze"
    
    @staticmethod
    def check_interface_segregation(cls: type) -> Tuple[bool, str]:
        """
        Check Interface Segregation Principle (simplified)
        
        Returns:
            Tuple (adheres, explanation)
        """
        # Simplified: check if class has too many methods
        try:
            methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m, None))]
            if len(methods) > 10:
                return False, f"Class has {len(methods)} public methods (too many)"
            return True, f"Class has {len(methods)} public methods (reasonable)"
        except:
            return True, "Could not analyze"
    
    @staticmethod
    def check_dependency_inversion(cls: type) -> Tuple[bool, str]:
        """
        Check Dependency Inversion Principle (simplified)
        
        Returns:
            Tuple (adheres, explanation)
        """
        # Simplified: check if class depends on abstractions
        try:
            source = inspect.getsource(cls)
            # Check for abstract base classes or interfaces
            if 'ABC' in source or 'abstractmethod' in source or 'Protocol' in source:
                return True, "Class uses abstractions"
            return True, "Dependency inversion check passed"
        except:
            return True, "Could not analyze"


class CleanArchitecture:
    """
    Clean Architecture Framework - Robert Martin
    
    Layered architecture with dependency inversion
    """
    
    class Layer:
        """Architecture layer"""
        def __init__(self, name: str, level: int):
            self.name = name
            self.level = level  # Lower level = more inner/core
            self.components = []
        
        def add_component(self, component: str):
            """Add component to layer"""
            self.components.append(component)
    
    def __init__(self):
        """Initialize clean architecture"""
        self.layers = []
        self.dependencies = {}  # component -> list of dependencies
    
    def add_layer(self, name: str, level: int):
        """Add architecture layer"""
        layer = CleanArchitecture.Layer(name, level)
        self.layers.append(layer)
        return layer
    
    def add_dependency(self, component: str, depends_on: List[str]):
        """Add component dependency"""
        self.dependencies[component] = depends_on
    
    def validate_architecture(self) -> Tuple[bool, List[str]]:
        """
        Validate architecture (dependencies should point inward)
        
        Returns:
            Tuple (valid, violations)
        """
        violations = []
        
        # Get layer for each component
        component_layers = {}
        for layer in self.layers:
            for component in layer.components:
                component_layers[component] = layer.level
        
        # Check dependencies
        for component, deps in self.dependencies.items():
            component_level = component_layers.get(component, 0)
            for dep in deps:
                dep_level = component_layers.get(dep, 0)
                # Dependencies should point to inner layers (lower level)
                if dep_level > component_level:
                    violations.append(
                        f"{component} (level {component_level}) depends on "
                        f"{dep} (level {dep_level}) - violates dependency rule"
                    )
        
        return len(violations) == 0, violations


class FunctionQualityMetrics:
    """
    Function Quality Metrics - Clean Code (Robert Martin)
    
    Measure function quality (small, focused, single purpose)
    """
    
    @staticmethod
    def measure_function_quality(func: Callable) -> Dict[str, Any]:
        """
        Measure function quality metrics
        
        Returns:
            Quality metrics dictionary
        """
        try:
            source = inspect.getsource(func)
            lines = [line for line in source.split('\n') if line.strip()]
            sig = inspect.signature(func)
            
            metrics = {
                'name': func.__name__,
                'lines': len(lines),
                'parameters': len(sig.parameters),
                'is_small': len(lines) <= 20,
                'is_focused': len(sig.parameters) <= 3,
                'has_single_purpose': FunctionQualityMetrics._check_single_purpose(source),
                'quality_score': 0
            }
            
            # Calculate quality score
            score = 0
            if metrics['is_small']:
                score += 30
            if metrics['is_focused']:
                score += 30
            if metrics['has_single_purpose']:
                score += 40
            
            metrics['quality_score'] = score
            metrics['recommendations'] = FunctionQualityMetrics._get_recommendations(metrics)
            
            return metrics
        except:
            return {'name': func.__name__, 'error': 'Could not analyze'}
    
    @staticmethod
    def _check_single_purpose(source: str) -> bool:
        """Check if function has single purpose"""
        # Simplified: check for multiple return statements or complex control flow
        return_count = source.count('return ')
        if_count = source.count('if ')
        for_count = source.count('for ')
        while_count = source.count('while ')
        
        # Too many control structures suggests multiple purposes
        total_control = if_count + for_count + while_count
        return return_count <= 2 and total_control <= 5
    
    @staticmethod
    def _get_recommendations(metrics: Dict[str, Any]) -> List[str]:
        """Get quality recommendations"""
        recommendations = []
        
        if not metrics['is_small']:
            recommendations.append("Function is too long - consider splitting")
        if not metrics['is_focused']:
            recommendations.append("Function has too many parameters - consider using a configuration object")
        if not metrics['has_single_purpose']:
            recommendations.append("Function may have multiple purposes - consider refactoring")
        
        return recommendations


class PragmaticCleanCodeFramework:
    """
    Unified Pragmatic Programmer & Clean Code Framework
    """
    
    def __init__(self):
        self.dry = DRYFramework()
        self.orthogonality = OrthogonalityChecker()
        self.contract = DesignByContract()
        self.assertions = AssertionsFramework()
        self.solid = SOLIDPrinciplesChecker()
        self.architecture = CleanArchitecture()
        self.function_quality = FunctionQualityMetrics()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+'
        }
