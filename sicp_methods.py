"""
SICP (Structure and Interpretation of Computer Programs) Methods
Functional programming, streams, and data abstraction for ML

Methods from:
- SICP: Higher-order functions, streams, data abstraction, symbolic computation
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable, Iterator, Generator, Union
from functools import reduce, partial
import itertools
import operator

sys.path.insert(0, str(Path(__file__).parent))


class FunctionalMLPipeline:
    """
    Functional ML Pipeline (SICP)
    
    Higher-order functions and function composition for ML
    """
    
    @staticmethod
    def map_ml(func: Callable, data: List[Any]) -> List[Any]:
        """
        Map function over ML data
        
        Args:
            func: Function to apply
            data: Data to process
            
        Returns:
            Mapped data
        """
        return list(map(func, data))
    
    @staticmethod
    def filter_ml(predicate: Callable, data: List[Any]) -> List[Any]:
        """
        Filter ML data by predicate
        
        Args:
            predicate: Filter function
            data: Data to filter
            
        Returns:
            Filtered data
        """
        return list(filter(predicate, data))
    
    @staticmethod
    def reduce_ml(func: Callable, data: List[Any], initial: Any = None) -> Any:
        """
        Reduce ML data with function
        
        Args:
            func: Reduction function
            data: Data to reduce
            initial: Initial value
            
        Returns:
            Reduced value
        """
        if initial is None:
            return reduce(func, data)
        return reduce(func, data, initial)
    
    @staticmethod
    def fold_left(func: Callable, initial: Any, data: List[Any]) -> Any:
        """
        Left fold (accumulate from left)
        
        Args:
            func: Accumulation function
            initial: Initial accumulator value
            data: Data to fold
            
        Returns:
            Folded value
        """
        result = initial
        for item in data:
            result = func(result, item)
        return result
    
    @staticmethod
    def fold_right(func: Callable, initial: Any, data: List[Any]) -> Any:
        """
        Right fold (accumulate from right)
        
        Args:
            func: Accumulation function
            initial: Initial accumulator value
            data: Data to fold
            
        Returns:
            Folded value
        """
        result = initial
        for item in reversed(data):
            result = func(item, result)
        return result
    
    @staticmethod
    def compose(*funcs: Callable) -> Callable:
        """
        Compose functions (right to left)
        
        Args:
            *funcs: Functions to compose
            
        Returns:
            Composed function
        """
        def composed(x):
            result = x
            for func in reversed(funcs):
                result = func(result)
            return result
        return composed
    
    @staticmethod
    def pipe(data: Any, *funcs: Callable) -> Any:
        """
        Pipe data through functions (left to right)
        
        Args:
            data: Input data
            *funcs: Functions to apply
            
        Returns:
            Final result
        """
        result = data
        for func in funcs:
            result = func(result)
        return result
    
    @staticmethod
    def curry(func: Callable, *args, **kwargs) -> Callable:
        """
        Curry function (partial application)
        
        Args:
            func: Function to curry
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Curried function
        """
        return partial(func, *args, **kwargs)
    
    @staticmethod
    def apply(func: Callable, args: Tuple = (), kwargs: Dict = None) -> Any:
        """
        Apply function with arguments
        
        Args:
            func: Function to apply
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if kwargs:
            return func(*args, **kwargs)
        return func(*args)
    
    @staticmethod
    def zip_with(func: Callable, *iterables) -> List[Any]:
        """
        Zip iterables and apply function
        
        Args:
            func: Function to apply
            *iterables: Iterables to zip
            
        Returns:
            List of results
        """
        return [func(*items) for items in zip(*iterables)]
    
    @staticmethod
    def flat_map(func: Callable, data: List[Any]) -> List[Any]:
        """
        Map and flatten
        
        Args:
            func: Function to apply (should return iterable)
            data: Data to process
            
        Returns:
            Flattened mapped results
        """
        result = []
        for item in data:
            mapped = func(item)
            if isinstance(mapped, (list, tuple)):
                result.extend(mapped)
            else:
                result.append(mapped)
        return result


class Stream:
    """
    Stream (SICP)
    
    Lazy evaluation and infinite sequences
    """
    
    def __init__(self, first: Any, compute_rest: Optional[Callable] = None):
        """
        Args:
            first: First element
            compute_rest: Function to compute rest of stream
        """
        self.first = first
        self._rest = None
        self._compute_rest = compute_rest
        self._computed = False
    
    @property
    def rest(self):
        """Get rest of stream (lazy)"""
        if not self._computed:
            if self._compute_rest:
                self._rest = self._compute_rest()
            else:
                self._rest = Stream.empty()
            self._computed = True
        return self._rest
    
    @staticmethod
    def empty():
        """Empty stream"""
        return None
    
    @staticmethod
    def from_list(lst: List[Any]) -> 'Stream':
        """Create stream from list"""
        if not lst:
            return Stream.empty()
        return Stream(lst[0], lambda: Stream.from_list(lst[1:]))
    
    @staticmethod
    def from_generator(gen: Generator) -> 'Stream':
        """Create stream from generator"""
        try:
            first = next(gen)
            return Stream(first, lambda: Stream.from_generator(gen))
        except StopIteration:
            return Stream.empty()
    
    @staticmethod
    def integers(start: int = 0, step: int = 1) -> 'Stream':
        """Infinite stream of integers"""
        return Stream(start, lambda: Stream.integers(start + step, step))
    
    @staticmethod
    def range_stream(start: int, end: int, step: int = 1) -> 'Stream':
        """Stream of integers in range"""
        if start >= end:
            return Stream.empty()
        return Stream(start, lambda: Stream.range_stream(start + step, end, step))
    
    def take(self, n: int) -> List[Any]:
        """Take first n elements"""
        if n <= 0 or self is Stream.empty():
            return []
        result = [self.first]
        rest = self.rest
        if rest is not Stream.empty():
            result.extend(rest.take(n - 1))
        return result
    
    def to_list(self, max_items: Optional[int] = None) -> List[Any]:
        """Convert stream to list"""
        if self is Stream.empty():
            return []
        result = [self.first]
        rest = self.rest
        count = 1
        while rest is not Stream.empty() and (max_items is None or count < max_items):
            result.append(rest.first)
            rest = rest.rest
            count += 1
        return result
    
    def map(self, func: Callable) -> 'Stream':
        """Map function over stream"""
        if self is Stream.empty() or self is None:
            return Stream.empty()
        rest = self.rest
        if rest is Stream.empty() or rest is None:
            return Stream(func(self.first), lambda: Stream.empty())
        return Stream(func(self.first), lambda: rest.map(func))
    
    def filter(self, predicate: Callable) -> 'Stream':
        """Filter stream by predicate"""
        if self is Stream.empty() or self is None:
            return Stream.empty()
        rest = self.rest
        if predicate(self.first):
            if rest is Stream.empty() or rest is None:
                return Stream(self.first, lambda: Stream.empty())
            return Stream(self.first, lambda: rest.filter(predicate))
        if rest is Stream.empty() or rest is None:
            return Stream.empty()
        return rest.filter(predicate)
    
    def reduce(self, func: Callable, initial: Any = None) -> Any:
        """Reduce stream"""
        if self is Stream.empty() or self is None:
            return initial
        rest = self.rest
        if initial is None:
            if rest is Stream.empty() or rest is None:
                return self.first
            return rest.reduce(func, self.first)
        if rest is Stream.empty() or rest is None:
            return func(initial, self.first)
        return rest.reduce(func, func(initial, self.first))
    
    def zip(self, other: 'Stream') -> 'Stream':
        """Zip two streams"""
        if self is Stream.empty() or self is None or other is Stream.empty() or other is None:
            return Stream.empty()
        rest_self = self.rest
        rest_other = other.rest
        if rest_self is Stream.empty() or rest_self is None or rest_other is Stream.empty() or rest_other is None:
            return Stream((self.first, other.first), lambda: Stream.empty())
        return Stream(
            (self.first, other.first),
            lambda: rest_self.zip(rest_other)
        )


class DataAbstraction:
    """
    Data Abstraction (SICP)
    
    Abstract data types and type constructors
    """
    
    class Pair:
        """Pair (cons/car/cdr)"""
        
        def __init__(self, first: Any, second: Any):
            self.first = first
            self.second = second
        
        @staticmethod
        def cons(first: Any, second: Any) -> 'DataAbstraction.Pair':
            """Construct pair"""
            return DataAbstraction.Pair(first, second)
        
        def car(self) -> Any:
            """Get first element"""
            return self.first
        
        def cdr(self) -> Any:
            """Get second element"""
            return self.second
    
    class List:
        """Functional list (built from pairs)"""
        
        @staticmethod
        def from_python_list(lst: List[Any]) -> 'DataAbstraction.Pair':
            """Convert Python list to functional list"""
            if not lst:
                return None
            return DataAbstraction.Pair.cons(
                lst[0],
                DataAbstraction.List.from_python_list(lst[1:])
            )
        
        @staticmethod
        def to_python_list(pair: Optional['DataAbstraction.Pair']) -> List[Any]:
            """Convert functional list to Python list"""
            if pair is None:
                return []
            return [pair.car()] + DataAbstraction.List.to_python_list(pair.cdr())
    
    class Tree:
        """Binary tree abstraction"""
        
        @staticmethod
        def make_tree(value: Any, left: Optional['DataAbstraction.Tree'] = None, 
                     right: Optional['DataAbstraction.Tree'] = None) -> 'DataAbstraction.Tree':
            """Make tree node"""
            tree = DataAbstraction.Tree()
            tree.value = value
            tree.left = left
            tree.right = right
            return tree
        
        @staticmethod
        def value(tree: 'DataAbstraction.Tree') -> Any:
            """Get tree value"""
            return tree.value if tree else None
        
        @staticmethod
        def left(tree: 'DataAbstraction.Tree') -> Optional['DataAbstraction.Tree']:
            """Get left subtree"""
            return tree.left if tree else None
        
        @staticmethod
        def right(tree: 'DataAbstraction.Tree') -> Optional['DataAbstraction.Tree']:
            """Get right subtree"""
            return tree.right if tree else None


class SymbolicComputation:
    """
    Symbolic Computation (SICP)
    
    Symbol manipulation and expression evaluation
    """
    
    class Expression:
        """Symbolic expression"""
        
        def __init__(self, operator: str, operands: List[Any]):
            self.operator = operator
            self.operands = operands
        
        @staticmethod
        def make_expression(operator: str, *operands) -> 'SymbolicComputation.Expression':
            """Make expression"""
            return SymbolicComputation.Expression(operator, list(operands))
        
        def evaluate(self, env: Dict[str, Any] = None) -> Any:
            """Evaluate expression"""
            if env is None:
                env = {}
            
            if self.operator == '+':
                return sum(op.evaluate(env) if isinstance(op, SymbolicComputation.Expression) else op 
                          for op in self.operands)
            elif self.operator == '*':
                result = 1
                for op in self.operands:
                    val = op.evaluate(env) if isinstance(op, SymbolicComputation.Expression) else op
                    result *= val
                return result
            elif self.operator == 'variable':
                return env.get(self.operands[0], 0)
            else:
                return self.operands[0] if self.operands else None
        
        def __str__(self):
            if self.operator in ['+', '*']:
                op_str = ' ' + self.operator + ' '
                return f"({op_str.join(str(op) for op in self.operands)})"
            return str(self.operands[0] if self.operands else '')


class SICPMethods:
    """
    Unified SICP Methods Framework
    """
    
    def __init__(self):
        self.functional = FunctionalMLPipeline()
        self.streams = Stream
        self.data_abstraction = DataAbstraction()
        self.symbolic = SymbolicComputation()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+'
        }
