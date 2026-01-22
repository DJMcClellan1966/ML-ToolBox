"""
Tests for SICP Methods
Test functional programming, streams, data abstraction, symbolic computation
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sicp_methods import (
        FunctionalMLPipeline,
        Stream,
        DataAbstraction,
        SymbolicComputation,
        SICPMethods
    )
    SICP_AVAILABLE = True
except ImportError:
    SICP_AVAILABLE = False
    pytestmark = pytest.mark.skip("SICP methods not available")


class TestFunctionalMLPipeline:
    """Tests for functional ML pipeline"""
    
    def test_map_ml(self):
        """Test map"""
        result = FunctionalMLPipeline.map_ml(lambda x: x * 2, [1, 2, 3])
        assert result == [2, 4, 6]
    
    def test_filter_ml(self):
        """Test filter"""
        result = FunctionalMLPipeline.filter_ml(lambda x: x > 2, [1, 2, 3, 4])
        assert result == [3, 4]
    
    def test_reduce_ml(self):
        """Test reduce"""
        result = FunctionalMLPipeline.reduce_ml(lambda x, y: x + y, [1, 2, 3])
        assert result == 6
    
    def test_fold_left(self):
        """Test left fold"""
        result = FunctionalMLPipeline.fold_left(lambda acc, x: acc + x, 0, [1, 2, 3])
        assert result == 6
    
    def test_fold_right(self):
        """Test right fold"""
        result = FunctionalMLPipeline.fold_right(lambda x, acc: x + acc, 0, [1, 2, 3])
        assert result == 6
    
    def test_compose(self):
        """Test function composition"""
        add_one = lambda x: x + 1
        multiply_two = lambda x: x * 2
        composed = FunctionalMLPipeline.compose(add_one, multiply_two)
        assert composed(3) == 7  # (3 * 2) + 1
    
    def test_pipe(self):
        """Test pipe"""
        add_one = lambda x: x + 1
        multiply_two = lambda x: x * 2
        result = FunctionalMLPipeline.pipe(3, multiply_two, add_one)
        assert result == 7
    
    def test_curry(self):
        """Test currying"""
        def add(a, b):
            return a + b
        
        add_five = FunctionalMLPipeline.curry(add, 5)
        assert add_five(3) == 8
    
    def test_zip_with(self):
        """Test zip_with"""
        result = FunctionalMLPipeline.zip_with(lambda x, y: x + y, [1, 2, 3], [4, 5, 6])
        assert result == [5, 7, 9]
    
    def test_flat_map(self):
        """Test flat_map"""
        result = FunctionalMLPipeline.flat_map(lambda x: [x, x * 2], [1, 2])
        assert result == [1, 2, 2, 4]


class TestStream:
    """Tests for streams"""
    
    def test_from_list(self):
        """Test stream from list"""
        stream = Stream.from_list([1, 2, 3])
        assert stream.first == 1
        assert stream.rest.first == 2
    
    def test_take(self):
        """Test take"""
        stream = Stream.from_list([1, 2, 3, 4, 5])
        result = stream.take(3)
        assert result == [1, 2, 3]
    
    def test_to_list(self):
        """Test to_list"""
        stream = Stream.from_list([1, 2, 3])
        result = stream.to_list()
        assert result == [1, 2, 3]
    
    def test_map(self):
        """Test stream map"""
        stream = Stream.from_list([1, 2, 3])
        mapped = stream.map(lambda x: x * 2)
        result = mapped.take(3)
        assert result == [2, 4, 6]
    
    def test_filter(self):
        """Test stream filter"""
        stream = Stream.from_list([1, 2, 3, 4, 5])
        filtered = stream.filter(lambda x: x > 2)
        result = filtered.take(5)
        assert result == [3, 4, 5]
    
    def test_integers(self):
        """Test infinite integer stream"""
        stream = Stream.integers(0, 1)
        result = stream.take(5)
        assert result == [0, 1, 2, 3, 4]
    
    def test_range_stream(self):
        """Test range stream"""
        stream = Stream.range_stream(0, 5)
        result = stream.to_list()
        assert result == [0, 1, 2, 3, 4]
    
    def test_reduce(self):
        """Test stream reduce"""
        stream = Stream.from_list([1, 2, 3, 4])
        result = stream.reduce(lambda x, y: x + y, 0)
        assert result == 10
    
    def test_zip(self):
        """Test stream zip"""
        stream1 = Stream.from_list([1, 2, 3])
        stream2 = Stream.from_list([4, 5, 6])
        zipped = stream1.zip(stream2)
        result = zipped.take(3)
        assert result == [(1, 4), (2, 5), (3, 6)]


class TestDataAbstraction:
    """Tests for data abstraction"""
    
    def test_pair(self):
        """Test pair"""
        pair = DataAbstraction.Pair.cons(1, 2)
        assert pair.car() == 1
        assert pair.cdr() == 2
    
    def test_list_from_python(self):
        """Test list from Python list"""
        func_list = DataAbstraction.List.from_python_list([1, 2, 3])
        assert func_list.car() == 1
        assert func_list.cdr().car() == 2
    
    def test_list_to_python(self):
        """Test list to Python list"""
        func_list = DataAbstraction.List.from_python_list([1, 2, 3])
        python_list = DataAbstraction.List.to_python_list(func_list)
        assert python_list == [1, 2, 3]
    
    def test_tree(self):
        """Test tree"""
        tree = DataAbstraction.Tree.make_tree(
            1,
            DataAbstraction.Tree.make_tree(2),
            DataAbstraction.Tree.make_tree(3)
        )
        assert DataAbstraction.Tree.value(tree) == 1
        assert DataAbstraction.Tree.value(DataAbstraction.Tree.left(tree)) == 2


class TestSymbolicComputation:
    """Tests for symbolic computation"""
    
    def test_expression(self):
        """Test expression"""
        expr = SymbolicComputation.Expression.make_expression('+', 1, 2, 3)
        result = expr.evaluate()
        assert result == 6
    
    def test_expression_multiplication(self):
        """Test multiplication expression"""
        expr = SymbolicComputation.Expression.make_expression('*', 2, 3, 4)
        result = expr.evaluate()
        assert result == 24


class TestSICPMethods:
    """Test unified framework"""
    
    def test_unified_interface(self):
        """Test SICPMethods"""
        sicp = SICPMethods()
        
        assert sicp.functional is not None
        assert sicp.streams is not None
        assert sicp.data_abstraction is not None
        assert sicp.symbolic is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
