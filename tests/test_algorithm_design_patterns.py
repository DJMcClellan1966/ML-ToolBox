"""
Tests for Algorithm Design Patterns
Test algorithm patterns, problem-solution mapping, back-of-envelope calculations
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from algorithm_design_patterns import (
        AlgorithmDesignPatterns,
        ProblemSolutionMapper,
        BackOfEnvelopeCalculator,
        AlgorithmDesignFramework
    )
    ALGORITHM_DESIGN_AVAILABLE = True
except ImportError:
    ALGORITHM_DESIGN_AVAILABLE = False
    pytestmark = pytest.mark.skip("Algorithm design patterns not available")


class TestAlgorithmDesignPatterns:
    """Tests for algorithm design patterns"""
    
    def test_greedy_template(self):
        """Test greedy template"""
        items = [('a', 10, 5), ('b', 20, 10), ('c', 15, 7)]
        
        def value_func(item):
            return item[1]  # Value
        
        def constraint_func(item, selected):
            total_weight = sum(i[2] for i in selected)
            return total_weight + item[2] <= 15
        
        result = AlgorithmDesignPatterns.greedy_template(
            items, value_func, constraint_func
        )
        
        assert len(result) > 0
    
    def test_divide_and_conquer_template(self):
        """Test divide and conquer template"""
        data = [1, 2, 3, 4, 5]
        
        def base_case(d):
            if len(d) <= 1:
                return d[0] if d else 0
        
        def divide(d):
            mid = len(d) // 2
            return [d[:mid], d[mid:]]
        
        def combine(results):
            return sum(results)
        
        result = AlgorithmDesignPatterns.divide_and_conquer_template(
            data, base_case, divide, combine
        )
        
        assert result == sum(data)
    
    def test_dynamic_programming_template(self):
        """Test dynamic programming template"""
        base_cases = {0: 0, 1: 1}
        
        def recurrence(n, memo):
            return memo[n-1] + memo[n-2]
        
        result = AlgorithmDesignPatterns.dynamic_programming_template(
            5, base_cases, recurrence
        )
        
        assert result == 5  # Fibonacci(5)


class TestProblemSolutionMapper:
    """Tests for problem-solution mapper"""
    
    def test_suggest_algorithm(self):
        """Test algorithm suggestion"""
        suggestions = ProblemSolutionMapper.suggest_algorithm(
            'shortest_path', {}
        )
        
        assert len(suggestions) > 0
        assert 'Dijkstra' in suggestions
    
    def test_get_algorithm_complexity(self):
        """Test get algorithm complexity"""
        complexity = ProblemSolutionMapper.get_algorithm_complexity('Dijkstra')
        
        assert 'time' in complexity
        assert 'space' in complexity


class TestBackOfEnvelopeCalculator:
    """Tests for back-of-envelope calculator"""
    
    def test_estimate_time_complexity(self):
        """Test time complexity estimation"""
        estimate = BackOfEnvelopeCalculator.estimate_time_complexity(1000, 'sort')
        assert estimate > 0
    
    def test_estimate_memory(self):
        """Test memory estimation"""
        memory = BackOfEnvelopeCalculator.estimate_memory(1000, 'int')
        assert memory == 4000  # 1000 * 4 bytes
    
    def test_quick_big_o_estimate(self):
        """Test Big O estimate"""
        big_o = BackOfEnvelopeCalculator.quick_big_o_estimate('quicksort', 1000)
        assert 'O' in big_o


class TestAlgorithmDesignFramework:
    """Test unified framework"""
    
    def test_unified_interface(self):
        """Test AlgorithmDesignFramework"""
        framework = AlgorithmDesignFramework()
        
        assert framework.patterns is not None
        assert framework.mapper is not None
        assert framework.calculator is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
