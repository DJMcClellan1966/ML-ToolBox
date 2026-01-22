"""
Algorithm Design Patterns - Skiena "Algorithm Design Manual"
Practical algorithm design patterns and problem-solution mapping

Methods from:
- Steven Skiena "The Algorithm Design Manual"
- Algorithm design patterns
- Problem-solution mapping
- Algorithm selection guide
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


class AlgorithmDesignPatterns:
    """
    Algorithm Design Patterns - Skiena
    
    Reusable algorithm templates and patterns
    """
    
    @staticmethod
    def greedy_template(items: List[Any], value_func: Callable, 
                       constraint_func: Callable) -> List[Any]:
        """
        Greedy Algorithm Template
        
        Args:
            items: List of items to select
            value_func: Function to compute value of item
            constraint_func: Function to check if item fits constraint
            
        Returns:
            Selected items
        """
        # Sort by value (descending)
        sorted_items = sorted(items, key=value_func, reverse=True)
        
        selected = []
        for item in sorted_items:
            if constraint_func(item, selected):
                selected.append(item)
        
        return selected
    
    @staticmethod
    def divide_and_conquer_template(data: List[Any], 
                                   base_case: Callable,
                                   divide: Callable,
                                   combine: Callable) -> Any:
        """
        Divide and Conquer Template
        
        Args:
            data: Input data
            base_case: Function to handle base case
            divide: Function to divide problem
            combine: Function to combine results
            
        Returns:
            Result
        """
        if base_case(data):
            return base_case(data)
        
        subproblems = divide(data)
        results = [AlgorithmDesignPatterns.divide_and_conquer_template(
            sub, base_case, divide, combine
        ) for sub in subproblems]
        
        return combine(results)
    
    @staticmethod
    def dynamic_programming_template(problem_size: int,
                                    base_cases: Dict[int, Any],
                                    recurrence: Callable) -> Any:
        """
        Dynamic Programming Template
        
        Args:
            problem_size: Size of problem
            base_cases: Base case solutions
            recurrence: Recurrence relation function
            
        Returns:
            Solution
        """
        memo = base_cases.copy()
        
        for i in range(max(base_cases.keys()) + 1, problem_size + 1):
            memo[i] = recurrence(i, memo)
        
        return memo[problem_size]
    
    @staticmethod
    def backtracking_template(candidates: List[Any],
                             is_valid: Callable,
                             is_complete: Callable,
                             make_move: Callable,
                             unmake_move: Callable) -> List[Any]:
        """
        Backtracking Template
        
        Args:
            candidates: List of candidate solutions
            is_valid: Check if candidate is valid
            is_complete: Check if solution is complete
            make_move: Make a move
            unmake_move: Undo a move
            
        Returns:
            Solutions
        """
        solutions = []
        
        def backtrack(current_solution):
            if is_complete(current_solution):
                solutions.append(current_solution.copy())
                return
            
            for candidate in candidates:
                if is_valid(candidate, current_solution):
                    make_move(candidate, current_solution)
                    backtrack(current_solution)
                    unmake_move(candidate, current_solution)
        
        backtrack([])
        return solutions


class ProblemSolutionMapper:
    """
    Problem-Solution Mapper - Skiena
    
    Map problems to appropriate algorithms
    """
    
    PROBLEM_ALGORITHM_MAP = {
        'shortest_path': ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall', 'A*'],
        'minimum_spanning_tree': ['Kruskal', "Prim's", 'Boruvka'],
        'sorting': ['Quicksort', 'Mergesort', 'Heapsort', 'Radix Sort'],
        'searching': ['Binary Search', 'Hash Table', 'Trie', 'Bloom Filter'],
        'optimization': ['Greedy', 'Dynamic Programming', 'Linear Programming'],
        'graph_traversal': ['DFS', 'BFS', 'Topological Sort'],
        'pattern_matching': ['KMP', 'Boyer-Moore', 'Rabin-Karp'],
        'subset_selection': ['Backtracking', 'Dynamic Programming', 'Greedy'],
        'scheduling': ['Greedy', 'Dynamic Programming', 'Linear Programming']
    }
    
    @staticmethod
    def suggest_algorithm(problem_type: str, constraints: Dict[str, Any]) -> List[str]:
        """
        Suggest algorithms for problem type
        
        Args:
            problem_type: Type of problem
            constraints: Problem constraints (time, space, etc.)
            
        Returns:
            List of suggested algorithms
        """
        suggestions = ProblemSolutionMapper.PROBLEM_ALGORITHM_MAP.get(
            problem_type, []
        )
        
        # Filter by constraints
        if constraints.get('time_complexity') == 'O(n log n)':
            suggestions = [s for s in suggestions if 'Sort' in s or 'Heap' in s]
        
        return suggestions
    
    @staticmethod
    def get_algorithm_complexity(algorithm: str) -> Dict[str, str]:
        """
        Get complexity of algorithm
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Complexity dictionary
        """
        complexities = {
            'Dijkstra': {'time': 'O((V + E) log V)', 'space': 'O(V)'},
            'Quicksort': {'time': 'O(n log n) average', 'space': 'O(log n)'},
            'Mergesort': {'time': 'O(n log n)', 'space': 'O(n)'},
            'Binary Search': {'time': 'O(log n)', 'space': 'O(1)'},
            'DFS': {'time': 'O(V + E)', 'space': 'O(V)'},
            'BFS': {'time': 'O(V + E)', 'space': 'O(V)'},
            'KMP': {'time': 'O(n + m)', 'space': 'O(m)'}
        }
        
        return complexities.get(algorithm, {'time': 'Unknown', 'space': 'Unknown'})


class BackOfEnvelopeCalculator:
    """
    Back-of-the-Envelope Calculator - Bentley "Programming Pearls"
    
    Quick performance estimates and calculations
    """
    
    @staticmethod
    def estimate_time_complexity(n: int, operation: str) -> float:
        """
        Estimate time for operation
        
        Args:
            n: Problem size
            operation: Type of operation ('sort', 'search', 'traverse')
            
        Returns:
            Estimated time (relative)
        """
        estimates = {
            'sort': n * np.log2(n) if n > 0 else 0,
            'search': np.log2(n) if n > 0 else 0,
            'traverse': n,
            'quadratic': n ** 2,
            'cubic': n ** 3
        }
        
        return estimates.get(operation, n)
    
    @staticmethod
    def estimate_memory(n: int, data_type: str = 'int') -> float:
        """
        Estimate memory usage
        
        Args:
            n: Number of elements
            data_type: Type of data ('int', 'float', 'string')
            
        Returns:
            Estimated memory in bytes
        """
        type_sizes = {
            'int': 4,
            'float': 8,
            'string': 50  # Average string size
        }
        
        return n * type_sizes.get(data_type, 4)
    
    @staticmethod
    def estimate_throughput(operations_per_second: float, 
                           operation_time: float) -> float:
        """
        Estimate throughput
        
        Args:
            operations_per_second: Operations per second
            operation_time: Time per operation (seconds)
            
        Returns:
            Throughput (operations/second)
        """
        return operations_per_second / operation_time if operation_time > 0 else 0
    
    @staticmethod
    def quick_big_o_estimate(algorithm: str, n: int) -> str:
        """
        Quick Big O estimate
        
        Args:
            algorithm: Algorithm name
            n: Problem size
            
        Returns:
            Big O notation
        """
        big_o_map = {
            'quicksort': 'O(n log n)',
            'mergesort': 'O(n log n)',
            'heapsort': 'O(n log n)',
            'binary_search': 'O(log n)',
            'linear_search': 'O(n)',
            'dfs': 'O(V + E)',
            'bfs': 'O(V + E)',
            'dijkstra': 'O((V + E) log V)'
        }
        
        return big_o_map.get(algorithm.lower(), 'O(unknown)')


class AlgorithmDesignFramework:
    """
    Unified Algorithm Design Framework
    """
    
    def __init__(self):
        self.patterns = AlgorithmDesignPatterns()
        self.mapper = ProblemSolutionMapper()
        self.calculator = BackOfEnvelopeCalculator()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+',
            'numpy': 'numpy>=1.26.0'
        }
