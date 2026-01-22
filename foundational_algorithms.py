"""
Foundational Algorithms - Other CS References
Implements algorithms from Sedgewick, Skiena, Aho/Hopcroft/Ullman, and Bentley

Algorithms from:
- Sedgewick: Production data structures (Red-Black, B-Tree, Skip List, Network Flow)
- Skiena: Practical algorithms (Backtracking framework, Approximation, Randomized)
- Aho/Hopcroft/Ullman: Classic data structures (AVL Tree, Advanced heaps)
- Bentley: Practical programming tricks (Bit manipulation, Kadane's, Two Sum)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict, deque
import heapq
import random
import math

sys.path.insert(0, str(Path(__file__).parent))


class SedgewickDataStructures:
    """
    Production Data Structures (Sedgewick)
    
    Red-Black trees, B-Trees, Skip lists, Network flow
    """
    
    class RedBlackTree:
        """Red-Black Tree - Self-balancing BST"""
        
        class Node:
            def __init__(self, key: Any, value: Any = None, color: bool = True):
                self.key = key
                self.value = value
                self.left = None
                self.right = None
                self.color = color  # True = red, False = black
        
        def __init__(self):
            self.root = None
        
        def _is_red(self, node: Optional[Node]) -> bool:
            """Check if node is red"""
            if node is None:
                return False
            return node.color
        
        def _rotate_left(self, node: Node) -> Node:
            """Left rotation"""
            x = node.right
            node.right = x.left
            x.left = node
            x.color = node.color
            node.color = True
            return x
        
        def _rotate_right(self, node: Node) -> Node:
            """Right rotation"""
            x = node.left
            node.left = x.right
            x.right = node
            x.color = node.color
            node.color = True
            return x
        
        def _flip_colors(self, node: Node):
            """Flip colors"""
            node.color = True
            node.left.color = False
            node.right.color = False
        
        def insert(self, key: Any, value: Any = None):
            """Insert key-value pair"""
            self.root = self._insert(self.root, key, value)
            self.root.color = False
        
        def _insert(self, node: Optional[Node], key: Any, value: Any) -> Node:
            """Recursive insert"""
            if node is None:
                return self.Node(key, value, True)
            
            if key < node.key:
                node.left = self._insert(node.left, key, value)
            elif key > node.key:
                node.right = self._insert(node.right, key, value)
            else:
                node.value = value
            
            # Balance
            if self._is_red(node.right) and not self._is_red(node.left):
                node = self._rotate_left(node)
            if self._is_red(node.left) and self._is_red(node.left.left):
                node = self._rotate_right(node)
            if self._is_red(node.left) and self._is_red(node.right):
                self._flip_colors(node)
            
            return node
        
        def search(self, key: Any) -> Optional[Any]:
            """Search for key"""
            node = self.root
            while node is not None:
                if key < node.key:
                    node = node.left
                elif key > node.key:
                    node = node.right
                else:
                    return node.value
            return None
    
    class SkipList:
        """Skip List - Probabilistic data structure"""
        
        class Node:
            def __init__(self, key: Any, value: Any = None, level: int = 0):
                self.key = key
                self.value = value
                self.forward = [None] * (level + 1)
        
        def __init__(self, max_level: int = 16, probability: float = 0.5):
            self.max_level = max_level
            self.probability = probability
            self.level = 0
            self.header = self.Node(None, None, max_level)
        
        def _random_level(self) -> int:
            """Generate random level"""
            level = 0
            while random.random() < self.probability and level < self.max_level:
                level += 1
            return level
        
        def insert(self, key: Any, value: Any = None):
            """Insert key-value pair"""
            update = [None] * (self.max_level + 1)
            current = self.header
            
            for i in range(self.level, -1, -1):
                while current.forward[i] and current.forward[i].key < key:
                    current = current.forward[i]
                update[i] = current
            
            current = current.forward[0]
            
            if current and current.key == key:
                current.value = value
            else:
                new_level = self._random_level()
                if new_level > self.level:
                    for i in range(self.level + 1, new_level + 1):
                        update[i] = self.header
                    self.level = new_level
                
                new_node = self.Node(key, value, new_level)
                for i in range(new_level + 1):
                    new_node.forward[i] = update[i].forward[i]
                    update[i].forward[i] = new_node
        
        def search(self, key: Any) -> Optional[Any]:
            """Search for key"""
            current = self.header
            
            for i in range(self.level, -1, -1):
                while current.forward[i] and current.forward[i].key < key:
                    current = current.forward[i]
            
            current = current.forward[0]
            if current and current.key == key:
                return current.value
            return None
    
    @staticmethod
    def max_flow_min_cut(graph: Dict[int, List[Tuple[int, float]]], source: int, sink: int) -> float:
        """
        Max Flow / Min Cut (Ford-Fulkerson with BFS)
        
        Args:
            graph: Adjacency list with capacities
            source: Source node
            sink: Sink node
            
        Returns:
            Maximum flow
        """
        # Build residual graph
        residual = defaultdict(dict)
        for u in graph:
            for v, cap in graph[u]:
                residual[u][v] = cap
                if v not in residual:
                    residual[v] = {}
        
        max_flow = 0
        
        while True:
            # BFS to find augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = -1
            
            found = False
            while queue:
                u = queue.popleft()
                for v, cap in residual.get(u, {}).items():
                    if v not in parent and cap > 0:
                        parent[v] = u
                        queue.append(v)
                        if v == sink:
                            found = True
                            break
                if found:
                    break
            
            if not found:
                break
            
            # Find minimum capacity along path
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u
            
            # Update residual graph
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                if u not in residual[v]:
                    residual[v][u] = 0
                residual[v][u] += path_flow
                v = u
            
            max_flow += path_flow
        
        return max_flow


class SkienaAlgorithms:
    """
    Practical Algorithms (Skiena - Algorithm Design Manual)
    
    Backtracking framework, approximation algorithms, randomized algorithms
    """
    
    @staticmethod
    def backtracking_framework(
        candidates: Callable,
        is_valid: Callable,
        is_complete: Callable,
        make_move: Callable,
        unmake_move: Callable
    ) -> List[Any]:
        """
        General Backtracking Framework
        
        Args:
            candidates: Function(state) -> list of candidate moves
            is_valid: Function(state, move) -> bool
            is_complete: Function(state) -> bool
            make_move: Function(state, move) -> new state
            unmake_move: Function(state, move) -> previous state
            
        Returns:
            List of solutions
        """
        solutions = []
        
        def backtrack(state):
            if is_complete(state):
                solutions.append(state.copy() if hasattr(state, 'copy') else state)
                return
            
            for move in candidates(state):
                if is_valid(state, move):
                    new_state = make_move(state, move)
                    backtrack(new_state)
                    state = unmake_move(new_state, move)
        
        backtrack({})
        return solutions
    
    @staticmethod
    def greedy_approximation(
        items: List[Any],
        value_func: Callable,
        constraint_func: Callable,
        selection_func: Callable
    ) -> List[Any]:
        """
        Greedy Approximation Algorithm
        
        Args:
            items: List of items
            value_func: Function(item) -> value
            constraint_func: Function(selected, item) -> bool (can add?)
            selection_func: Function(items) -> best item (greedy choice)
            
        Returns:
            Selected items
        """
        selected = []
        remaining = items.copy()
        
        while remaining:
            best = selection_func(remaining)
            if constraint_func(selected, best):
                selected.append(best)
            remaining.remove(best)
        
        return selected
    
    @staticmethod
    def monte_carlo_algorithm(
        problem_func: Callable,
        n_trials: int = 1000
    ) -> Dict[str, Any]:
        """
        Monte Carlo Randomized Algorithm
        
        Args:
            problem_func: Function() -> result
            n_trials: Number of trials
            
        Returns:
            Dictionary with results
        """
        results = []
        for _ in range(n_trials):
            results.append(problem_func())
        
        return {
            'results': results,
            'mean': np.mean(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)
        }


class AhoHopcroftUllman:
    """
    Classic Data Structures (Aho, Hopcroft, Ullman)
    
    AVL trees, advanced heaps
    """
    
    class AVLTree:
        """AVL Tree - Self-balancing BST"""
        
        class Node:
            def __init__(self, key: Any, value: Any = None):
                self.key = key
                self.value = value
                self.left = None
                self.right = None
                self.height = 1
        
        def __init__(self):
            self.root = None
        
        def _height(self, node: Optional[Node]) -> int:
            """Get height"""
            return node.height if node else 0
        
        def _balance_factor(self, node: Node) -> int:
            """Get balance factor"""
            return self._height(node.left) - self._height(node.right)
        
        def _rotate_right(self, y: Node) -> Node:
            """Right rotation"""
            x = y.left
            T2 = x.right
            
            x.right = y
            y.left = T2
            
            y.height = 1 + max(self._height(y.left), self._height(y.right))
            x.height = 1 + max(self._height(x.left), self._height(x.right))
            
            return x
        
        def _rotate_left(self, x: Node) -> Node:
            """Left rotation"""
            y = x.right
            T2 = y.left
            
            y.left = x
            x.right = T2
            
            x.height = 1 + max(self._height(x.left), self._height(x.right))
            y.height = 1 + max(self._height(y.left), self._height(y.right))
            
            return y
        
        def insert(self, key: Any, value: Any = None):
            """Insert key-value pair"""
            self.root = self._insert(self.root, key, value)
        
        def _insert(self, node: Optional[Node], key: Any, value: Any) -> Node:
            """Recursive insert"""
            if node is None:
                return self.Node(key, value)
            
            if key < node.key:
                node.left = self._insert(node.left, key, value)
            elif key > node.key:
                node.right = self._insert(node.right, key, value)
            else:
                node.value = value
                return node
            
            node.height = 1 + max(self._height(node.left), self._height(node.right))
            balance = self._balance_factor(node)
            
            # Left Left
            if balance > 1 and key < node.left.key:
                return self._rotate_right(node)
            
            # Right Right
            if balance < -1 and key > node.right.key:
                return self._rotate_left(node)
            
            # Left Right
            if balance > 1 and key > node.left.key:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            
            # Right Left
            if balance < -1 and key < node.right.key:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
            
            return node
        
        def search(self, key: Any) -> Optional[Any]:
            """Search for key"""
            node = self.root
            while node:
                if key < node.key:
                    node = node.left
                elif key > node.key:
                    node = node.right
                else:
                    return node.value
            return None


class BentleyAlgorithms:
    """
    Practical Programming Tricks (Bentley - Programming Pearls)
    
    Bit manipulation, common problems, optimizations
    """
    
    @staticmethod
    def maximum_subarray(arr: List[float]) -> Tuple[float, int, int]:
        """
        Maximum Subarray (Kadane's Algorithm)
        
        Find contiguous subarray with maximum sum
        
        Args:
            arr: Array of numbers
            
        Returns:
            Tuple (max_sum, start_index, end_index)
        """
        if not arr:
            return 0, 0, 0
        
        max_sum = arr[0]
        current_sum = arr[0]
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(1, len(arr)):
            if current_sum < 0:
                current_sum = arr[i]
                temp_start = i
            else:
                current_sum += arr[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return max_sum, start, end
    
    @staticmethod
    def two_sum(arr: List[int], target: int) -> Optional[Tuple[int, int]]:
        """
        Two Sum Problem (Hash-based)
        
        Find two numbers that sum to target
        
        Args:
            arr: Array of integers
            target: Target sum
            
        Returns:
            Tuple of indices or None
        """
        seen = {}
        for i, num in enumerate(arr):
            complement = target - num
            if complement in seen:
                return (seen[complement], i)
            seen[num] = i
        return None
    
    @staticmethod
    def rotate_array(arr: List[Any], k: int) -> List[Any]:
        """
        Array Rotation (In-place)
        
        Rotate array left by k positions
        
        Args:
            arr: Array to rotate
            k: Rotation amount
            
        Returns:
            Rotated array
        """
        n = len(arr)
        k = k % n
        return arr[k:] + arr[:k]
    
    @staticmethod
    def search_rotated_array(arr: List[int], target: int) -> Optional[int]:
        """
        Search in Rotated Sorted Array
        
        Binary search in rotated array
        
        Args:
            arr: Rotated sorted array
            target: Target value
            
        Returns:
            Index or None
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            
            if arr[left] <= arr[mid]:
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return None
    
    @staticmethod
    def bit_manipulation_tricks() -> Dict[str, Callable]:
        """
        Bit Manipulation Tricks
        
        Returns:
            Dictionary of bit manipulation functions
        """
        def is_power_of_two(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0
        
        def count_set_bits(n: int) -> int:
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count
        
        def get_lowest_set_bit(n: int) -> int:
            return n & -n
        
        def clear_lowest_set_bit(n: int) -> int:
            return n & (n - 1)
        
        return {
            'is_power_of_two': is_power_of_two,
            'count_set_bits': count_set_bits,
            'get_lowest_set_bit': get_lowest_set_bit,
            'clear_lowest_set_bit': clear_lowest_set_bit
        }


class FoundationalAlgorithms:
    """
    Unified interface for all foundational algorithms
    """
    
    def __init__(self):
        self.sedgewick = SedgewickDataStructures()
        self.skiena = SkienaAlgorithms()
        self.aho_hopcroft_ullman = AhoHopcroftUllman()
        self.bentley = BentleyAlgorithms()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'numpy': 'numpy>=1.26.0',
            'python': 'Python 3.8+'
        }
