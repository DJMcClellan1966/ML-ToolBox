"""
Sedgewick & Wayne Algorithms
Practical, production-ready algorithms for ML

Methods from:
- Robert Sedgewick & Kevin Wayne "Algorithms" (4th edition)
- Priority queues, symbol tables, advanced graphs, sorting, string structures
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable, Set
import heapq
from collections import defaultdict, deque
import math

sys.path.insert(0, str(Path(__file__).parent))


class IndexedPriorityQueue:
    """
    Indexed Priority Queue - Sedgewick & Wayne
    
    Priority queue with index access for efficient updates
    """
    
    def __init__(self, max_size: int):
        """
        Args:
            max_size: Maximum number of elements
        """
        self.max_size = max_size
        self.pq = [None]  # Binary heap (1-indexed, so index 0 is unused)
        self.keys = [None] * (max_size + 1)  # keys[i] = priority of element i
        self.qp = [-1] * (max_size + 1)  # qp[i] = position of element i in pq
        self.n = 0  # Number of elements
    
    def insert(self, i: int, key: float):
        """
        Insert element i with priority key
        
        Args:
            i: Element index
            key: Priority
        """
        if i < 0 or i >= self.max_size:
            raise ValueError(f"Index {i} out of range")
        if self.contains(i):
            raise ValueError(f"Index {i} already exists")
        
        self.n += 1
        self.qp[i] = self.n
        if len(self.pq) <= self.n:
            self.pq.append(i)
        else:
            self.pq[self.n] = i
        self.keys[i] = key
        self._swim(self.n)
    
    def contains(self, i: int) -> bool:
        """Check if element i exists"""
        if i < 0 or i >= self.max_size:
            return False
        return self.qp[i] != -1
    
    def change_key(self, i: int, key: float):
        """Change priority of element i"""
        if not self.contains(i):
            raise ValueError(f"Index {i} does not exist")
        
        old_key = self.keys[i]
        self.keys[i] = key
        
        if key < old_key:
            self._swim(self.qp[i])
        else:
            self._sink(self.qp[i])
    
    def delete_min(self) -> int:
        """Delete and return element with minimum priority"""
        if self.n == 0:
            raise ValueError("Priority queue is empty")
        
        min_idx = self.pq[1]
        self._exch(1, self.n)
        self.n -= 1
        if self.n > 0:
            self._sink(1)
        self.qp[min_idx] = -1
        self.keys[min_idx] = None
        return min_idx
    
    def min_key(self) -> float:
        """Get minimum priority"""
        if self.n == 0:
            raise ValueError("Priority queue is empty")
        return self.keys[self.pq[1]]
    
    def min_index(self) -> int:
        """Get index of minimum element"""
        if self.n == 0:
            raise ValueError("Priority queue is empty")
        return self.pq[1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self.n == 0
    
    def size(self) -> int:
        """Get size"""
        return self.n
    
    def _swim(self, k: int):
        """Swim up the heap"""
        while k > 1 and self._greater(k // 2, k):
            self._exch(k, k // 2)
            k = k // 2
    
    def _sink(self, k: int):
        """Sink down the heap"""
        while 2 * k <= self.n:
            j = 2 * k
            if j < self.n and self._greater(j, j + 1):
                j += 1
            if not self._greater(k, j):
                break
            self._exch(k, j)
            k = j
    
    def _greater(self, i: int, j: int) -> bool:
        """Compare priorities"""
        return self.keys[self.pq[i]] > self.keys[self.pq[j]]
    
    def _exch(self, i: int, j: int):
        """Exchange elements"""
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.qp[self.pq[i]] = i
        self.qp[self.pq[j]] = j


class OrderedSymbolTable:
    """
    Ordered Symbol Table - Sedgewick & Wayne
    
    Symbol table with ordered operations
    """
    
    def __init__(self):
        """Initialize ordered symbol table"""
        self.keys = []
        self.values = {}
    
    def put(self, key: Any, value: Any):
        """Put key-value pair"""
        if key not in self.values:
            # Insert in sorted order
            import bisect
            pos = bisect.bisect_left(self.keys, key)
            self.keys.insert(pos, key)
        self.values[key] = value
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value for key"""
        return self.values.get(key)
    
    def delete(self, key: Any):
        """Delete key"""
        if key in self.values:
            self.keys.remove(key)
            del self.values[key]
    
    def contains(self, key: Any) -> bool:
        """Check if key exists"""
        return key in self.values
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self.keys) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self.keys)
    
    def min_key(self) -> Optional[Any]:
        """Get minimum key"""
        return self.keys[0] if self.keys else None
    
    def max_key(self) -> Optional[Any]:
        """Get maximum key"""
        return self.keys[-1] if self.keys else None
    
    def floor(self, key: Any) -> Optional[Any]:
        """Get largest key <= given key"""
        import bisect
        pos = bisect.bisect_right(self.keys, key) - 1
        return self.keys[pos] if pos >= 0 else None
    
    def ceiling(self, key: Any) -> Optional[Any]:
        """Get smallest key >= given key"""
        import bisect
        pos = bisect.bisect_left(self.keys, key)
        return self.keys[pos] if pos < len(self.keys) else None
    
    def rank(self, key: Any) -> int:
        """Get number of keys < given key"""
        import bisect
        return bisect.bisect_left(self.keys, key)
    
    def select(self, k: int) -> Optional[Any]:
        """Get key of rank k"""
        return self.keys[k] if 0 <= k < len(self.keys) else None
    
    def range_keys(self, lo: Any, hi: Any) -> List[Any]:
        """Get keys in range [lo, hi]"""
        import bisect
        left = bisect.bisect_left(self.keys, lo)
        right = bisect.bisect_right(self.keys, hi)
        return self.keys[left:right]


class AStarSearch:
    """
    A* Search Algorithm - Sedgewick & Wayne
    
    Heuristic search for shortest path
    """
    
    @staticmethod
    def search(
        graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        goal: Any,
        heuristic: Callable[[Any], float]
    ) -> Optional[Tuple[List[Any], float]]:
        """
        A* search for shortest path
        
        Args:
            graph: Adjacency list (node -> [(neighbor, cost), ...])
            start: Start node
            goal: Goal node
            heuristic: Heuristic function h(node) -> estimated cost to goal
            
        Returns:
            Tuple (path, cost) or None if no path
        """
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}
        g_score = {start: 0}  # Actual cost from start
        f_score = {start: heuristic(start)}  # Estimated total cost
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[goal]
            
            for neighbor, cost in graph.get(current, []):
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found


class BidirectionalSearch:
    """
    Bidirectional Search - Sedgewick & Wayne
    
    Search from both start and goal simultaneously
    """
    
    @staticmethod
    def search(
        graph: Dict[Any, List[Any]],
        start: Any,
        goal: Any
    ) -> Optional[List[Any]]:
        """
        Bidirectional search
        
        Args:
            graph: Adjacency list
            start: Start node
            goal: Goal node
            
        Returns:
            Path or None
        """
        if start == goal:
            return [start]
        
        # Forward search from start
        forward_queue = deque([start])
        forward_visited = {start: None}
        
        # Backward search from goal
        backward_queue = deque([goal])
        backward_visited = {goal: None}
        
        while forward_queue or backward_queue:
            # Forward step
            if forward_queue:
                current = forward_queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        if neighbor in backward_visited:
                            # Found meeting point
                            return BidirectionalSearch._reconstruct_path(
                                forward_visited, backward_visited, neighbor, start, goal
                            )
                        forward_queue.append(neighbor)
            
            # Backward step
            if backward_queue:
                current = backward_queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        if neighbor in forward_visited:
                            # Found meeting point
                            return BidirectionalSearch._reconstruct_path(
                                forward_visited, backward_visited, neighbor, start, goal
                            )
                        backward_queue.append(neighbor)
        
        return None
    
    @staticmethod
    def _reconstruct_path(
        forward_visited: Dict[Any, Any],
        backward_visited: Dict[Any, Any],
        meeting: Any,
        start: Any,
        goal: Any
    ) -> List[Any]:
        """Reconstruct path from meeting point"""
        # Path from start to meeting
        path = []
        node = meeting
        while node is not None:
            path.append(node)
            node = forward_visited.get(node)
        path.reverse()
        
        # Path from meeting to goal
        node = backward_visited.get(meeting)
        while node is not None:
            path.append(node)
            node = backward_visited.get(node)
        
        return path


class ThreeWayQuicksort:
    """
    3-Way Quicksort - Sedgewick & Wayne
    
    Efficient sorting with duplicate keys
    """
    
    @staticmethod
    def sort(arr: List[Any], key: Optional[Callable] = None) -> List[Any]:
        """
        3-way quicksort
        
        Args:
            arr: List to sort
            key: Optional key function
            
        Returns:
            Sorted list
        """
        if key is None:
            key = lambda x: x
        
        arr_copy = arr.copy()
        ThreeWayQuicksort._sort(arr_copy, 0, len(arr_copy) - 1, key)
        return arr_copy
    
    @staticmethod
    def _sort(arr: List[Any], lo: int, hi: int, key: Callable):
        """Recursive sort"""
        if hi <= lo:
            return
        
        lt, i, gt = lo, lo + 1, hi
        pivot_key = key(arr[lo])
        
        while i <= gt:
            cmp = (key(arr[i]) > pivot_key) - (key(arr[i]) < pivot_key)
            if cmp < 0:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif cmp > 0:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                i += 1
        
        ThreeWayQuicksort._sort(arr, lo, lt - 1, key)
        ThreeWayQuicksort._sort(arr, gt + 1, hi, key)


class Trie:
    """
    Trie (Prefix Tree) - Sedgewick & Wayne
    
    Efficient string prefix matching
    """
    
    class Node:
        """Trie node"""
        def __init__(self):
            self.children = {}
            self.value = None
            self.is_end = False
    
    def __init__(self):
        """Initialize trie"""
        self.root = Trie.Node()
    
    def insert(self, key: str, value: Any = None):
        """Insert key"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = Trie.Node()
            node = node.children[char]
        node.is_end = True
        node.value = value
    
    def search(self, key: str) -> Optional[Any]:
        """Search for key"""
        node = self._find_node(key)
        return node.value if node and node.is_end else None
    
    def starts_with(self, prefix: str) -> bool:
        """Check if prefix exists"""
        node = self._find_node(prefix)
        return node is not None
    
    def _find_node(self, key: str) -> Optional['Trie.Node']:
        """Find node for key"""
        node = self.root
        for char in key:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def keys_with_prefix(self, prefix: str) -> List[str]:
        """Get all keys with prefix"""
        node = self._find_node(prefix)
        if node is None:
            return []
        
        results = []
        self._collect_keys(node, prefix, results)
        return results
    
    def _collect_keys(self, node: 'Trie.Node', prefix: str, results: List[str]):
        """Collect keys from node"""
        if node.is_end:
            results.append(prefix)
        for char, child in node.children.items():
            self._collect_keys(child, prefix + char, results)


class BloomFilter:
    """
    Bloom Filter - Sedgewick & Wayne
    
    Probabilistic membership testing
    """
    
    def __init__(self, n: int, k: int = 3):
        """
        Args:
            n: Expected number of elements
            k: Number of hash functions
        """
        self.n = n
        self.k = k
        self.m = int(-n * math.log(0.01) / (math.log(2) ** 2))  # Optimal size
        self.bit_array = [False] * self.m
    
    def add(self, item: str):
        """Add item"""
        for i in range(self.k):
            index = self._hash(item, i) % self.m
            self.bit_array[index] = True
    
    def contains(self, item: str) -> bool:
        """Check if item might be in set"""
        for i in range(self.k):
            index = self._hash(item, i) % self.m
            if not self.bit_array[index]:
                return False
        return True
    
    def _hash(self, item: str, seed: int) -> int:
        """Hash function"""
        hash_val = hash(item + str(seed))
        return abs(hash_val)


class SedgewickWayneAlgorithms:
    """
    Unified Sedgewick & Wayne Algorithms Framework
    """
    
    def __init__(self):
        self.indexed_priority_queue = IndexedPriorityQueue
        self.ordered_symbol_table = OrderedSymbolTable
        self.astar = AStarSearch
        self.bidirectional_search = BidirectionalSearch
        self.three_way_quicksort = ThreeWayQuicksort
        self.trie = Trie
        self.bloom_filter = BloomFilter
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+',
            'heapq': 'Python heapq (built-in)',
            'bisect': 'Python bisect (built-in)'
        }
