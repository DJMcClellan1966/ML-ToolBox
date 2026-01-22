"""
Advanced Algorithms - Additional Knuth-Like Improvements
Combines TAOCP Vol. 2 (Numerical), CLRS (DP/Greedy), and data structures

Algorithms from:
- TAOCP Vol. 2: Numerical methods (Horner's, GCD, modular arithmetic)
- CLRS: Dynamic programming, greedy algorithms
- TAOCP Vol. 1: Advanced data structures (Heap, BST, Hash table, Union-Find)
- Sedgewick: Advanced graph and string algorithms
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict
import heapq

sys.path.insert(0, str(Path(__file__).parent))


class NumericalMethods:
    """
    Numerical Methods (TAOCP Vol. 2)
    
    Polynomial evaluation, GCD, modular arithmetic, multiple-precision
    """
    
    @staticmethod
    def horner_method(coeffs: List[float], x: float) -> float:
        """
        Horner's Method - Algorithm A (Vol. 2)
        
        Efficient polynomial evaluation: a_n*x^n + ... + a_0
        
        Args:
            coeffs: Coefficients [a_n, a_{n-1}, ..., a_0]
            x: Value to evaluate at
            
        Returns:
            Polynomial value
        """
        result = 0.0
        for coeff in coeffs:
            result = result * x + coeff
        return result
    
    @staticmethod
    def euclidean_gcd(a: int, b: int) -> int:
        """
        Euclidean GCD Algorithm - Algorithm E (Vol. 2)
        
        Greatest Common Divisor using Euclidean algorithm
        
        Args:
            a, b: Integers
            
        Returns:
            GCD of a and b
        """
        while b != 0:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def extended_euclidean(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm - Algorithm X (Vol. 2)
        
        Returns (gcd, x, y) such that gcd = a*x + b*y
        
        Args:
            a, b: Integers
            
        Returns:
            Tuple (gcd, x, y)
        """
        if a == 0:
            return abs(b), 0, 1 if b > 0 else -1
        
        gcd, x1, y1 = NumericalMethods.extended_euclidean(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    @staticmethod
    def modular_inverse(a: int, m: int) -> Optional[int]:
        """
        Modular Inverse using Extended Euclidean
        
        Find x such that (a * x) mod m = 1
        
        Args:
            a: Integer
            m: Modulus
            
        Returns:
            Modular inverse or None if doesn't exist
        """
        gcd, x, _ = NumericalMethods.extended_euclidean(a, m)
        if gcd != 1:
            return None  # Inverse doesn't exist
        return (x % m + m) % m
    
    @staticmethod
    def modular_exponentiation(base: int, exp: int, mod: int) -> int:
        """
        Fast Modular Exponentiation
        
        Compute (base^exp) mod mod efficiently
        
        Args:
            base: Base
            exp: Exponent
            mod: Modulus
            
        Returns:
            (base^exp) mod mod
        """
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        
        return result
    
    @staticmethod
    def multiple_precision_add(a: List[int], b: List[int], base: int = 10) -> List[int]:
        """
        Multiple-Precision Addition (Vol. 2)
        
        Add two large numbers represented as digit lists
        
        Args:
            a, b: Numbers as digit lists (least significant first)
            base: Number base (default 10)
            
        Returns:
            Sum as digit list
        """
        result = []
        carry = 0
        max_len = max(len(a), len(b))
        
        for i in range(max_len):
            digit_a = a[i] if i < len(a) else 0
            digit_b = b[i] if i < len(b) else 0
            
            total = digit_a + digit_b + carry
            result.append(total % base)
            carry = total // base
        
        if carry > 0:
            result.append(carry)
        
        return result


class DynamicProgramming:
    """
    Dynamic Programming Algorithms (CLRS)
    
    Optimal solutions for optimization problems
    """
    
    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> Tuple[int, str]:
        """
        Longest Common Subsequence (LCS)
        
        Find longest subsequence common to both strings
        
        Args:
            s1, s2: Strings
            
        Returns:
            Tuple (length, LCS string)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Reconstruct LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                lcs.append(s1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return dp[m][n], ''.join(reversed(lcs))
    
    @staticmethod
    def knapsack_01(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
        """
        0/1 Knapsack Problem
        
        Maximize value without exceeding capacity
        
        Args:
            weights: Item weights
            values: Item values
            capacity: Maximum weight
            
        Returns:
            Tuple (max_value, selected_items)
        """
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        # Build DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        dp[i-1][w],
                        dp[i-1][w - weights[i-1]] + values[i-1]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Reconstruct solution
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
        
        return dp[n][capacity], selected
    
    @staticmethod
    def matrix_chain_multiplication(dims: List[int]) -> Tuple[int, List[List[int]]]:
        """
        Matrix Chain Multiplication
        
        Find optimal parenthesization to minimize multiplications
        
        Args:
            dims: Dimensions [d0, d1, d2, ..., dn] for matrices A0...An-1
                  where Ai has dimensions dims[i] x dims[i+1]
            
        Returns:
            Tuple (min_cost, split_table)
        """
        n = len(dims) - 1
        dp = [[0] * n for _ in range(n)]
        splits = [[0] * n for _ in range(n)]
        
        # Build DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        splits[i][j] = k
        
        return dp[0][n-1], splits


class GreedyAlgorithms:
    """
    Greedy Algorithms (CLRS)
    
    Fast approximate solutions
    """
    
    @staticmethod
    def huffman_coding(frequencies: Dict[str, int]) -> Dict[str, str]:
        """
        Huffman Coding
        
        Optimal prefix-free encoding
        
        Args:
            frequencies: Character frequencies
            
        Returns:
            Dictionary mapping characters to binary codes
        """
        if len(frequencies) == 0:
            return {}
        if len(frequencies) == 1:
            return {list(frequencies.keys())[0]: '0'}
        
        # Build Huffman tree
        heap = [(freq, char) for char, freq in frequencies.items()]
        heapq.heapify(heap)
        
        tree = {}
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)
            
            merged = (freq1 + freq2, (node1, node2))
            heapq.heappush(heap, merged)
        
        # Build codes from tree
        codes = {}
        
        def build_codes(node, code=''):
            if isinstance(node, str):
                codes[node] = code if code else '0'
            else:
                build_codes(node[0], code + '0')
                build_codes(node[1], code + '1')
        
        if heap:
            build_codes(heap[0][1])
        
        return codes
    
    @staticmethod
    def kruskal_mst(edges: List[Tuple[int, int, float]], n: int) -> List[Tuple[int, int, float]]:
        """
        Kruskal's Minimum Spanning Tree
        
        Args:
            edges: List of (u, v, weight) edges
            n: Number of vertices
            
        Returns:
            List of edges in MST
        """
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        # Union-Find for cycle detection
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        mst = []
        for u, v, weight in edges:
            if find(u) != find(v):
                mst.append((u, v, weight))
                union(u, v)
                if len(mst) == n - 1:
                    break
        
        return mst
    
    @staticmethod
    def fractional_knapsack(weights: List[float], values: List[float], capacity: float) -> Tuple[float, List[float]]:
        """
        Fractional Knapsack (Greedy)
        
        Can take fractions of items
        
        Args:
            weights: Item weights
            values: Item values
            capacity: Maximum weight
            
        Returns:
            Tuple (max_value, fractions_taken)
        """
        # Sort by value/weight ratio
        items = [(values[i] / weights[i], weights[i], values[i], i) for i in range(len(weights))]
        items.sort(reverse=True)
        
        total_value = 0.0
        fractions = [0.0] * len(weights)
        remaining = capacity
        
        for ratio, weight, value, idx in items:
            if remaining >= weight:
                fractions[idx] = 1.0
                total_value += value
                remaining -= weight
            else:
                fractions[idx] = remaining / weight
                total_value += value * fractions[idx]
                break
        
        return total_value, fractions


class AdvancedDataStructures:
    """
    Advanced Data Structures (TAOCP Vol. 1, CLRS)
    
    Heap, BST, Hash table, Union-Find, Trie
    """
    
    class MinHeap:
        """Min Heap implementation"""
        
        def __init__(self):
            self.heap = []
        
        def push(self, item: Any, priority: float):
            """Insert item with priority"""
            heapq.heappush(self.heap, (priority, item))
        
        def pop(self) -> Tuple[float, Any]:
            """Extract minimum"""
            return heapq.heappop(self.heap)
        
        def peek(self) -> Tuple[float, Any]:
            """Get minimum without removing"""
            return self.heap[0] if self.heap else None
        
        def is_empty(self) -> bool:
            return len(self.heap) == 0
        
        def size(self) -> int:
            return len(self.heap)
    
    class BinarySearchTree:
        """Binary Search Tree"""
        
        class Node:
            def __init__(self, key: Any, value: Any = None):
                self.key = key
                self.value = value
                self.left = None
                self.right = None
        
        def __init__(self):
            self.root = None
        
        def insert(self, key: Any, value: Any = None):
            """Insert key-value pair"""
            self.root = self._insert(self.root, key, value)
        
        def _insert(self, node: Optional[Node], key: Any, value: Any) -> Node:
            if node is None:
                return self.Node(key, value)
            
            if key < node.key:
                node.left = self._insert(node.left, key, value)
            elif key > node.key:
                node.right = self._insert(node.right, key, value)
            else:
                node.value = value
            
            return node
        
        def search(self, key: Any) -> Optional[Any]:
            """Search for key"""
            node = self._search(self.root, key)
            return node.value if node else None
        
        def _search(self, node: Optional[Node], key: Any) -> Optional[Node]:
            if node is None or node.key == key:
                return node
            if key < node.key:
                return self._search(node.left, key)
            return self._search(node.right, key)
    
    class UnionFind:
        """Disjoint-Set (Union-Find) data structure"""
        
        def __init__(self, n: int):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x: int) -> int:
            """Find root with path compression"""
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x: int, y: int):
            """Union two sets"""
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return
            
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
        
        def connected(self, x: int, y: int) -> bool:
            """Check if x and y are in same set"""
            return self.find(x) == self.find(y)
    
    class HashTable:
        """Hash table with chaining"""
        
        def __init__(self, size: int = 101):
            self.size = size
            self.table = [[] for _ in range(size)]
        
        def _hash(self, key: Any) -> int:
            """Hash function"""
            return hash(key) % self.size
        
        def insert(self, key: Any, value: Any):
            """Insert key-value pair"""
            idx = self._hash(key)
            for pair in self.table[idx]:
                if pair[0] == key:
                    pair[1] = value
                    return
            self.table[idx].append([key, value])
        
        def get(self, key: Any) -> Optional[Any]:
            """Get value by key"""
            idx = self._hash(key)
            for pair in self.table[idx]:
                if pair[0] == key:
                    return pair[1]
            return None
        
        def delete(self, key: Any) -> bool:
            """Delete key"""
            idx = self._hash(key)
            for i, pair in enumerate(self.table[idx]):
                if pair[0] == key:
                    del self.table[idx][i]
                    return True
            return False
    
    class Trie:
        """Trie (Prefix Tree)"""
        
        class Node:
            def __init__(self):
                self.children = {}
                self.is_end = False
                self.value = None
        
        def __init__(self):
            self.root = self.Node()
        
        def insert(self, word: str, value: Any = None):
            """Insert word"""
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = self.Node()
                node = node.children[char]
            node.is_end = True
            node.value = value
        
        def search(self, word: str) -> Optional[Any]:
            """Search for word"""
            node = self.root
            for char in word:
                if char not in node.children:
                    return None
                node = node.children[char]
            return node.value if node.is_end else None
        
        def starts_with(self, prefix: str) -> List[str]:
            """Find all words with prefix"""
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]
            
            words = []
            self._collect_words(node, prefix, words)
            return words
        
        def _collect_words(self, node: Node, prefix: str, words: List[str]):
            """Collect all words from node"""
            if node.is_end:
                words.append(prefix)
            for char, child in node.children.items():
                self._collect_words(child, prefix + char, words)


class AdvancedGraphAlgorithms:
    """
    Advanced Graph Algorithms (CLRS, Sedgewick)
    """
    
    @staticmethod
    def strongly_connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
        """
        Strongly Connected Components (Tarjan's algorithm)
        
        Args:
            graph: Adjacency list
            
        Returns:
            List of SCCs
        """
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        sccs = []
        
        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            
            for w in graph.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])
            
            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
        
        for v in graph:
            if v not in indices:
                strongconnect(v)
        
        return sccs
    
    @staticmethod
    def floyd_warshall(graph: Dict[int, List[Tuple[int, float]]], n: int) -> np.ndarray:
        """
        Floyd-Warshall All-Pairs Shortest Path
        
        Args:
            graph: Weighted graph (adjacency list)
            n: Number of vertices
            
        Returns:
            Distance matrix
        """
        dist = np.full((n, n), np.inf)
        
        # Initialize
        for i in range(n):
            dist[i][i] = 0
            for neighbor, weight in graph.get(i, []):
                dist[i][neighbor] = weight
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist


class AdvancedAlgorithms:
    """
    Unified interface for all advanced algorithms
    """
    
    def __init__(self):
        self.numerical = NumericalMethods()
        self.dp = DynamicProgramming()
        self.greedy = GreedyAlgorithms()
        self.data_structures = AdvancedDataStructures()
        self.graph = AdvancedGraphAlgorithms()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'numpy': 'numpy>=1.26.0',
            'python': 'Python 3.8+'
        }
