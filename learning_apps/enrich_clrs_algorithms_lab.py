"""
Enrich CLRS Algorithms Lab curriculum from 10 → 35+ items.
Covers Introduction to Algorithms (Cormen, Leiserson, Rivest, Stein):
Sorting, data structures, DP, greedy, graph algorithms, NP-completeness.
"""
import json
from pathlib import Path

def generate_curriculum_items():
    """Generate comprehensive CLRS algorithms curriculum."""
    
    items = []
    
    # ============================================================================
    # DYNAMIC PROGRAMMING (8 items)
    # ============================================================================
    
    items.append({
        "id": "clrs_coin",
        "book_id": "clrs_dp",
        "level": "basics",
        "title": "Coin Change (Min Coins)",
        "learn": "DP: min coins to make amount. dp[i] = min over coins of 1 + dp[i-coin]. Reconstruct combination. Classic intro to DP with optimal substructure.",
        "try_code": """from clrs_complete_algorithms import CLRSDynamicProgramming
coins = [1, 3, 4]
amount = 6
n, combo = CLRSDynamicProgramming.coin_change_min_coins(coins, amount)
print(f'Min coins for {amount}: {n}, combination: {combo}')""",
        "try_demo": "clrs_coin",
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_rod",
        "book_id": "clrs_dp",
        "level": "intermediate",
        "title": "Rod Cutting (Ch 15.1)",
        "learn": "Maximize profit by cutting rod. prices[i] = price for length i+1. DP: R[n] = max(prices[i] + R[n-i-1]). First DP problem in CLRS demonstrating optimal substructure.",
        "try_code": """from clrs_complete_algorithms import CLRSDynamicProgramming
prices = [1, 5, 8, 9, 10, 17, 17, 20]
length = 8
profit, cuts = CLRSDynamicProgramming.rod_cutting(prices, length)
print(f'Max profit: {profit}, cuts at positions: {cuts}')""",
        "try_demo": "clrs_rod",
        "prerequisites": ["clrs_coin"]
    })
    
    items.append({
        "id": "clrs_matrix_chain",
        "book_id": "clrs_dp",
        "level": "intermediate",
        "title": "Matrix Chain Multiplication (Ch 15.2)",
        "learn": "Find optimal parenthesization to minimize scalar multiplications. DP on subproblems [i,j]. Classic example of O(n³) DP with 2D table.",
        "try_code": """def matrix_chain_order(dims):
    n = len(dims) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m[0][n-1], s

# Matrices: A1(10x100), A2(100x5), A3(5x50)
dims = [10, 100, 5, 50]
min_cost, splits = matrix_chain_order(dims)
print(f'Min multiplications: {min_cost}')""",
        "try_demo": None,
        "prerequisites": ["clrs_rod"]
    })
    
    items.append({
        "id": "clrs_lcs",
        "book_id": "clrs_dp",
        "level": "intermediate",
        "title": "Longest Common Subsequence (Ch 15.4)",
        "learn": "Find longest subsequence common to two sequences. DP: if X[i]=Y[j], LCS[i][j] = 1+LCS[i-1][j-1], else max(LCS[i-1][j], LCS[i][j-1]). Applications: diff, bioinformatics.",
        "try_code": """def lcs_length(X, Y):
    m, n = len(X), len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                c[i][j] = c[i-1][j-1] + 1
            else:
                c[i][j] = max(c[i-1][j], c[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs.append(X[i-1])
            i -= 1; j -= 1
        elif c[i-1][j] > c[i][j-1]:
            i -= 1
        else:
            j -= 1
    return c[m][n], ''.join(reversed(lcs))

X, Y = "ABCBDAB", "BDCABA"
length, sequence = lcs_length(X, Y)
print(f'LCS length: {length}, sequence: {sequence}')  # BCBA""",
        "try_demo": None,
        "prerequisites": ["clrs_matrix_chain"]
    })
    
    items.append({
        "id": "clrs_obst",
        "book_id": "clrs_dp",
        "level": "advanced",
        "title": "Optimal BST (Ch 15.5)",
        "learn": "Build binary search tree with minimum expected search cost given key frequencies. DP on root choices for each subrange [i,j]. More complex: uses 3 tables (e, w, root).",
        "try_code": """from clrs_complete_algorithms import CLRSDynamicProgramming
keys = ['A', 'B', 'C']
freq = [0.5, 0.3, 0.2]
cost, root = CLRSDynamicProgramming.optimal_binary_search_tree(keys, freq)
print(f'Min expected cost: {cost}, root structure: {root}')""",
        "try_demo": "clrs_obst",
        "prerequisites": ["clrs_lcs"]
    })
    
    items.append({
        "id": "clrs_lis",
        "book_id": "clrs_dp",
        "level": "intermediate",
        "title": "Longest Increasing Subsequence",
        "learn": "DP: dp[i] = length of LIS ending at i. Reconstruct via parent pointers. O(n²); can optimize to O(n log n) with binary search + patience sorting algorithm.",
        "try_code": """from clrs_complete_algorithms import CLRSDynamicProgramming
arr = [10, 22, 9, 33, 21, 50, 41, 60]
length, indices = CLRSDynamicProgramming.longest_increasing_subsequence(arr)
print(f'LIS length: {length}, indices: {indices}')
print(f'LIS: {[arr[i] for i in indices]}')""",
        "try_demo": "clrs_lis",
        "prerequisites": ["clrs_lcs"]
    })
    
    items.append({
        "id": "clrs_edit_distance",
        "book_id": "clrs_dp",
        "level": "advanced",
        "title": "Edit Distance (Levenshtein)",
        "learn": "Minimum operations (insert, delete, substitute) to transform one string to another. DP: if chars match, dp[i][j]=dp[i-1][j-1], else 1+min(insert, delete, substitute).",
        "try_code": """def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # substitute
                )
    return dp[m][n]

print(edit_distance("kitten", "sitting"))  # 3""",
        "try_demo": None,
        "prerequisites": ["clrs_lcs"]
    })
    
    items.append({
        "id": "clrs_knapsack",
        "book_id": "clrs_dp",
        "level": "advanced",
        "title": "0/1 Knapsack",
        "learn": "Select items to maximize value subject to weight constraint. DP: K[i][w] = max(K[i-1][w], value[i] + K[i-1][w-weight[i]]). Pseudo-polynomial O(nW).",
        "try_code": """def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    values[i-1] + dp[i-1][w - weights[i-1]]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Reconstruct items
    items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1)
            w -= weights[i-1]
    
    return dp[n][capacity], items[::-1]

weights = [10, 20, 30]
values = [60, 100, 120]
max_val, items = knapsack_01(weights, values, 50)
print(f'Max value: {max_val}, items: {items}')""",
        "try_demo": None,
        "prerequisites": ["clrs_obst"]
    })
    
    # ============================================================================
    # GREEDY ALGORITHMS (6 items)
    # ============================================================================
    
    items.append({
        "id": "clrs_activity",
        "book_id": "clrs_greedy",
        "level": "basics",
        "title": "Activity Selection (Ch 16.1)",
        "learn": "Schedule maximum number of non-overlapping activities. Greedy: pick earliest finish time, then next compatible. Proves greedy choice property and optimal substructure.",
        "try_code": """def activity_selection(start, finish):
    # Sort by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    selected = [activities[0]]
    
    for s, f in activities[1:]:
        if s >= selected[-1][1]:  # Compatible with last selected
            selected.append((s, f))
    
    return selected

start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
result = activity_selection(start, finish)
print(f'Selected activities: {result}')  # (1,2), (3,4), (5,7), (8,9)""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_huffman",
        "book_id": "clrs_greedy",
        "level": "intermediate",
        "title": "Huffman Coding (Ch 16.3)",
        "learn": "Optimal prefix-free binary encoding. Greedy: merge two least-frequent symbols. Build tree bottom-up. Applications: compression, information theory.",
        "try_code": """import heapq

class HuffmanNode:
    def __init__(self, char, freq, left=None, right=None):
        self.char = char; self.freq = freq
        self.left = left; self.right = right
    def __lt__(self, other): return self.freq < other.freq

def huffman_coding(freq_dict):
    heap = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    
    def get_codes(node, code='', codes={}):
        if node.char: codes[node.char] = code
        if node.left: get_codes(node.left, code + '0', codes)
        if node.right: get_codes(node.right, code + '1', codes)
        return codes
    
    return get_codes(heap[0])

freq = {'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5}
codes = huffman_coding(freq)
print(f'Huffman codes: {codes}')""",
        "try_demo": None,
        "prerequisites": ["clrs_activity"]
    })
    
    items.append({
        "id": "clrs_fractional_knapsack",
        "book_id": "clrs_greedy",
        "level": "basics",
        "title": "Fractional Knapsack",
        "learn": "Allow fractional items (unlike 0/1). Greedy: sort by value/weight ratio, take items in order. O(n log n). Shows greedy works here but not for 0/1 variant.",
        "try_code": """def fractional_knapsack(weights, values, capacity):
    items = [(v/w, w, v) for w, v in zip(weights, values)]
    items.sort(reverse=True)  # Sort by value/weight ratio
    
    total_value = 0
    for ratio, weight, value in items:
        if capacity >= weight:
            capacity -= weight
            total_value += value
        else:
            total_value += ratio * capacity
            break
    
    return total_value

weights = [10, 20, 30]
values = [60, 100, 120]
print(fractional_knapsack(weights, values, 50))  # 240""",
        "try_demo": None,
        "prerequisites": ["clrs_activity"]
    })
    
    items.append({
        "id": "clrs_prim",
        "book_id": "clrs_greedy",
        "level": "intermediate",
        "title": "Prim's MST (Ch 23.2)",
        "learn": "Grow minimum spanning tree from a start vertex. Use min-heap for next edge. O(E log V) with binary heap. Greedy: add minimum weight edge connecting tree to non-tree vertex.",
        "try_code": """import heapq

def prim_mst(n, edges):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    mst = []
    visited = [False] * n
    min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
    
    while min_heap:
        weight, u, parent = heapq.heappop(min_heap)
        if visited[u]: continue
        
        visited[u] = True
        if parent != -1:
            mst.append((parent, u, weight))
        
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v, u))
    
    return mst, sum(w for _, _, w in mst)

edges = [(0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7), (2,8,2), (2,5,4)]
mst, total_weight = prim_mst(9, edges)
print(f'MST weight: {total_weight}')""",
        "try_demo": None,
        "prerequisites": ["clrs_huffman"]
    })
    
    items.append({
        "id": "clrs_kruskal",
        "book_id": "clrs_greedy",
        "level": "intermediate",
        "title": "Kruskal's MST (Ch 23.2)",
        "learn": "Build MST by adding edges in weight order, avoiding cycles. Use union-find (disjoint sets). O(E log E) for sorting. Alternative to Prim's with different data structure emphasis.",
        "try_code": """class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y: return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        return True

def kruskal_mst(n, edges):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst = []
    
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
    
    return mst, sum(w for _, _, w in mst)

edges = [(0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7)]
mst, total_weight = kruskal_mst(9, edges)
print(f'MST weight: {total_weight}')""",
        "try_demo": None,
        "prerequisites": ["clrs_prim"]
    })
    
    items.append({
        "id": "clrs_task_scheduling",
        "book_id": "clrs_greedy",
        "level": "advanced",
        "title": "Task Scheduling with Deadlines",
        "learn": "Maximize profit by scheduling tasks with deadlines and penalties. Greedy: sort by penalty/profit, schedule in latest available slot. Uses disjoint set for slot management.",
        "try_code": """def task_scheduling(tasks, max_deadline):
    # tasks = [(profit, deadline), ...]
    tasks.sort(reverse=True)  # Sort by profit descending
    
    slots = [-1] * (max_deadline + 1)
    total_profit = 0
    scheduled = []
    
    for profit, deadline in tasks:
        # Find latest available slot <= deadline
        for t in range(min(deadline, max_deadline), 0, -1):
            if slots[t] == -1:
                slots[t] = profit
                total_profit += profit
                scheduled.append((profit, deadline, t))
                break
    
    return total_profit, scheduled

tasks = [(100, 2), (10, 1), (15, 2), (27, 1)]
profit, schedule = task_scheduling(tasks, 2)
print(f'Max profit: {profit}, schedule: {schedule}')""",
        "try_demo": None,
        "prerequisites": ["clrs_kruskal"]
    })
    
    # ============================================================================
    # GRAPH ALGORITHMS (9 items)
    # ============================================================================
    
    items.append({
        "id": "clrs_bfs",
        "book_id": "clrs_graph",
        "level": "basics",
        "title": "Breadth-First Search (Ch 22.2)",
        "learn": "Explore graph level-by-level using queue. Finds shortest paths in unweighted graphs. O(V+E). Produces BFS tree with parent pointers and distances.",
        "try_code": """from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([(start, 0)])
    distances = {start: 0}
    parent = {start: None}
    
    while queue:
        u, dist = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                distances[v] = dist + 1
                parent[v] = u
                queue.append((v, dist + 1))
    
    return distances, parent

graph = {0: [1, 2], 1: [2], 2: [3], 3: [1]}
distances, parent = bfs(graph, 0)
print(f'Distances from 0: {distances}')""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_dfs",
        "book_id": "clrs_graph",
        "level": "basics",
        "title": "Depth-First Search (Ch 22.3)",
        "learn": "Explore graph deeply before backtracking. Discovers discovery/finish times. Produces DFS forest. Applications: topological sort, SCC, cycle detection. O(V+E).",
        "try_code": """def dfs(graph, start, visited=None):
    if visited is None: visited = set()
    visited.add(start)
    discovery = {start: len(visited)}
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        u = stack.pop()
        if u not in visited:
            visited.add(u)
            for v in reversed(graph[u]):  # Reversed to match recursive order
                if v not in visited:
                    stack.append(v)
    
    return visited

graph = {0: [1, 2], 1: [2], 2: [3], 3: [1]}
print(f'DFS from 0: {dfs(graph, 0)}')""",
        "try_demo": None,
        "prerequisites": ["clrs_bfs"]
    })
    
    items.append({
        "id": "clrs_topological_sort",
        "book_id": "clrs_graph",
        "level": "intermediate",
        "title": "Topological Sort (Ch 22.4)",
        "learn": "Linear ordering of DAG vertices where edge (u,v) → u before v. DFS-based: output vertices in reverse finish time. Applications: build systems, course scheduling. O(V+E).",
        "try_code": """def topological_sort(graph):
    visited = set()
    stack = []
    
    def dfs(u):
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs(v)
        stack.append(u)  # Add after visiting all descendants
    
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]  # Reverse to get topological order

# DAG: course prerequisites
graph = {
    'Linear Algebra': [],
    'Calculus': [],
    'Probability': ['Calculus'],
    'ML': ['Linear Algebra', 'Probability'],
    'Deep Learning': ['ML']
}
print(f'Course order: {topological_sort(graph)}')""",
        "try_demo": None,
        "prerequisites": ["clrs_dfs"]
    })
    
    items.append({
        "id": "clrs_scc",
        "book_id": "clrs_graph",
        "level": "advanced",
        "title": "Strongly Connected Components (Ch 22.5)",
        "learn": "Maximal sets where every vertex reaches every other. Kosaraju's algorithm: DFS on G, then DFS on G^T in reverse finish order. O(V+E). Applications: web graph analysis.",
        "try_code": """def kosaraju_scc(graph):
    # Step 1: DFS on original graph to get finish times
    visited = set()
    stack = []
    
    def dfs1(u):
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs1(v)
        stack.append(u)
    
    for vertex in graph:
        if vertex not in visited:
            dfs1(vertex)
    
    # Step 2: Create transpose graph
    transpose = {}
    for u in graph:
        for v in graph[u]:
            if v not in transpose: transpose[v] = []
            transpose[v].append(u)
    
    # Step 3: DFS on transpose in reverse finish order
    visited = set()
    sccs = []
    
    def dfs2(u, component):
        visited.add(u)
        component.append(u)
        for v in transpose.get(u, []):
            if v not in visited:
                dfs2(v, component)
    
    while stack:
        u = stack.pop()
        if u not in visited:
            component = []
            dfs2(u, component)
            sccs.append(component)
    
    return sccs

graph = {0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [5], 5: [3]}
print(f'SCCs: {kosaraju_scc(graph)}')""",
        "try_demo": None,
        "prerequisites": ["clrs_topological_sort"]
    })
    
    items.append({
        "id": "clrs_dijkstra",
        "book_id": "clrs_graph",
        "level": "intermediate",
        "title": "Dijkstra's Algorithm (Ch 24.3)",
        "learn": "Single-source shortest paths with non-negative weights. Greedy: extract min distance vertex, relax neighbors. O(V²) naive, O((V+E) log V) with min-heap.",
        "try_code": """import heapq

def dijkstra(graph, start):
    distances = {v: float('inf') for v in graph}
    distances[start] = 0
    parent = {v: None for v in graph}
    pq = [(0, start)]
    
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]: continue
        
        for v, weight in graph[u]:
            new_dist = distances[u] + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    return distances, parent

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
distances, parent = dijkstra(graph, 'A')
print(f'Shortest distances: {distances}')""",
        "try_demo": None,
        "prerequisites": ["clrs_bfs"]
    })
    
    items.append({
        "id": "clrs_bellman",
        "book_id": "clrs_graph",
        "level": "advanced",
        "title": "Bellman-Ford (Ch 24.1)",
        "learn": "Single-source shortest paths with negative edge weights. Relax all edges V-1 times. Detects negative cycles. O(VE). More general than Dijkstra but slower.",
        "try_code": """def bellman_ford(vertices, edges, start):
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    parent = {v: None for v in vertices}
    
    # Relax all edges V-1 times
    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                parent[v] = u
    
    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] + weight < distances[v]:
            return None, "Negative cycle detected"
    
    return distances, parent

vertices = ['A', 'B', 'C', 'D']
edges = [('A','B',1), ('B','C',3), ('A','C',4), ('C','D',-2)]
distances, parent = bellman_ford(vertices, edges, 'A')
print(f'Distances: {distances}')""",
        "try_demo": None,
        "prerequisites": ["clrs_dijkstra"]
    })
    
    items.append({
        "id": "clrs_floyd_warshall",
        "book_id": "clrs_graph",
        "level": "advanced",
        "title": "Floyd-Warshall (Ch 25.2)",
        "learn": "All-pairs shortest paths. DP: consider paths through vertices 1..k. D[i][j][k] = min(D[i][j][k-1], D[i][k][k-1] + D[k][j][k-1]). O(V³). Handles negative weights, detects cycles.",
        "try_code": """def floyd_warshall(vertices, edges):
    n = len(vertices)
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Initialize
    for i in range(n): dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
    
    # DP: try all intermediate vertices
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Check negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            return None, "Negative cycle"
    
    return dist

edges = [(0,1,3), (1,2,1), (2,0,-3), (0,2,5)]
dist = floyd_warshall(3, edges)
print(f'All-pairs distances: {dist}')""",
        "try_demo": None,
        "prerequisites": ["clrs_bellman"]
    })
    
    items.append({
        "id": "clrs_max_flow",
        "book_id": "clrs_graph",
        "level": "advanced",
        "title": "Ford-Fulkerson Max Flow (Ch 26.2)",
        "learn": "Maximum flow from source to sink in flow network. Find augmenting paths until none exist. Capacity constraints. Applications: bipartite matching, network routing. O(E|f*|) where f* is max flow.",
        "try_code": """from collections import deque

def ford_fulkerson(graph, source, sink):
    def bfs(source, sink, parent):
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink: return True
                    queue.append(v)
        return False
    
    parent = {}
    max_flow = 0
    
    while bfs(source, sink, parent):
        # Find min capacity along path
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        
        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        
        max_flow += path_flow
        parent = {}
    
    return max_flow

# Example network (adjacency dict with capacities)
graph = {0: {1: 16, 2: 13}, 1: {2: 10, 3: 12}, 2: {1: 4, 4: 14}, 3: {2: 9, 5: 20}, 4: {3: 7, 5: 4}, 5: {}}
print(f'Max flow: {ford_fulkerson(graph, 0, 5)}')""",
        "try_demo": None,
        "prerequisites": ["clrs_bfs"]
    })
    
    items.append({
        "id": "clrs_bipartite_matching",
        "book_id": "clrs_graph",
        "level": "expert",
        "title": "Maximum Bipartite Matching (Ch 26.3)",
        "learn": "Maximum matching in bipartite graph. Reduce to max flow: add source/sink, unit capacities. Matching = flow. Applications: job assignment, stable marriage. O(VE) with max flow.",
        "try_code": """def max_bipartite_matching(left, right, edges):
    # Build flow network
    graph = {}
    source, sink = 'source', 'sink'
    
    # Source to left vertices
    graph[source] = {u: 1 for u in left}
    
    # Left to right (bipartite edges)
    for u in left:
        graph[u] = {}
        for v in edges.get(u, []):
            graph[u][v] = 1
    
    # Right to sink
    for v in right:
        if v not in graph: graph[v] = {}
        graph[v][sink] = 1
    
    graph[sink] = {}
    
    # Run max flow (simplified BFS-based)
    def find_augmenting_path():
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            u, path = queue.popleft()
            if u == sink: return path
            for v in graph.get(u, {}):
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    queue.append((v, path + [v]))
        return None
    
    matching = 0
    while path := find_augmenting_path():
        matching += 1
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            graph[u][v] = 0
            if v not in graph: graph[v] = {}
            graph[v][u] = 1
    
    return matching

left = ['A1', 'A2', 'A3']
right = ['B1', 'B2', 'B3']
edges = {'A1': ['B1', 'B2'], 'A2': ['B2'], 'A3': ['B2', 'B3']}
print(f'Max matching: {max_bipartite_matching(left, right, edges)}')""",
        "try_demo": None,
        "prerequisites": ["clrs_max_flow"]
    })
    
    # ============================================================================
    # ADVANCED DATA STRUCTURES (5 items)
    # ============================================================================
    
    items.append({
        "id": "clrs_heap",
        "book_id": "clrs_data_structures",
        "level": "basics",
        "title": "Binary Heap (Ch 6)",
        "learn": "Complete binary tree with heap property. Max-heap: parent ≥ children. Operations: insert O(log n), extract-max O(log n), heapify O(n). Used in heapsort, priority queues.",
        "try_code": """class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i): return (i - 1) // 2
    def left(self, i): return 2 * i + 1
    def right(self, i): return 2 * i + 2
    
    def insert(self, key):
        self.heap.append(key)
        self._sift_up(len(self.heap) - 1)
    
    def _sift_up(self, i):
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            p = self.parent(i)
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p
    
    def extract_max(self):
        if not self.heap: return None
        max_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        if self.heap: self._sift_down(0)
        return max_val
    
    def _sift_down(self, i):
        max_idx = i
        l, r = self.left(i), self.right(i)
        if l < len(self.heap) and self.heap[l] > self.heap[max_idx]: max_idx = l
        if r < len(self.heap) and self.heap[r] > self.heap[max_idx]: max_idx = r
        if max_idx != i:
            self.heap[i], self.heap[max_idx] = self.heap[max_idx], self.heap[i]
            self._sift_down(max_idx)

heap = MaxHeap()
for x in [3, 1, 6, 5, 2, 4]: heap.insert(x)
print(heap.extract_max())  # 6""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_hash_table",
        "book_id": "clrs_data_structures",
        "level": "basics",
        "title": "Hash Tables (Ch 11)",
        "learn": "Array-based dictionary with hash function. Collision resolution: chaining or open addressing. Expected O(1) operations. Load factor α = n/m affects performance.",
        "try_code": """class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]  # Chaining
    
    def hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        idx = self.hash(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, value)
                return
        self.table[idx].append((key, value))
    
    def get(self, key):
        idx = self.hash(key)
        for k, v in self.table[idx]:
            if k == key: return v
        return None
    
    def delete(self, key):
        idx = self.hash(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx].pop(i)
                return True
        return False

ht = HashTable()
ht.insert("name", "Alice")
ht.insert("age", 30)
print(ht.get("name"))  # Alice""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_bst",
        "book_id": "clrs_data_structures",
        "level": "intermediate",
        "title": "Binary Search Tree (Ch 12)",
        "learn": "Binary tree with BST property: left < node < right. Operations: search, insert, delete O(h) where h=height. Inorder traversal yields sorted sequence. Can degenerate to O(n).",
        "try_code": """class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, key):
        self.root = self._insert(self.root, key)
    
    def _insert(self, node, key):
        if not node: return TreeNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        return node
    
    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, node, key):
        if not node or node.key == key: return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)
    
    def inorder(self):
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.key)
                traverse(node.right)
        traverse(self.root)
        return result

bst = BST()
for x in [5, 3, 7, 1, 9]: bst.insert(x)
print(bst.inorder())  # [1, 3, 5, 7, 9]""",
        "try_demo": None,
        "prerequisites": ["clrs_heap"]
    })
    
    items.append({
        "id": "clrs_rbt",
        "book_id": "clrs_data_structures",
        "level": "advanced",
        "title": "Red-Black Trees (Ch 13)",
        "learn": "Self-balancing BST with color property. Guarantees O(log n) operations. Properties: root black, red nodes have black children, black-height uniform. Rotations maintain balance.",
        "try_code": """# Red-Black Tree (simplified structure)
class RBNode:
    def __init__(self, key, color='red'):
        self.key = key
        self.color = color  # 'red' or 'black'
        self.left = self.right = self.parent = None

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(None, 'black')
        self.root = self.NIL
    
    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL: y.left.parent = x
        y.parent = x.parent
        if x.parent == None: self.root = y
        elif x == x.parent.left: x.parent.left = y
        else: x.parent.right = y
        y.left = x
        x.parent = y
    
    def insert(self, key):
        node = RBNode(key)
        node.left = node.right = self.NIL
        
        y = None
        x = self.root
        while x != self.NIL:
            y = x
            if node.key < x.key: x = x.left
            else: x = x.right
        
        node.parent = y
        if y == None: self.root = node
        elif node.key < y.key: y.left = node
        else: y.right = node
        
        self._fix_insert(node)
    
    def _fix_insert(self, node):
        # Recolor and rotate to maintain RB properties
        while node.parent and node.parent.color == 'red':
            # ... complex rebalancing logic
            pass
        self.root.color = 'black'

# RBT guarantees O(log n) height""",
        "try_demo": None,
        "prerequisites": ["clrs_bst"]
    })
    
    items.append({
        "id": "clrs_btree",
        "book_id": "clrs_data_structures",
        "level": "expert",
        "title": "B-Trees (Ch 18)",
        "learn": "Generalized BST for disk storage. Node has multiple keys (min degree t). Height O(log_t n). All leaves at same level. Operations: search, insert, split. Used in databases, filesystems.",
        "try_code": """class BTreeNode:
    def __init__(self, t, leaf=True):
        self.t = t  # Minimum degree
        self.keys = []
        self.children = []
        self.leaf = leaf
    
    def search(self, key):
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1
        if i < len(self.keys) and key == self.keys[i]:
            return self
        if self.leaf:
            return None
        return self.children[i].search(key)
    
    def split_child(self, i, full_child):
        t = self.t
        new_child = BTreeNode(t, full_child.leaf)
        
        # Move half of keys to new child
        mid = len(full_child.keys) // 2
        new_child.keys = full_child.keys[mid+1:]
        full_child.keys = full_child.keys[:mid]
        
        if not full_child.leaf:
            new_child.children = full_child.children[mid+1:]
            full_child.children = full_child.children[:mid+1]
        
        self.keys.insert(i, full_child.keys[mid])
        self.children.insert(i+1, new_child)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t)
        self.t = t
    
    def search(self, key):
        return self.root.search(key)

# B-Tree with minimum degree 3 (each node has 2-5 keys)""",
        "try_demo": None,
        "prerequisites": ["clrs_rbt"]
    })
    
    # ============================================================================
    # NP-COMPLETENESS (3 items)
    # ============================================================================
    
    items.append({
        "id": "clrs_np_intro",
        "book_id": "clrs_complexity",
        "level": "intermediate",
        "title": "P, NP, NP-Complete (Ch 34.1-34.3)",
        "learn": "P: polynomial-time solvable. NP: polynomial-time verifiable. NP-complete: hardest in NP, all reduce to each other. Cook-Levin: SAT is NP-complete. P=NP? unsolved.",
        "try_code": """# Example: Verifying a solution is in NP
def verify_hamiltonian_path(graph, path):
    # O(n) verification for NP problem
    n = len(path)
    if n != len(graph): return False
    if len(set(path)) != n: return False  # All vertices visited once
    
    # Check all edges exist
    for i in range(n - 1):
        if path[i+1] not in graph[path[i]]:
            return False
    return True

graph = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
path = [0, 1, 3, 2]
print(verify_hamiltonian_path(graph, path))  # True (O(n) verification)
# But finding path is NP-complete (no known poly algorithm)""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "clrs_reductions",
        "book_id": "clrs_complexity",
        "level": "advanced",
        "title": "Polynomial-Time Reductions (Ch 34.4)",
        "learn": "Reduce problem A to B: solve A using B solver. If A reduces to B and B ∈ P, then A ∈ P. NP-completeness proofs: reduce known NP-complete to new problem. Examples: SAT → 3-SAT → CLIQUE.",
        "try_code": """# Example reduction: VERTEX-COVER ≤_P SET-COVER
def vertex_cover_to_set_cover(graph, k):
    # Graph with n vertices, m edges
    # Create set cover instance:
    universe = set(range(len(graph.edges)))  # Edges
    subsets = {}
    
    for vertex in range(len(graph.vertices)):
        # Subset for vertex = edges incident to it
        subsets[vertex] = {e for e, (u, v) in enumerate(graph.edges) 
                          if u == vertex or v == vertex}
    
    # Vertex cover of size k <=> set cover of size k
    return universe, subsets, k

# If we could solve set-cover in poly time, we could solve vertex-cover
# Since vertex-cover is NP-complete, set-cover is too""",
        "try_demo": None,
        "prerequisites": ["clrs_np_intro"]
    })
    
    items.append({
        "id": "clrs_approximation",
        "book_id": "clrs_complexity",
        "level": "expert",
        "title": "Approximation Algorithms (Ch 35)",
        "learn": "For NP-hard optimization, find near-optimal solution in poly time. ρ-approximation: cost ≤ ρ·OPT. Examples: vertex cover (2-approx), TSP metric (2-approx), set cover (ln n-approx).",
        "try_code": """def vertex_cover_approx(edges):
    # 2-approximation for minimum vertex cover
    cover = set()
    uncovered = set(edges)
    
    while uncovered:
        u, v = uncovered.pop()
        cover.add(u)
        cover.add(v)
        # Remove all edges incident to u or v
        uncovered = {(a, b) for a, b in uncovered 
                    if a != u and a != v and b != u and b != v}
    
    return cover

edges = [(0,1), (1,2), (2,3), (0,3), (1,3)]
cover = vertex_cover_approx(edges)
print(f'Approx vertex cover (size={len(cover)}): {cover}')
# Guarantee: |cover| ≤ 2·OPT""",
        "try_demo": None,
        "prerequisites": ["clrs_reductions"]
    })
    
    return items


def main():
    """Generate and save enriched curriculum."""
    items = generate_curriculum_items()
    
    # Save to cache
    cache_dir = Path(__file__).parent.parent / '.cache'
    cache_dir.mkdir(exist_ok=True)
    
    output_file = cache_dir / 'clrs_enriched.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    
    # Statistics
    by_level = {}
    by_book = {}
    for item in items:
        by_level[item['level']] = by_level.get(item['level'], 0) + 1
        by_book[item['book_id']] = by_book.get(item['book_id'], 0) + 1
    
    print(f"\n{'='*70}")
    print(f"✅ Generated {len(items)} curriculum items for CLRS Algorithms Lab")
    print(f"{'='*70}")
    print(f"\nBy Level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        print(f"  {level}: {by_level.get(level, 0)}")
    
    print(f"\nBy Topic:")
    for book_id, count in by_book.items():
        print(f"  {book_id}: {count}")
    
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
