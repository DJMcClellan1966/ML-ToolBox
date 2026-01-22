"""
Tests for CLRS Complete Algorithms
Test missing CLRS algorithms: DP, graph, greedy
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from clrs_complete_algorithms import (
        CLRSDynamicProgramming,
        CLRSGraphAlgorithms,
        CLRSGreedyAlgorithms,
        CLRSComplete
    )
    CLRS_COMPLETE_AVAILABLE = True
except ImportError:
    CLRS_COMPLETE_AVAILABLE = False
    pytestmark = pytest.mark.skip("CLRS complete algorithms not available")


class TestCLRSDynamicProgramming:
    """Tests for additional CLRS dynamic programming"""
    
    def test_optimal_binary_search_tree(self):
        """Test optimal BST"""
        keys = [10, 20, 30]
        freq = [0.5, 0.3, 0.2]
        
        min_cost, root = CLRSDynamicProgramming.optimal_binary_search_tree(keys, freq)
        assert min_cost > 0
        assert root is not None
    
    def test_longest_increasing_subsequence(self):
        """Test LIS"""
        arr = [10, 22, 9, 33, 21, 50, 41, 60]
        length, indices = CLRSDynamicProgramming.longest_increasing_subsequence(arr)
        assert length == 5
        assert len(indices) == length
    
    def test_coin_change_min_coins(self):
        """Test coin change"""
        coins = [1, 5, 10, 25]
        amount = 30
        
        min_coins, combination = CLRSDynamicProgramming.coin_change_min_coins(coins, amount)
        assert min_coins > 0
        assert sum(combination) == amount
    
    def test_rod_cutting(self):
        """Test rod cutting"""
        prices = [1, 5, 8, 9, 10, 17, 17, 20]
        length = 8
        
        max_profit, cuts = CLRSDynamicProgramming.rod_cutting(prices, length)
        assert max_profit > 0
        assert sum(cuts) == length


class TestCLRSGraphAlgorithms:
    """Tests for additional CLRS graph algorithms"""
    
    def test_bellman_ford(self):
        """Test Bellman-Ford"""
        graph = {
            0: [(1, 6), (2, 5), (3, 5)],
            1: [(2, -2), (4, -1)],
            2: [(4, 1)],
            3: [(2, -2), (5, -1)],
            4: [(5, 3)],
            5: []
        }
        
        dist, has_cycle = CLRSGraphAlgorithms.bellman_ford(graph, 0, 6)
        assert 0 in dist
        assert dist[0] == 0
        assert not has_cycle
    
    def test_bipartite_matching(self):
        """Test bipartite matching"""
        graph = {
            0: [3, 4],
            1: [3],
            2: [4, 5]
        }
        left = [0, 1, 2]
        right = [3, 4, 5]
        
        matching = CLRSGraphAlgorithms.bipartite_matching(graph, left, right)
        assert len(matching) > 0
        assert all(u in left and v in right for u, v in matching)


class TestCLRSGreedyAlgorithms:
    """Tests for additional CLRS greedy algorithms"""
    
    def test_prims_mst(self):
        """Test Prim's MST"""
        graph = {
            0: [(1, 4), (2, 2)],
            1: [(0, 4), (2, 1), (3, 5)],
            2: [(0, 2), (1, 1), (3, 8), (4, 10)],
            3: [(1, 5), (2, 8), (4, 2)],
            4: [(2, 10), (3, 2)]
        }
        
        mst = CLRSGreedyAlgorithms.prims_mst(graph, 0)
        assert len(mst) == 4  # n-1 edges
        total_weight = sum(edge[2] for edge in mst)
        assert total_weight > 0
    
    def test_activity_selection(self):
        """Test activity selection"""
        activities = [(1, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
        selected = CLRSGreedyAlgorithms.activity_selection(activities)
        assert len(selected) > 0
        # Check no overlaps
        selected_acts = [activities[i] for i in selected]
        selected_acts.sort(key=lambda x: x[1])
        for i in range(len(selected_acts) - 1):
            assert selected_acts[i][1] <= selected_acts[i+1][0]
    
    def test_set_cover_greedy(self):
        """Test set cover greedy"""
        universe = {1, 2, 3, 4, 5}
        subsets = [
            {1, 2, 3},
            {2, 4},
            {3, 4, 5},
            {1, 5}
        ]
        
        selected = CLRSGreedyAlgorithms.set_cover_greedy(universe, subsets)
        assert len(selected) > 0
        # Check coverage
        covered = set()
        for idx in selected:
            covered |= subsets[idx]
        assert covered == universe


class TestCLRSComplete:
    """Test unified interface"""
    
    def test_unified_interface(self):
        """Test CLRSComplete"""
        clrs = CLRSComplete()
        assert clrs.dp is not None
        assert clrs.graph is not None
        assert clrs.greedy is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
