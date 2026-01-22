"""
Tests for Sedgewick & Wayne Algorithms
Test priority queues, symbol tables, graph algorithms, sorting, string structures
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sedgewick_wayne_algorithms import (
        IndexedPriorityQueue,
        OrderedSymbolTable,
        AStarSearch,
        BidirectionalSearch,
        ThreeWayQuicksort,
        Trie,
        BloomFilter,
        SedgewickWayneAlgorithms
    )
    SEDGEWICK_WAYNE_AVAILABLE = True
except ImportError:
    SEDGEWICK_WAYNE_AVAILABLE = False
    pytestmark = pytest.mark.skip("Sedgewick & Wayne algorithms not available")


class TestIndexedPriorityQueue:
    """Tests for indexed priority queue"""
    
    def test_insert_and_delete_min(self):
        """Test insert and delete min"""
        pq = IndexedPriorityQueue(10)
        pq.insert(0, 5.0)
        pq.insert(1, 3.0)
        pq.insert(2, 7.0)
        
        assert pq.min_index() == 1
        assert pq.min_key() == 3.0
        
        min_idx = pq.delete_min()
        assert min_idx == 1
        assert pq.min_index() == 0
    
    def test_change_key(self):
        """Test change key"""
        pq = IndexedPriorityQueue(10)
        pq.insert(0, 5.0)
        pq.insert(1, 3.0)
        
        pq.change_key(0, 1.0)
        assert pq.min_index() == 0
        assert pq.min_key() == 1.0
    
    def test_contains(self):
        """Test contains"""
        pq = IndexedPriorityQueue(10)
        pq.insert(0, 5.0)
        
        assert pq.contains(0)
        assert not pq.contains(1)


class TestOrderedSymbolTable:
    """Tests for ordered symbol table"""
    
    def test_put_and_get(self):
        """Test put and get"""
        st = OrderedSymbolTable()
        st.put('b', 2)
        st.put('a', 1)
        st.put('c', 3)
        
        assert st.get('a') == 1
        assert st.get('b') == 2
        assert st.min_key() == 'a'
        assert st.max_key() == 'c'
    
    def test_floor_and_ceiling(self):
        """Test floor and ceiling"""
        st = OrderedSymbolTable()
        st.put('b', 2)
        st.put('d', 4)
        
        assert st.floor('c') == 'b'
        assert st.ceiling('c') == 'd'
    
    def test_range_keys(self):
        """Test range keys"""
        st = OrderedSymbolTable()
        for i in range(10):
            st.put(i, i)
        
        keys = st.range_keys(3, 7)
        assert keys == [3, 4, 5, 6, 7]


class TestAStarSearch:
    """Tests for A* search"""
    
    def test_astar_search(self):
        """Test A* search"""
        graph = {
            'A': [('B', 1), ('C', 3)],
            'B': [('D', 2)],
            'C': [('D', 1)],
            'D': []
        }
        
        def heuristic(node):
            # Simple heuristic: distance to goal
            distances = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
            return distances.get(node, 0)
        
        result = AStarSearch.search(graph, 'A', 'D', heuristic)
        assert result is not None
        path, cost = result
        assert path[0] == 'A'
        assert path[-1] == 'D'


class TestBidirectionalSearch:
    """Tests for bidirectional search"""
    
    def test_bidirectional_search(self):
        """Test bidirectional search"""
        graph = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }
        
        path = BidirectionalSearch.search(graph, 'A', 'D')
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'D'


class TestThreeWayQuicksort:
    """Tests for 3-way quicksort"""
    
    def test_sort_with_duplicates(self):
        """Test sorting with duplicates"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        sorted_arr = ThreeWayQuicksort.sort(arr)
        
        assert sorted_arr == [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]
    
    def test_sort_empty(self):
        """Test sorting empty list"""
        arr = []
        sorted_arr = ThreeWayQuicksort.sort(arr)
        assert sorted_arr == []


class TestTrie:
    """Tests for trie"""
    
    def test_insert_and_search(self):
        """Test insert and search"""
        trie = Trie()
        trie.insert('hello', 1)
        trie.insert('world', 2)
        
        assert trie.search('hello') == 1
        assert trie.search('world') == 2
        assert trie.search('hi') is None
    
    def test_starts_with(self):
        """Test starts with"""
        trie = Trie()
        trie.insert('hello')
        trie.insert('world')
        
        assert trie.starts_with('he')
        assert trie.starts_with('wor')
        assert not trie.starts_with('xyz')
    
    def test_keys_with_prefix(self):
        """Test keys with prefix"""
        trie = Trie()
        trie.insert('hello')
        trie.insert('help')
        trie.insert('world')
        
        keys = trie.keys_with_prefix('hel')
        assert 'hello' in keys
        assert 'help' in keys
        assert 'world' not in keys


class TestBloomFilter:
    """Tests for bloom filter"""
    
    def test_add_and_contains(self):
        """Test add and contains"""
        bf = BloomFilter(100)
        bf.add('hello')
        bf.add('world')
        
        assert bf.contains('hello')
        assert bf.contains('world')
        # May have false positives, but should not have false negatives
        assert bf.contains('hello')  # Should definitely be true


class TestSedgewickWayneAlgorithms:
    """Test unified framework"""
    
    def test_unified_interface(self):
        """Test SedgewickWayneAlgorithms"""
        sw = SedgewickWayneAlgorithms()
        
        assert sw.indexed_priority_queue is not None
        assert sw.ordered_symbol_table is not None
        assert sw.astar is not None
        assert sw.bidirectional_search is not None
        assert sw.three_way_quicksort is not None
        assert sw.trie is not None
        assert sw.bloom_filter is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
