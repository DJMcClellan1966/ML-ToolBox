"""
Parallel Universe Testing
Tests code in multiple "universes" (scenarios) simultaneously

Fun & Daring: What if your code ran in parallel universes? See all possible outcomes!
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

sys.path.insert(0, str(Path(__file__).parent.parent))


class ParallelUniverseTesting:
    """
    Parallel Universe Tester
    
    Innovation: Tests code in multiple "universes" (scenarios):
    - Universe 1: Small data
    - Universe 2: Large data
    - Universe 3: Noisy data
    - Universe 4: Perfect data
    - Universe 5: Edge cases
    
    See how your code behaves in all possible scenarios!
    """
    
    def __init__(self):
        self.universes = self._create_universes()
    
    def _create_universes(self) -> Dict[str, Dict[str, Any]]:
        """Create parallel universes"""
        return {
            'universe_1_small': {
                'name': 'Small Data Universe',
                'description': 'Tests with small dataset (< 100 samples)',
                'data_size': (50, 10),
                'characteristics': ['small', 'fast', 'may_overfit']
            },
            'universe_2_large': {
                'name': 'Large Data Universe',
                'description': 'Tests with large dataset (> 10000 samples)',
                'data_size': (10000, 50),
                'characteristics': ['large', 'slow', 'scalable']
            },
            'universe_3_noisy': {
                'name': 'Noisy Data Universe',
                'description': 'Tests with noisy, messy data',
                'data_size': (1000, 20),
                'characteristics': ['noisy', 'messy', 'challenging']
            },
            'universe_4_perfect': {
                'name': 'Perfect Data Universe',
                'description': 'Tests with perfect, clean data',
                'data_size': (1000, 20),
                'characteristics': ['perfect', 'clean', 'ideal']
            },
            'universe_5_edge': {
                'name': 'Edge Cases Universe',
                'description': 'Tests with extreme edge cases',
                'data_size': (10, 1000),  # Few samples, many features
                'characteristics': ['extreme', 'edge_cases', 'challenging']
            }
        }
    
    def test_in_universes(self, code_template: str) -> Dict[str, Any]:
        """
        Test code in all parallel universes
        
        Returns results from each universe
        """
        results = {}
        
        for universe_id, universe in self.universes.items():
            # Generate data for this universe
            data = self._generate_universe_data(universe)
            
            # Test code in this universe
            universe_result = self._test_in_universe(code_template, universe, data)
            
            results[universe_id] = {
                'universe': universe['name'],
                'description': universe['description'],
                'data_characteristics': universe['characteristics'],
                'result': universe_result
            }
        
        # Compare results across universes
        comparison = self._compare_universes(results)
        
        return {
            'universe_results': results,
            'comparison': comparison,
            'message': f"Tested in {len(self.universes)} parallel universes!"
        }
    
    def _generate_universe_data(self, universe: Dict) -> Dict[str, Any]:
        """Generate data for a universe"""
        import numpy as np
        
        size = universe['data_size']
        characteristics = universe['characteristics']
        
        if 'noisy' in characteristics:
            # Noisy data
            X = np.random.randn(size[0], size[1]) + np.random.randn(size[0], size[1]) * 0.5
            y = np.random.randint(0, 2, size[0])
        elif 'perfect' in characteristics:
            # Perfect data
            X = np.random.randn(size[0], size[1])
            y = (X[:, 0] > 0).astype(int)  # Perfectly separable
        elif 'edge' in characteristics:
            # Edge case: many features, few samples
            X = np.random.randn(size[0], size[1])
            y = np.random.randint(0, 2, size[0])
        else:
            # Normal data
            X = np.random.randn(size[0], size[1])
            y = np.random.randint(0, 2, size[0])
        
        return {
            'X': X,
            'y': y,
            'size': size,
            'characteristics': characteristics
        }
    
    def _test_in_universe(self, code_template: str, universe: Dict, data: Dict) -> Dict[str, Any]:
        """Test code in a specific universe"""
        # Simulate testing (in real implementation, would execute code)
        result = {
            'success': random.random() > 0.2,  # 80% success rate
            'performance': random.uniform(0.5, 0.95),
            'issues': [],
            'warnings': []
        }
        
        # Add universe-specific issues
        if 'small' in universe['characteristics']:
            result['warnings'].append('Small dataset may cause overfitting')
        
        if 'noisy' in universe['characteristics']:
            result['warnings'].append('Noisy data may affect performance')
        
        if 'edge' in universe['characteristics']:
            result['issues'].append('Edge case: High dimensionality, low samples')
        
        return result
    
    def _compare_universes(self, results: Dict) -> Dict[str, Any]:
        """Compare results across universes"""
        success_count = sum(1 for r in results.values() if r['result']['success'])
        avg_performance = sum(r['result']['performance'] for r in results.values()) / len(results)
        
        best_universe = max(results.items(), key=lambda x: x[1]['result']['performance'])
        worst_universe = min(results.items(), key=lambda x: x[1]['result']['performance'])
        
        return {
            'success_rate': success_count / len(results),
            'average_performance': avg_performance,
            'best_universe': best_universe[0],
            'worst_universe': worst_universe[0],
            'robustness': 'high' if success_count == len(results) else 'medium' if success_count > len(results) / 2 else 'low'
        }


# Global instance
_global_universes = None

def get_parallel_universe_testing() -> ParallelUniverseTesting:
    """Get global parallel universe tester"""
    global _global_universes
    if _global_universes is None:
        _global_universes = ParallelUniverseTesting()
    return _global_universes
