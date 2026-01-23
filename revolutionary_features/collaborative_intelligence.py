"""
Collaborative Intelligence System
Multiple toolbox instances share knowledge and learn from each other

Revolutionary: Your toolbox learns from all users (privacy-preserving)
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import hashlib
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


class CollaborativeIntelligence:
    """
    Collaborative Intelligence System
    
    Innovation: Toolbox instances share knowledge:
    - What patterns work best
    - What configurations are optimal
    - What errors to avoid
    - Performance insights
    
    All while preserving privacy (no data sharing, only patterns)
    """
    
    def __init__(self, enable_sharing: bool = True):
        self.enable_sharing = enable_sharing
        self.local_knowledge = defaultdict(dict)
        self.shared_patterns = {}
        self.contribution_history = []
    
    def learn_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], performance: float):
        """Learn a pattern locally"""
        pattern_key = self._generate_pattern_key(pattern_type, pattern_data)
        
        self.local_knowledge[pattern_type][pattern_key] = {
            'pattern': pattern_data,
            'performance': performance,
            'timestamp': time.time(),
            'usage_count': self.local_knowledge[pattern_type].get(pattern_key, {}).get('usage_count', 0) + 1
        }
        
        # Share if enabled and pattern is good
        if self.enable_sharing and performance > 0.8:
            self._share_pattern(pattern_type, pattern_key, pattern_data, performance)
    
    def _generate_pattern_key(self, pattern_type: str, pattern_data: Dict[str, Any]) -> str:
        """Generate unique key for pattern"""
        # Hash pattern data (not actual data, just structure)
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def _share_pattern(self, pattern_type: str, pattern_key: str, pattern_data: Dict[str, Any], performance: float):
        """Share pattern with community (conceptual - would use network in real implementation)"""
        # In real implementation, this would send to a central server
        # For now, just store locally as "shared"
        
        shared_pattern = {
            'pattern_type': pattern_type,
            'pattern_key': pattern_key,
            'pattern_structure': self._extract_structure(pattern_data),  # No actual data
            'performance': performance,
            'timestamp': time.time()
        }
        
        self.shared_patterns[pattern_key] = shared_pattern
        
        # Record contribution
        self.contribution_history.append({
            'pattern_key': pattern_key,
            'timestamp': time.time()
        })
    
    def _extract_structure(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structure without actual data (privacy-preserving)"""
        structure = {}
        
        for key, value in pattern_data.items():
            if isinstance(value, (int, float)):
                structure[key] = type(value).__name__
            elif isinstance(value, (list, tuple)):
                structure[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                structure[key] = self._extract_structure(value)
            else:
                structure[key] = type(value).__name__
        
        return structure
    
    def get_recommended_pattern(self, pattern_type: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get recommended pattern from community knowledge"""
        # Check local knowledge first
        if pattern_type in self.local_knowledge:
            patterns = self.local_knowledge[pattern_type]
            if patterns:
                # Get best performing pattern
                best_pattern = max(patterns.values(), key=lambda p: p['performance'])
                return best_pattern['pattern']
        
        # Check shared patterns
        if self.shared_patterns:
            matching_patterns = [
                p for p in self.shared_patterns.values()
                if p['pattern_type'] == pattern_type
            ]
            if matching_patterns:
                # Get most recent high-performing pattern
                best_shared = max(matching_patterns, key=lambda p: p['performance'])
                return best_shared.get('pattern_structure', {})
        
        return None
    
    def get_community_insights(self) -> Dict[str, Any]:
        """Get insights from community"""
        insights = {
            'total_patterns_shared': len(self.shared_patterns),
            'patterns_contributed': len(self.contribution_history),
            'pattern_types': list(set(p['pattern_type'] for p in self.shared_patterns.values())),
            'average_performance': sum(p['performance'] for p in self.shared_patterns.values()) / len(self.shared_patterns) if self.shared_patterns else 0
        }
        
        return insights


# Global instance
_global_collaborative = None

def get_collaborative_intelligence(enable_sharing: bool = True) -> CollaborativeIntelligence:
    """Get global collaborative intelligence instance"""
    global _global_collaborative
    if _global_collaborative is None:
        _global_collaborative = CollaborativeIntelligence(enable_sharing=enable_sharing)
    return _global_collaborative
