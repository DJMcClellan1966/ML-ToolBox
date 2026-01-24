"""
Attention Mechanism - Focus and Filtering

Brain-like attention for focusing on relevant information
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionWeight:
    """Attention weight"""
    item: Any
    weight: float
    reason: str = ""


class FocusFilter:
    """
    Focus Filter
    
    Filters information based on relevance and importance
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize focus filter
        
        Parameters
        ----------
        top_k : int
            Number of items to focus on
        """
        self.top_k = top_k
    
    def filter(self, items: List[Any], relevance_scores: List[float]) -> List[Any]:
        """
        Filter items by relevance
        
        Parameters
        ----------
        items : list
            Items to filter
        relevance_scores : list
            Relevance scores
            
        Returns
        -------
        filtered : list
            Top-k most relevant items
        """
        if len(items) != len(relevance_scores):
            return items
        
        # Sort by relevance
        indexed = list(zip(relevance_scores, items))
        indexed.sort(reverse=True)
        
        # Return top-k
        return [item for _, item in indexed[:self.top_k]]


class AttentionMechanism:
    """
    Attention Mechanism
    
    Brain-like attention for focusing on relevant information
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize attention mechanism
        
        Parameters
        ----------
        top_k : int
            Number of items to attend to
        """
        self.top_k = top_k
        self.focus_filter = FocusFilter(top_k)
        self.attention_history: List[Dict] = []
    
    def attend(self, items: List[Any], query: Optional[str] = None,
              importance_weights: Optional[List[float]] = None) -> List[AttentionWeight]:
        """
        Apply attention to items
        
        Parameters
        ----------
        items : list
            Items to attend to
        query : str, optional
            Query for relevance
        importance_weights : list, optional
            Pre-computed importance weights
            
        Returns
        -------
        attention_weights : list
            Items with attention weights
        """
        if not items:
            return []
        
        # Compute attention weights
        if importance_weights:
            weights = importance_weights
        elif query:
            # Simple relevance-based attention
            weights = self._compute_relevance_weights(items, query)
        else:
            # Uniform attention
            weights = [1.0 / len(items)] * len(items)
        
        # Normalize (softmax-like)
        weights = self._softmax(weights)
        
        # Create attention weights
        attention_weights = [
            AttentionWeight(item=item, weight=weight, reason="relevance")
            for item, weight in zip(items, weights)
        ]
        
        # Sort by weight
        attention_weights.sort(key=lambda x: x.weight, reverse=True)
        
        # Record attention
        self.attention_history.append({
            'items_count': len(items),
            'focused_count': min(self.top_k, len(items)),
            'max_weight': attention_weights[0].weight if attention_weights else 0.0
        })
        
        return attention_weights[:self.top_k]
    
    def _compute_relevance_weights(self, items: List[Any], query: str) -> List[float]:
        """Compute relevance weights"""
        query_lower = query.lower()
        weights = []
        
        for item in items:
            if isinstance(item, str):
                # Simple keyword matching
                item_lower = item.lower()
                score = sum(1 for word in query_lower.split() if word in item_lower)
                weights.append(score / len(query_lower.split()) if query_lower.split() else 0.0)
            else:
                # Default relevance
                weights.append(0.5)
        
        return weights
    
    def _softmax(self, values: List[float]) -> List[float]:
        """Softmax normalization"""
        if not values:
            return []
        
        exp_values = [np.exp(v) for v in values]
        sum_exp = sum(exp_values)
        
        if sum_exp == 0:
            return [1.0 / len(values)] * len(values)
        
        return [v / sum_exp for v in exp_values]
    
    def get_focused_items(self, items: List[Any], query: Optional[str] = None) -> List[Any]:
        """Get focused items (top-k)"""
        attention_weights = self.attend(items, query)
        return [aw.item for aw in attention_weights]
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention statistics"""
        if not self.attention_history:
            return {}
        
        return {
            'total_attention_events': len(self.attention_history),
            'avg_items_per_event': np.mean([h['items_count'] for h in self.attention_history]),
            'avg_focused_count': np.mean([h['focused_count'] for h in self.attention_history])
        }
