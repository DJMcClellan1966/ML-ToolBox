"""
Predictive Intelligence System
Predicts what user wants to do next and suggests it automatically

Revolutionary: The toolbox anticipates your needs before you ask
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict, deque
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class PredictiveIntelligence:
    """
    Predictive Intelligence System
    
    Innovation: Predicts what you want to do next based on:
    - Your usage patterns
    - Common ML workflows
    - Context of current code
    - Historical patterns
    
    Example: After training a model, suggests evaluation automatically
    """
    
    def __init__(self):
        self.usage_patterns = defaultdict(list)
        self.workflow_sequences = defaultdict(int)
        self.context_history = deque(maxlen=100)
        self.prediction_cache = {}
    
    def record_action(self, action: str, context: Dict[str, Any]):
        """Record user action for learning"""
        self.context_history.append({
            'action': action,
            'context': context,
            'timestamp': time.time()
        })
        
        # Learn patterns
        if len(self.context_history) > 1:
            prev_action = self.context_history[-2]['action']
            sequence = f"{prev_action} -> {action}"
            self.workflow_sequences[sequence] += 1
    
    def predict_next_action(self, current_action: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict what user wants to do next
        
        Returns: List of predicted actions with confidence scores
        """
        predictions = []
        
        # Pattern-based prediction
        for sequence, count in self.workflow_sequences.items():
            if sequence.startswith(f"{current_action} ->"):
                next_action = sequence.split(" -> ")[1]
                confidence = count / max(1, sum(self.workflow_sequences.values()))
                predictions.append({
                    'action': next_action,
                    'confidence': confidence,
                    'type': 'pattern'
                })
        
        # Context-based prediction
        context_predictions = self._predict_from_context(context)
        predictions.extend(context_predictions)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions[:5]  # Top 5 predictions
    
    def _predict_from_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict based on current context"""
        predictions = []
        
        # If model was just trained, suggest evaluation
        if context.get('action') == 'train_model':
            predictions.append({
                'action': 'evaluate_model',
                'confidence': 0.9,
                'type': 'workflow',
                'suggestion': 'Evaluate the trained model'
            })
            predictions.append({
                'action': 'save_model',
                'confidence': 0.7,
                'type': 'workflow',
                'suggestion': 'Save the model for later use'
            })
        
        # If data was preprocessed, suggest training
        if context.get('action') == 'preprocess_data':
            predictions.append({
                'action': 'train_model',
                'confidence': 0.85,
                'type': 'workflow',
                'suggestion': 'Train a model on preprocessed data'
            })
        
        # If model was evaluated, suggest optimization
        if context.get('action') == 'evaluate_model':
            if context.get('accuracy', 0) < 0.8:
                predictions.append({
                    'action': 'optimize_model',
                    'confidence': 0.8,
                    'type': 'workflow',
                    'suggestion': 'Optimize model hyperparameters'
                })
        
        return predictions
    
    def get_suggestions(self, current_context: Dict[str, Any]) -> List[str]:
        """Get human-readable suggestions"""
        current_action = current_context.get('action', 'unknown')
        predictions = self.predict_next_action(current_action, current_context)
        
        suggestions = []
        for pred in predictions:
            if pred.get('suggestion'):
                suggestions.append(pred['suggestion'])
            else:
                suggestions.append(f"Consider: {pred['action']}")
        
        return suggestions


# Global instance
_global_predictive = None

def get_predictive_intelligence() -> PredictiveIntelligence:
    """Get global predictive intelligence instance"""
    global _global_predictive
    if _global_predictive is None:
        _global_predictive = PredictiveIntelligence()
    return _global_predictive
