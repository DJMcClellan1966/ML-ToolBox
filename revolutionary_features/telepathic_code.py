"""
Telepathic Code Feature
Reads your intent and suggests code - "I know what you're thinking!"

Fun & Daring: The toolbox reads your mind and suggests code before you write it!
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox.ai_agent import MLCodeAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class TelepathicCode:
    """
    Telepathic Code Reader
    
    Innovation: Reads your intent and suggests code:
    - Analyzes partial code
    - Detects what you're trying to do
    - Suggests completion
    - Predicts next steps
    - "I know what you're thinking!"
    """
    
    def __init__(self):
        self.agent = MLCodeAgent(use_llm=False, use_pattern_composition=True) if AGENT_AVAILABLE else None
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load intent detection patterns"""
        return {
            'train_model': {
                'indicators': ['toolbox', 'fit', 'train', 'model'],
                'intent': 'Want to train a model',
                'suggestion': 'toolbox.fit(X, y, task_type="classification")'
            },
            'preprocess': {
                'indicators': ['preprocess', 'clean', 'prepare', 'data'],
                'intent': 'Want to preprocess data',
                'suggestion': 'toolbox.universal_preprocessor.preprocess(data)'
            },
            'predict': {
                'indicators': ['predict', 'forecast', 'estimate'],
                'intent': 'Want to make predictions',
                'suggestion': 'toolbox.predict(model_id, X)'
            },
            'classify': {
                'indicators': ['classify', 'category', 'class'],
                'intent': 'Want to classify data',
                'suggestion': 'result = toolbox.fit(X, y, task_type="classification")'
            },
            'regress': {
                'indicators': ['regress', 'predict.*value', 'continuous'],
                'intent': 'Want to do regression',
                'suggestion': 'result = toolbox.fit(X, y, task_type="regression")'
            }
        }
    
    def read_mind(self, partial_code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Read your mind and suggest what you want to do
        
        Analyzes partial code and predicts intent
        """
        detected_intents = []
        suggestions = []
        
        partial_lower = partial_code.lower()
        
        # Detect intents
        for intent_name, intent_data in self.intent_patterns.items():
            score = 0
            for indicator in intent_data['indicators']:
                if indicator in partial_lower:
                    score += 1
            
            if score > 0:
                detected_intents.append({
                    'intent': intent_name,
                    'description': intent_data['intent'],
                    'confidence': min(1.0, score / len(intent_data['indicators'])),
                    'suggestion': intent_data['suggestion']
                })
        
        # Sort by confidence
        detected_intents.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Generate suggestions
        if detected_intents:
            primary_intent = detected_intents[0]
            suggestions.append({
                'type': 'completion',
                'suggestion': primary_intent['suggestion'],
                'confidence': primary_intent['confidence'],
                'message': f"I think you want to: {primary_intent['description']}"
            })
        
        # Predict next steps
        next_steps = self._predict_next_steps(partial_code, detected_intents)
        suggestions.extend(next_steps)
        
        return {
            'detected_intents': detected_intents,
            'primary_intent': detected_intents[0] if detected_intents else None,
            'suggestions': suggestions,
            'message': f"I read your mind! Detected {len(detected_intents)} intents"
        }
    
    def _predict_next_steps(self, code: str, intents: List[Dict]) -> List[Dict[str, Any]]:
        """Predict what you'll want to do next"""
        next_steps = []
        
        if not intents:
            return next_steps
        
        primary_intent = intents[0]['intent']
        
        # Predict next steps based on intent
        if primary_intent == 'train_model':
            next_steps.append({
                'type': 'next_step',
                'suggestion': 'Evaluate the model',
                'code': 'evaluator.evaluate_model(result["model"], X, y)',
                'reason': 'After training, you usually want to evaluate'
            })
            next_steps.append({
                'type': 'next_step',
                'suggestion': 'Save the model',
                'code': 'toolbox.register_model(result["model"], "my_model")',
                'reason': 'You might want to save the trained model'
            })
        
        elif primary_intent == 'preprocess':
            next_steps.append({
                'type': 'next_step',
                'suggestion': 'Train a model on preprocessed data',
                'code': 'result = toolbox.fit(preprocessed_data, y)',
                'reason': 'Preprocessed data is usually used for training'
            })
        
        return next_steps
    
    def complete_thought(self, partial_code: str) -> Dict[str, Any]:
        """Complete your thought - suggest code completion"""
        mind_read = self.read_mind(partial_code)
        
        if mind_read['primary_intent']:
            completion = mind_read['primary_intent']['suggestion']
            return {
                'completion': completion,
                'full_code': partial_code + '\n' + completion,
                'intent': mind_read['primary_intent']['description'],
                'message': 'I completed your thought!'
            }
        
        return {
            'completion': None,
            'message': "I'm not sure what you're thinking - provide more context"
        }


# Global instance
_global_telepathic = None

def get_telepathic_code() -> TelepathicCode:
    """Get global telepathic code reader"""
    global _global_telepathic
    if _global_telepathic is None:
        _global_telepathic = TelepathicCode()
    return _global_telepathic
