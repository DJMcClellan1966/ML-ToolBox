"""
LLM Evaluator - Evaluate LLM responses for quality
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    LLM Evaluator
    
    Evaluates:
    - Response quality
    - Relevance
    - Accuracy
    - Completeness
    """
    
    def __init__(self):
        self.evaluation_history = []
        self.quality_metrics = {}
    
    def evaluate_response(self, prompt: str, response: str, 
                         expected_output: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate LLM response
        
        Parameters
        ----------
        prompt : str
            Original prompt
        response : str
            LLM response
        expected_output : str, optional
            Expected output (for accuracy calculation)
            
        Returns
        -------
        evaluation : dict
            Evaluation scores (0-1)
        """
        evaluation = {
            'relevance': self._evaluate_relevance(prompt, response),
            'completeness': self._evaluate_completeness(response),
            'clarity': self._evaluate_clarity(response),
            'structure': self._evaluate_structure(response)
        }
        
        if expected_output:
            evaluation['accuracy'] = self._evaluate_accuracy(response, expected_output)
        
        # Overall score
        evaluation['overall'] = sum(evaluation.values()) / len(evaluation)
        
        # Store evaluation
        self.evaluation_history.append({
            'prompt': prompt[:100],  # Store first 100 chars
            'response': response[:100],
            'scores': evaluation
        })
        
        return evaluation
    
    def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """Evaluate if response is relevant to prompt"""
        # Simple heuristic: check for common keywords
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(prompt_words & response_words)
        total = len(prompt_words)
        
        if total == 0:
            return 0.5  # Neutral if no words
        
        return min(overlap / total, 1.0)
    
    def _evaluate_completeness(self, response: str) -> float:
        """Evaluate if response is complete"""
        # Heuristic: longer responses with structure are more complete
        length_score = min(len(response) / 200, 1.0)  # 200 chars = full score
        
        # Check for structure indicators
        has_structure = any(indicator in response.lower() 
                           for indicator in ['step', 'first', 'then', 'finally', 'conclusion'])
        structure_score = 0.3 if has_structure else 0.0
        
        return min(length_score + structure_score, 1.0)
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity"""
        # Heuristic: check for clear language
        unclear_indicators = ['unclear', 'confusing', 'not sure', 'maybe', 'possibly']
        unclear_count = sum(1 for indicator in unclear_indicators if indicator in response.lower())
        
        clarity_score = max(0.0, 1.0 - (unclear_count * 0.2))
        
        return clarity_score
    
    def _evaluate_structure(self, response: str) -> float:
        """Evaluate response structure"""
        # Check for structured format
        structure_indicators = [
            'step', 'first', 'second', 'third',
            '1.', '2.', '3.',
            'conclusion', 'summary'
        ]
        
        found = sum(1 for indicator in structure_indicators if indicator in response.lower())
        return min(found / 3, 1.0)  # 3+ indicators = full score
    
    def _evaluate_accuracy(self, response: str, expected: str) -> float:
        """Evaluate accuracy against expected output"""
        # Simple word overlap
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.5
        
        overlap = len(response_words & expected_words)
        return min(overlap / len(expected_words), 1.0)
    
    def get_evaluation_stats(self) -> Dict:
        """Get evaluation statistics"""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        all_scores = [e['scores'] for e in self.evaluation_history]
        
        avg_scores = {}
        for key in all_scores[0].keys():
            avg_scores[f'avg_{key}'] = sum(s.get(key, 0) for s in all_scores) / len(all_scores)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            **avg_scores
        }
