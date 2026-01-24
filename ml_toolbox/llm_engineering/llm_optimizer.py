"""
LLM Optimization - Optimize LLM usage for performance and cost
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LLMOptimizer:
    """
    LLM Optimizer
    
    Optimizes:
    - Prompt length
    - Token usage
    - Response quality
    - Cost efficiency
    """
    
    def __init__(self):
        self.optimization_history = []
        self.token_usage = []
        self.cost_tracking = {}
    
    def optimize_prompt_length(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Optimize prompt to fit within token limit
        
        Parameters
        ----------
        prompt : str
            Original prompt
        max_tokens : int
            Maximum tokens allowed
            
        Returns
        -------
        optimized_prompt : str
            Optimized prompt
        """
        # Simple token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Truncate intelligently
        # Keep first part (instructions) and last part (task)
        lines = prompt.split('\n')
        important_lines = [line for line in lines if any(keyword in line.lower() 
                          for keyword in ['task', 'instruction', 'example', 'output'])]
        
        optimized = '\n'.join(important_lines)
        
        # If still too long, truncate
        if len(optimized) // 4 > max_tokens:
            optimized = optimized[:max_tokens * 4]
        
        return optimized
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count
        
        Parameters
        ----------
        text : str
            Text to estimate
            
        Returns
        -------
        token_count : int
            Estimated token count
        """
        # Simple estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def optimize_for_cost(self, prompt: str, model_pricing: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize for cost efficiency
        
        Parameters
        ----------
        prompt : str
            Prompt to optimize
        model_pricing : dict
            Pricing per token for different models
            
        Returns
        -------
        optimization : dict
            Optimization recommendations
        """
        token_count = self.estimate_tokens(prompt)
        
        recommendations = {
            'current_tokens': token_count,
            'cost_estimates': {},
            'recommendations': []
        }
        
        # Estimate cost for each model
        for model, price_per_token in model_pricing.items():
            cost = token_count * price_per_token
            recommendations['cost_estimates'][model] = cost
        
        # Recommendations
        if token_count > 2000:
            recommendations['recommendations'].append(
                "Prompt is long. Consider using RAG to reduce prompt size."
            )
        
        if token_count > 4000:
            recommendations['recommendations'].append(
                "Prompt is very long. Consider breaking into multiple calls."
            )
        
        # Find cheapest model
        if recommendations['cost_estimates']:
            cheapest_model = min(recommendations['cost_estimates'].items(), 
                                key=lambda x: x[1])
            recommendations['cheapest_model'] = cheapest_model[0]
            recommendations['cheapest_cost'] = cheapest_model[1]
        
        return recommendations
    
    def cache_key(self, prompt: str) -> str:
        """
        Generate cache key for prompt
        
        Parameters
        ----------
        prompt : str
            Prompt text
            
        Returns
        -------
        cache_key : str
            Hash-like key for caching
        """
        # Simple hash (in production, use proper hashing)
        return str(hash(prompt))
    
    def should_cache(self, prompt: str, response: str) -> bool:
        """
        Determine if response should be cached
        
        Parameters
        ----------
        prompt : str
            Prompt text
        response : str
            Response text
            
        Returns
        -------
        should_cache : bool
            Whether to cache
        """
        # Cache if prompt is deterministic and response is useful
        # Simple heuristic: cache if response length > 100
        return len(response) > 100 and 'error' not in response.lower()
    
    def track_usage(self, prompt: str, response: str, model: str, cost: float):
        """
        Track LLM usage
        
        Parameters
        ----------
        prompt : str
            Prompt used
        response : str
            Response received
        model : str
            Model used
        cost : float
            Cost incurred
        """
        usage = {
            'prompt_tokens': self.estimate_tokens(prompt),
            'response_tokens': self.estimate_tokens(response),
            'total_tokens': self.estimate_tokens(prompt) + self.estimate_tokens(response),
            'model': model,
            'cost': cost
        }
        
        self.token_usage.append(usage)
        
        if model not in self.cost_tracking:
            self.cost_tracking[model] = 0.0
        self.cost_tracking[model] += cost
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        if not self.token_usage:
            return {'total_calls': 0}
        
        total_tokens = sum(u['total_tokens'] for u in self.token_usage)
        total_cost = sum(u['cost'] for u in self.token_usage)
        
        return {
            'total_calls': len(self.token_usage),
            'total_tokens': total_tokens,
            'avg_tokens_per_call': total_tokens / len(self.token_usage),
            'total_cost': total_cost,
            'cost_by_model': self.cost_tracking
        }
