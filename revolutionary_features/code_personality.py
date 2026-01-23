"""
Code Personality Feature
Analyzes code and gives it a personality, suggests improvements based on personality traits

Fun & Daring: Your code has a personality! Is it bold? Cautious? Creative?
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


class CodePersonality:
    """
    Code Personality Analyzer
    
    Innovation: Analyzes code and assigns it a personality:
    - Bold (takes risks, tries new things)
    - Cautious (safe, defensive)
    - Creative (innovative, experimental)
    - Efficient (optimized, minimal)
    - Friendly (well-documented, clear)
    
    Then suggests improvements based on personality!
    """
    
    def __init__(self):
        self.personality_traits = {
            'bold': {
                'indicators': ['experimental', 'try', 'new', 'cutting_edge', 'advanced'],
                'description': 'Bold and adventurous - takes risks, tries new approaches'
            },
            'cautious': {
                'indicators': ['check', 'verify', 'validate', 'safe', 'defensive', 'if.*else'],
                'description': 'Cautious and careful - defensive programming, lots of checks'
            },
            'creative': {
                'indicators': ['unique', 'custom', 'creative', 'innovative', 'unusual'],
                'description': 'Creative and innovative - thinks outside the box'
            },
            'efficient': {
                'indicators': ['vectorize', 'optimize', 'cache', 'parallel', 'batch'],
                'description': 'Efficient and optimized - performance-focused'
            },
            'friendly': {
                'indicators': ['comment', 'docstring', 'explain', 'clear', 'readable'],
                'description': 'Friendly and clear - well-documented, easy to understand'
            },
            'minimalist': {
                'indicators': ['simple', 'minimal', 'concise', 'short'],
                'description': 'Minimalist - simple and concise code'
            },
            'perfectionist': {
                'indicators': ['precise', 'exact', 'perfect', 'accurate', 'detailed'],
                'description': 'Perfectionist - precise and detailed'
            }
        }
    
    def analyze_personality(self, code: str) -> Dict[str, Any]:
        """Analyze code personality"""
        personality_scores = {}
        code_lower = code.lower()
        
        # Score each personality trait
        for trait, data in self.personality_traits.items():
            score = 0
            for indicator in data['indicators']:
                if re.search(indicator, code_lower, re.IGNORECASE):
                    score += 1
            
            # Additional scoring based on code patterns
            if trait == 'cautious' and re.search(r'if\s+.*:\s*\n\s*else', code, re.MULTILINE):
                score += 2
            
            if trait == 'efficient' and re.search(r'for\s+.*in\s+range\(len\(', code):
                score -= 1  # Inefficient pattern
            
            if trait == 'friendly' and len(re.findall(r'#|""".*"""', code, re.DOTALL)) > 3:
                score += 2
            
            personality_scores[trait] = score
        
        # Determine primary personality
        primary_personality = max(personality_scores.items(), key=lambda x: x[1])
        
        # Get secondary personalities
        sorted_traits = sorted(personality_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_personalities = [trait for trait, score in sorted_traits[1:4] if score > 0]
        
        return {
            'primary_personality': primary_personality[0],
            'primary_score': primary_personality[1],
            'secondary_personalities': secondary_personalities,
            'all_scores': personality_scores,
            'description': self.personality_traits[primary_personality[0]]['description'],
            'suggestions': self._get_personality_suggestions(primary_personality[0], code)
        }
    
    def _get_personality_suggestions(self, personality: str, code: str) -> List[str]:
        """Get suggestions based on personality"""
        suggestions = []
        
        if personality == 'bold':
            suggestions.append("💪 Bold code! Consider adding error handling for robustness")
            suggestions.append("🚀 Try even more experimental approaches - you're adventurous!")
        
        elif personality == 'cautious':
            suggestions.append("🛡️ Very cautious! Consider simplifying some checks for readability")
            suggestions.append("⚡ Add some performance optimizations - you're safe but could be faster")
        
        elif personality == 'creative':
            suggestions.append("🎨 Creative code! Document your innovative approach")
            suggestions.append("💡 Share your creative solutions - others might benefit")
        
        elif personality == 'efficient':
            suggestions.append("⚡ Efficient code! Consider adding comments for maintainability")
            suggestions.append("📊 Your code is fast - consider sharing optimization techniques")
        
        elif personality == 'friendly':
            suggestions.append("😊 Friendly code! Consider adding performance optimizations")
            suggestions.append("📚 Well-documented - others will appreciate your clarity")
        
        elif personality == 'minimalist':
            suggestions.append("✨ Minimalist code! Consider adding error handling")
            suggestions.append("🎯 Simple and clean - consider documenting key decisions")
        
        elif personality == 'perfectionist':
            suggestions.append("🎯 Perfectionist code! Consider balancing precision with performance")
            suggestions.append("📐 Very detailed - consider if all details are necessary")
        
        return suggestions
    
    def suggest_personality_improvement(self, code: str, target_personality: str) -> Dict[str, Any]:
        """Suggest how to improve code to match target personality"""
        current = self.analyze_personality(code)
        target_traits = self.personality_traits.get(target_personality, {})
        
        improvements = []
        
        if target_personality == 'efficient' and current['primary_personality'] != 'efficient':
            improvements.append("Add vectorization: Replace loops with NumPy operations")
            improvements.append("Enable caching: Add use_cache=True to toolbox calls")
            improvements.append("Use batch processing: Process data in batches")
        
        elif target_personality == 'friendly' and current['primary_personality'] != 'friendly':
            improvements.append("Add comments: Explain complex logic")
            improvements.append("Add docstrings: Document functions and classes")
            improvements.append("Use clear variable names: Make code self-documenting")
        
        elif target_personality == 'cautious' and current['primary_personality'] != 'cautious':
            improvements.append("Add error handling: Wrap risky operations in try-except")
            improvements.append("Add validation: Check inputs before processing")
            improvements.append("Add defensive checks: Verify assumptions")
        
        return {
            'current_personality': current['primary_personality'],
            'target_personality': target_personality,
            'improvements': improvements,
            'description': target_traits.get('description', '')
        }


# Global instance
_global_personality = None

def get_code_personality() -> CodePersonality:
    """Get global code personality analyzer"""
    global _global_personality
    if _global_personality is None:
        _global_personality = CodePersonality()
    return _global_personality
