"""
Safety Guardrails - Ensure safe and appropriate LLM usage
"""
from typing import Dict, List, Optional, Any
import logging
import re

logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    Safety Guardrails
    
    Implements:
    - Content filtering
    - Prompt injection detection
    - Output validation
    - Safety checks
    """
    
    def __init__(self):
        self.blocked_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'you\s+are\s+now',
            r'new\s+instructions',
            r'system\s+prompt'
        ]
        self.sensitive_keywords = [
            'password', 'secret', 'api key', 'token',
            'credit card', 'ssn', 'social security'
        ]
        self.safety_history = []
    
    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Check prompt for safety issues
        
        Parameters
        ----------
        prompt : str
            Prompt to check
            
        Returns
        -------
        safety_check : dict
            Safety check results
        """
        issues = []
        severity = 'low'
        
        # Check for prompt injection
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"Potential prompt injection detected: {pattern}")
                severity = 'high'
        
        # Check for sensitive information
        prompt_lower = prompt.lower()
        for keyword in self.sensitive_keywords:
            if keyword in prompt_lower:
                issues.append(f"Sensitive keyword detected: {keyword}")
                if severity == 'low':
                    severity = 'medium'
        
        # Check prompt length (extremely long prompts might be suspicious)
        if len(prompt) > 10000:
            issues.append("Prompt is extremely long (potential attack)")
            if severity == 'low':
                severity = 'medium'
        
        is_safe = len(issues) == 0
        
        result = {
            'is_safe': is_safe,
            'severity': severity,
            'issues': issues,
            'recommendation': 'safe' if is_safe else 'review_required'
        }
        
        if not is_safe:
            result['recommendation'] = 'Block or sanitize prompt before processing'
        
        # Store check
        self.safety_history.append({
            'prompt_length': len(prompt),
            'is_safe': is_safe,
            'severity': severity
        })
        
        return result
    
    def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize prompt by removing suspicious content
        
        Parameters
        ----------
        prompt : str
            Prompt to sanitize
            
        Returns
        -------
        sanitized : str
            Sanitized prompt
        """
        sanitized = prompt
        
        # Remove prompt injection patterns
        for pattern in self.blocked_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def check_response(self, response: str) -> Dict[str, Any]:
        """
        Check response for safety issues
        
        Parameters
        ----------
        response : str
            Response to check
            
        Returns
        -------
        safety_check : dict
            Safety check results
        """
        issues = []
        severity = 'low'
        
        # Check for sensitive information leakage
        response_lower = response.lower()
        for keyword in self.sensitive_keywords:
            if keyword in response_lower:
                issues.append(f"Sensitive information in response: {keyword}")
                severity = 'high'
        
        # Check for harmful content
        harmful_patterns = [
            r'how\s+to\s+hack',
            r'bypass\s+security',
            r'exploit\s+vulnerability'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Potentially harmful content: {pattern}")
                severity = 'high'
        
        is_safe = len(issues) == 0
        
        return {
            'is_safe': is_safe,
            'severity': severity,
            'issues': issues,
            'recommendation': 'safe' if is_safe else 'review_response'
        }
    
    def filter_response(self, response: str) -> str:
        """
        Filter response to remove unsafe content
        
        Parameters
        ----------
        response : str
            Response to filter
            
        Returns
        -------
        filtered : str
            Filtered response
        """
        filtered = response
        
        # Remove sensitive information patterns
        for keyword in self.sensitive_keywords:
            # Replace with placeholder
            filtered = re.sub(rf'\b{keyword}\b', '[REDACTED]', filtered, flags=re.IGNORECASE)
        
        return filtered
    
    def get_safety_stats(self) -> Dict:
        """Get safety statistics"""
        if not self.safety_history:
            return {'total_checks': 0}
        
        unsafe_count = sum(1 for h in self.safety_history if not h['is_safe'])
        high_severity_count = sum(1 for h in self.safety_history if h['severity'] == 'high')
        
        return {
            'total_checks': len(self.safety_history),
            'unsafe_count': unsafe_count,
            'safe_rate': 1.0 - (unsafe_count / len(self.safety_history)),
            'high_severity_count': high_severity_count
        }
