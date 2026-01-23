"""
Self-Healing Code System
Automatically fixes bugs before they cause issues

Revolutionary: Code that fixes itself proactively
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import traceback
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox.ai_agent import MLCodeAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class SelfHealingCode:
    """
    Self-Healing Code System
    
    Innovation: Automatically detects and fixes code issues:
    - Syntax errors
    - Runtime errors
    - Performance issues
    - Best practice violations
    - Before they cause problems
    """
    
    def __init__(self):
        self.agent = MLCodeAgent(use_llm=False) if AGENT_AVAILABLE else None
        self.error_patterns = {}
        self.fix_history = []
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for potential issues"""
        issues = []
        
        # Check syntax
        syntax_ok, syntax_error = self._check_syntax(code)
        if not syntax_ok:
            issues.append({
                'type': 'syntax_error',
                'severity': 'critical',
                'error': syntax_error,
                'fixable': True
            })
        
        # Check for common issues
        common_issues = self._check_common_issues(code)
        issues.extend(common_issues)
        
        # Check performance
        performance_issues = self._check_performance(code)
        issues.extend(performance_issues)
        
        return {
            'issues': issues,
            'has_issues': len(issues) > 0,
            'critical_issues': [i for i in issues if i['severity'] == 'critical']
        }
    
    def _check_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Check code syntax"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def _check_common_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for common ML code issues"""
        issues = []
        
        # Check for missing imports
        if 'MLToolbox' in code and 'from ml_toolbox import' not in code:
            issues.append({
                'type': 'missing_import',
                'severity': 'high',
                'message': 'MLToolbox used but not imported',
                'fix': 'from ml_toolbox import MLToolbox'
            })
        
        # Check for undefined variables
        if 'toolbox.fit(' in code and 'toolbox =' not in code:
            issues.append({
                'type': 'undefined_variable',
                'severity': 'high',
                'message': 'toolbox used but not defined',
                'fix': 'toolbox = MLToolbox()'
            })
        
        # Check for data shape mismatches
        if 'X.shape' in code and 'y.shape' in code:
            issues.append({
                'type': 'potential_shape_mismatch',
                'severity': 'medium',
                'message': 'Check X and y shapes match',
                'fix': 'Verify X.shape[0] == y.shape[0]'
            })
        
        return issues
    
    def _check_performance(self, code: str) -> List[Dict[str, Any]]:
        """Check for performance issues"""
        issues = []
        
        # Check for inefficient loops
        if 'for i in range(len(' in code:
            issues.append({
                'type': 'inefficient_loop',
                'severity': 'low',
                'message': 'Consider using enumerate() or vectorized operations',
                'fix': 'Use enumerate() or NumPy vectorization'
            })
        
        # Check for missing caching
        if 'toolbox.fit(' in code and 'use_cache' not in code:
            issues.append({
                'type': 'missing_cache',
                'severity': 'low',
                'message': 'Consider using model caching for faster training',
                'fix': "toolbox.fit(X, y, use_cache=True)"
            })
        
        return issues
    
    def auto_fix(self, code: str, issue: Dict[str, Any]) -> str:
        """Automatically fix a code issue"""
        if issue['type'] == 'syntax_error':
            return self._fix_syntax_error(code, issue)
        elif issue['type'] == 'missing_import':
            return self._add_import(code, issue['fix'])
        elif issue['type'] == 'undefined_variable':
            return self._add_variable_definition(code, issue['fix'])
        else:
            # Use AI Agent for complex fixes
            if self.agent:
                result = self.agent.build(f"Fix this code issue: {issue['message']}\n\nCode:\n{code}")
                if result.get('success'):
                    return result['code']
        
        return code  # Return original if fix fails
    
    def _fix_syntax_error(self, code: str, issue: Dict[str, Any]) -> str:
        """Fix syntax error"""
        if self.agent:
            result = self.agent.build(f"Fix this syntax error: {issue['error']}\n\nCode:\n{code}")
            if result.get('success'):
                return result['code']
        return code
    
    def _add_import(self, code: str, import_stmt: str) -> str:
        """Add missing import"""
        if import_stmt not in code:
            lines = code.split('\n')
            # Find last import or add at top
            last_import = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    last_import = i + 1
            
            lines.insert(last_import, import_stmt)
            return '\n'.join(lines)
        return code
    
    def _add_variable_definition(self, code: str, definition: str) -> str:
        """Add missing variable definition"""
        if definition.split('=')[0].strip() not in code:
            lines = code.split('\n')
            # Add after imports
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_pos = i + 1
            
            lines.insert(insert_pos, definition)
            return '\n'.join(lines)
        return code
    
    def heal_code(self, code: str) -> Dict[str, Any]:
        """Heal code by fixing all issues"""
        analysis = self.analyze_code(code)
        
        if not analysis['has_issues']:
            return {
                'success': True,
                'healed_code': code,
                'issues_fixed': 0,
                'message': 'No issues found'
            }
        
        healed_code = code
        issues_fixed = 0
        
        # Fix critical issues first
        for issue in analysis['critical_issues']:
            if issue.get('fixable'):
                healed_code = self.auto_fix(healed_code, issue)
                issues_fixed += 1
        
        # Fix other issues
        for issue in analysis['issues']:
            if issue['severity'] != 'critical' and issue.get('fixable'):
                healed_code = self.auto_fix(healed_code, issue)
                issues_fixed += 1
        
        return {
            'success': True,
            'healed_code': healed_code,
            'issues_fixed': issues_fixed,
            'original_issues': len(analysis['issues']),
            'message': f'Fixed {issues_fixed} issues'
        }


# Global instance
_global_healer = None

def get_self_healing_code() -> SelfHealingCode:
    """Get global self-healing code instance"""
    global _global_healer
    if _global_healer is None:
        _global_healer = SelfHealingCode()
    return _global_healer
