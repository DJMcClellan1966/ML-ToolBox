"""
Third Eye Feature - Code Oracle
Looks into the future of code to predict outcomes, suggest improvements, and discover alternative uses

Revolutionary: Sees what code will do before it runs, predicts success/failure, suggests better paths
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ast
import re
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox.ai_agent import MLCodeAgent
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False


class ThirdEye:
    """
    Third Eye - Code Oracle
    
    Innovation: Looks into the future of code to:
    - Predict if code will work or fail
    - See where code is headed
    - Suggest better directions
    - Discover alternative uses
    - Warn about potential issues
    """
    
    def __init__(self):
        self.agent = MLCodeAgent(use_llm=False, use_pattern_composition=True) if TOOLBOX_AVAILABLE else None
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
        self.code_patterns = self._load_code_patterns()
        self.outcome_history = defaultdict(list)
        self.alternative_uses = defaultdict(list)
    
    def _load_code_patterns(self) -> Dict[str, Any]:
        """Load patterns that predict outcomes"""
        return {
            'success_patterns': [
                r'toolbox\.fit\([^)]+\)',  # Proper toolbox usage
                r'from ml_toolbox import',  # Correct imports
                r'X\.shape\[0\] == y\.shape\[0\]',  # Shape checking
                r'use_cache=True',  # Performance optimization
            ],
            'failure_patterns': [
                r'toolbox\.fit\(\)\s*$',  # Missing arguments
                r'X\s*=\s*None',  # None data
                r'y\s*=\s*None',  # None labels
                r'for\s+\w+\s+in\s+range\(len\(X\)\):',  # Inefficient loop
            ],
            'warning_patterns': [
                r'X\.shape\[0\]\s*<\s*10',  # Too little data
                r'X\.shape\[1\]\s*>\s*1000',  # Too many features
                r'task_type\s*=\s*[\'"]auto[\'"]',  # Auto-detection might fail
            ]
        }
    
    def look_into_future(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Look into the future of code execution
        
        Predicts:
        - Will it work?
        - Will it fail?
        - What issues might occur?
        - What's the code heading towards?
        """
        predictions = {
            'will_work': False,
            'will_fail': False,
            'confidence': 0.0,
            'predicted_outcome': 'unknown',
            'issues': [],
            'warnings': [],
            'direction': None,
            'suggestions': [],
            'alternative_uses': []
        }
        
        # Analyze code structure
        structure_analysis = self._analyze_structure(code)
        
        # Check patterns
        pattern_analysis = self._check_patterns(code)
        
        # Predict outcome
        outcome_prediction = self._predict_outcome(code, structure_analysis, pattern_analysis)
        
        # Determine direction
        direction = self._determine_direction(code, context)
        
        # Find issues
        issues = self._find_issues(code, structure_analysis)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(code, outcome_prediction, issues)
        
        # Discover alternative uses
        alternative_uses = self._discover_alternative_uses(code, context)
        
        # Combine predictions
        predictions.update({
            'will_work': outcome_prediction['will_work'],
            'will_fail': outcome_prediction['will_fail'],
            'confidence': outcome_prediction['confidence'],
            'predicted_outcome': outcome_prediction['outcome'],
            'issues': issues,
            'warnings': pattern_analysis.get('warnings', []),
            'direction': direction,
            'suggestions': suggestions,
            'alternative_uses': alternative_uses,
            'structure_analysis': structure_analysis,
            'pattern_analysis': pattern_analysis
        })
        
        return predictions
    
    def _analyze_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure"""
        analysis = {
            'has_imports': False,
            'has_toolbox_init': False,
            'has_data': False,
            'has_training': False,
            'has_evaluation': False,
            'syntax_valid': False,
            'complexity': 'low'
        }
        
        # Check syntax
        try:
            ast.parse(code)
            analysis['syntax_valid'] = True
        except SyntaxError:
            analysis['syntax_valid'] = False
            return analysis
        
        # Check for imports
        if re.search(r'from ml_toolbox|import.*MLToolbox', code):
            analysis['has_imports'] = True
        
        # Check for toolbox initialization
        if re.search(r'toolbox\s*=\s*MLToolbox\(\)', code):
            analysis['has_toolbox_init'] = True
        
        # Check for data
        if re.search(r'X\s*=|y\s*=', code):
            analysis['has_data'] = True
        
        # Check for training
        if re.search(r'\.fit\(|toolbox\.fit', code):
            analysis['has_training'] = True
        
        # Check for evaluation
        if re.search(r'\.predict|\.evaluate|accuracy|score', code):
            analysis['has_evaluation'] = True
        
        # Estimate complexity
        lines = len(code.split('\n'))
        if lines > 50:
            analysis['complexity'] = 'high'
        elif lines > 20:
            analysis['complexity'] = 'medium'
        
        return analysis
    
    def _check_patterns(self, code: str) -> Dict[str, Any]:
        """Check code against success/failure patterns"""
        results = {
            'success_matches': [],
            'failure_matches': [],
            'warnings': []
        }
        
        # Check success patterns
        for pattern in self.code_patterns['success_patterns']:
            if re.search(pattern, code, re.IGNORECASE):
                results['success_matches'].append(pattern)
        
        # Check failure patterns
        for pattern in self.code_patterns['failure_patterns']:
            if re.search(pattern, code, re.IGNORECASE):
                results['failure_matches'].append(pattern)
        
        # Check warning patterns
        for pattern in self.code_patterns['warning_patterns']:
            if re.search(pattern, code, re.IGNORECASE):
                warning_msg = self._pattern_to_warning(pattern)
                results['warnings'].append(warning_msg)
        
        return results
    
    def _pattern_to_warning(self, pattern: str) -> str:
        """Convert pattern to warning message"""
        warnings = {
            r'X\.shape\[0\]\s*<\s*10': 'Very small dataset (< 10 samples) - may cause overfitting',
            r'X\.shape\[1\]\s*>\s*1000': 'Very high dimensionality (> 1000 features) - consider feature selection',
            r'task_type\s*=\s*[\'"]auto[\'"]': 'Auto task detection may fail - specify task_type explicitly'
        }
        return warnings.get(pattern, 'Potential issue detected')
    
    def _predict_outcome(self, code: str, structure: Dict, patterns: Dict) -> Dict[str, Any]:
        """Predict if code will work or fail"""
        will_work = False
        will_fail = False
        confidence = 0.0
        outcome = 'unknown'
        
        # Check syntax first
        if not structure['syntax_valid']:
            return {
                'will_work': False,
                'will_fail': True,
                'confidence': 0.95,
                'outcome': 'syntax_error'
            }
        
        # Check for required components
        has_required = (
            structure['has_imports'] or 
            structure['has_toolbox_init'] or
            structure['has_data']
        )
        
        if not has_required:
            return {
                'will_work': False,
                'will_fail': True,
                'confidence': 0.8,
                'outcome': 'missing_components'
            }
        
        # Count success vs failure indicators
        success_score = len(patterns['success_matches'])
        failure_score = len(patterns['failure_matches'])
        
        if failure_score > success_score:
            will_fail = True
            confidence = min(0.9, 0.5 + failure_score * 0.1)
            outcome = 'likely_failure'
        elif success_score > failure_score:
            will_work = True
            confidence = min(0.9, 0.5 + success_score * 0.1)
            outcome = 'likely_success'
        else:
            # Balanced - check structure
            if structure['has_training'] and structure['has_data']:
                will_work = True
                confidence = 0.6
                outcome = 'possible_success'
            else:
                will_fail = True
                confidence = 0.6
                outcome = 'possible_failure'
        
        return {
            'will_work': will_work,
            'will_fail': will_fail,
            'confidence': confidence,
            'outcome': outcome
        }
    
    def _determine_direction(self, code: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Determine where the code is heading"""
        direction = {
            'intended_use': 'unknown',
            'likely_outcome': 'unknown',
            'path': [],
            'destination': 'unknown'
        }
        
        # Detect intended use
        if re.search(r'classif|classify', code, re.IGNORECASE):
            direction['intended_use'] = 'classification'
        elif re.search(r'regress|predict.*price|forecast', code, re.IGNORECASE):
            direction['intended_use'] = 'regression'
        elif re.search(r'cluster|group|segment', code, re.IGNORECASE):
            direction['intended_use'] = 'clustering'
        elif re.search(r'preprocess|clean|prepare', code, re.IGNORECASE):
            direction['intended_use'] = 'preprocessing'
        elif re.search(r'evaluate|test|score', code, re.IGNORECASE):
            direction['intended_use'] = 'evaluation'
        
        # Determine path
        path = []
        if re.search(r'import|from', code):
            path.append('setup')
        if re.search(r'X\s*=|y\s*=', code):
            path.append('data_preparation')
        if re.search(r'preprocess|clean', code):
            path.append('preprocessing')
        if re.search(r'\.fit\(|train', code):
            path.append('training')
        if re.search(r'\.predict|evaluate', code):
            path.append('prediction')
        
        direction['path'] = path
        direction['destination'] = path[-1] if path else 'unknown'
        
        # Predict likely outcome
        if 'training' in path:
            direction['likely_outcome'] = 'model_trained'
        elif 'prediction' in path:
            direction['likely_outcome'] = 'predictions_made'
        elif 'preprocessing' in path:
            direction['likely_outcome'] = 'data_preprocessed'
        
        return direction
    
    def _find_issues(self, code: str, structure: Dict) -> List[Dict[str, Any]]:
        """Find potential issues"""
        issues = []
        
        # Missing imports
        if not structure['has_imports'] and 'MLToolbox' in code:
            issues.append({
                'type': 'missing_import',
                'severity': 'high',
                'message': 'MLToolbox used but not imported',
                'fix': 'Add: from ml_toolbox import MLToolbox'
            })
        
        # Missing initialization
        if not structure['has_toolbox_init'] and 'toolbox.' in code:
            issues.append({
                'type': 'missing_init',
                'severity': 'high',
                'message': 'toolbox used but not initialized',
                'fix': 'Add: toolbox = MLToolbox()'
            })
        
        # Missing data
        if structure['has_training'] and not structure['has_data']:
            issues.append({
                'type': 'missing_data',
                'severity': 'high',
                'message': 'Training code found but no data defined',
                'fix': 'Define X and y before training'
            })
        
        # Inefficient patterns
        if re.search(r'for\s+\w+\s+in\s+range\(len\(', code):
            issues.append({
                'type': 'inefficient_loop',
                'severity': 'medium',
                'message': 'Inefficient loop pattern detected',
                'fix': 'Consider using NumPy vectorization'
            })
        
        return issues
    
    def _generate_suggestions(self, code: str, outcome: Dict, issues: List[Dict]) -> List[str]:
        """Generate suggestions for better direction"""
        suggestions = []
        
        # If likely to fail, suggest fixes
        if outcome['will_fail']:
            if issues:
                for issue in issues[:3]:  # Top 3 issues
                    suggestions.append(f"Fix: {issue['message']} - {issue.get('fix', '')}")
        
        # Suggest improvements
        if 'toolbox.fit(' in code and 'use_cache' not in code:
            suggestions.append("Add use_cache=True for faster training on repeated data")
        
        if 'toolbox.fit(' in code and 'task_type' not in code:
            suggestions.append("Specify task_type explicitly for better results")
        
        # Suggest next steps
        if 'toolbox.fit(' in code and 'evaluate' not in code.lower():
            suggestions.append("Consider evaluating the model after training")
        
        if 'toolbox.fit(' in code and 'save' not in code.lower():
            suggestions.append("Consider saving the model for later use")
        
        # Suggest alternative approaches
        if 'for' in code and 'range(len(' in code:
            suggestions.append("Consider using NumPy vectorization for better performance")
        
        return suggestions
    
    def _discover_alternative_uses(self, code: str, context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Discover alternative uses for the code"""
        alternatives = []
        
        # If code is for classification, suggest regression
        if 'classification' in code.lower() or 'classify' in code.lower():
            alternatives.append({
                'alternative_use': 'regression',
                'description': 'This code structure could be adapted for regression tasks',
                'modification': 'Change task_type to "regression" and use continuous target'
            })
        
        # If code is for single model, suggest ensemble
        if 'toolbox.fit(' in code and 'ensemble' not in code.lower():
            alternatives.append({
                'alternative_use': 'ensemble',
                'description': 'Consider using ensemble methods for better performance',
                'modification': 'Use toolbox.ai_orchestrator.build_optimal_model() with ensemble=True'
            })
        
        # If code is for training, suggest preprocessing
        if 'toolbox.fit(' in code and 'preprocess' not in code.lower():
            alternatives.append({
                'alternative_use': 'preprocessing',
                'description': 'Add preprocessing step for better model performance',
                'modification': 'Use toolbox.universal_preprocessor.preprocess() before training'
            })
        
        # If code is basic, suggest advanced features
        if 'toolbox.fit(' in code and 'ai_orchestrator' not in code.lower():
            alternatives.append({
                'alternative_use': 'ai_orchestration',
                'description': 'Use AI Orchestrator for automatic model selection and tuning',
                'modification': 'Use toolbox.ai_orchestrator.build_optimal_model() instead'
            })
        
        return alternatives
    
    def see_different_use(self, code: str, intended_use: str) -> Dict[str, Any]:
        """
        See a different use than intended
        
        Analyzes code and suggests alternative applications
        """
        analysis = {
            'intended_use': intended_use,
            'alternative_uses': [],
            'modifications': [],
            'benefits': []
        }
        
        # Analyze code for alternative uses
        if 'classification' in intended_use.lower():
            # Could be used for anomaly detection
            analysis['alternative_uses'].append({
                'use': 'anomaly_detection',
                'description': 'Classification model can detect anomalies',
                'modification': 'Use one-class classification or threshold-based detection'
            })
            
            # Could be used for ranking
            analysis['alternative_uses'].append({
                'use': 'ranking',
                'description': 'Classification probabilities can be used for ranking',
                'modification': 'Use predict_proba() to get scores for ranking'
            })
        
        if 'regression' in intended_use.lower():
            # Could be used for classification with threshold
            analysis['alternative_uses'].append({
                'use': 'binary_classification',
                'description': 'Regression can be converted to binary classification',
                'modification': 'Apply threshold to continuous predictions'
            })
        
        # General alternatives
        analysis['alternative_uses'].append({
            'use': 'feature_importance',
            'description': 'Any trained model can provide feature importance',
            'modification': 'Extract feature_importances_ or use SHAP values'
        })
        
        analysis['alternative_uses'].append({
            'use': 'transfer_learning',
            'description': 'Trained model can be used as base for transfer learning',
            'modification': 'Use model as starting point for similar tasks'
        })
        
        return analysis
    
    def predict_future_issues(self, code: str, execution_steps: int = 5) -> List[Dict[str, Any]]:
        """Predict issues that might occur during execution"""
        future_issues = []
        
        # Check for data issues
        if 'X =' in code and 'y =' in code:
            # Predict shape mismatch
            future_issues.append({
                'step': 'data_loading',
                'issue': 'Shape mismatch between X and y',
                'probability': 0.3,
                'prevention': 'Verify X.shape[0] == y.shape[0] before training'
            })
        
        # Check for model issues
        if 'toolbox.fit(' in code:
            # Predict convergence issues
            future_issues.append({
                'step': 'training',
                'issue': 'Model may not converge',
                'probability': 0.2,
                'prevention': 'Check data quality and consider scaling'
            })
            
            # Predict memory issues
            if 'X.shape[1]' in code or 'features' in code.lower():
                future_issues.append({
                    'step': 'training',
                    'issue': 'Memory issues with large feature space',
                    'probability': 0.15,
                    'prevention': 'Use feature selection or batch processing'
                })
        
        # Check for prediction issues
        if '.predict(' in code:
            future_issues.append({
                'step': 'prediction',
                'issue': 'Shape mismatch in prediction data',
                'probability': 0.25,
                'prevention': 'Ensure prediction data has same features as training'
            })
        
        return future_issues


# Global instance
_global_third_eye = None

def get_third_eye() -> ThirdEye:
    """Get global Third Eye instance"""
    global _global_third_eye
    if _global_third_eye is None:
        _global_third_eye = ThirdEye()
    return _global_third_eye
