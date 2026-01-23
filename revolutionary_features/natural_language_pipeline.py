"""
Natural Language to ML Pipeline
Convert natural language descriptions into complete ML pipelines

Revolutionary: Describe what you want in plain English, get a complete ML solution
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox.ai_agent import MLCodeAgent
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False


class NaturalLanguagePipeline:
    """
    Natural Language to ML Pipeline Converter
    
    Innovation: Converts natural language descriptions into complete ML pipelines
    
    Example:
    "Classify emails as spam or not spam using text content"
    → Complete pipeline with preprocessing, training, evaluation
    """
    
    def __init__(self):
        self.agent = MLCodeAgent(use_llm=False, use_pattern_composition=True) if TOOLBOX_AVAILABLE else None
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
        self.pipeline_templates = self._load_pipeline_templates()
    
    def _load_pipeline_templates(self) -> Dict[str, Any]:
        """Load pipeline templates"""
        return {
            'classification': {
                'steps': ['load_data', 'preprocess', 'train', 'evaluate', 'save'],
                'description': 'Classification pipeline'
            },
            'regression': {
                'steps': ['load_data', 'preprocess', 'train', 'evaluate', 'save'],
                'description': 'Regression pipeline'
            },
            'clustering': {
                'steps': ['load_data', 'preprocess', 'cluster', 'evaluate', 'save'],
                'description': 'Clustering pipeline'
            }
        }
    
    def parse_description(self, description: str) -> Dict[str, Any]:
        """Parse natural language description"""
        parsed = {
            'task_type': self._detect_task_type(description),
            'data_type': self._detect_data_type(description),
            'target': self._extract_target(description),
            'features': self._extract_features(description),
            'requirements': self._extract_requirements(description)
        }
        return parsed
    
    def _detect_task_type(self, description: str) -> str:
        """Detect task type from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['classify', 'classification', 'category', 'class']):
            return 'classification'
        elif any(word in description_lower for word in ['predict', 'regression', 'forecast', 'estimate']):
            return 'regression'
        elif any(word in description_lower for word in ['cluster', 'group', 'segment']):
            return 'clustering'
        else:
            return 'classification'  # Default
    
    def _detect_data_type(self, description: str) -> str:
        """Detect data type from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['text', 'email', 'document', 'sentence']):
            return 'text'
        elif any(word in description_lower for word in ['image', 'picture', 'photo']):
            return 'image'
        elif any(word in description_lower for word in ['numeric', 'number', 'value']):
            return 'numeric'
        else:
            return 'numeric'  # Default
    
    def _extract_target(self, description: str) -> Optional[str]:
        """Extract target variable from description"""
        # Look for patterns like "predict X", "classify as X"
        patterns = [
            r'predict\s+(\w+)',
            r'classify\s+as\s+(\w+)',
            r'target\s+is\s+(\w+)',
            r'predict\s+if\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_features(self, description: str) -> List[str]:
        """Extract feature mentions from description"""
        features = []
        
        # Look for "using X", "with X", "based on X"
        patterns = [
            r'using\s+(\w+)',
            r'with\s+(\w+)',
            r'based\s+on\s+(\w+)',
            r'from\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            features.extend(matches)
        
        return list(set(features))
    
    def _extract_requirements(self, description: str) -> List[str]:
        """Extract special requirements"""
        requirements = []
        
        description_lower = description.lower()
        
        if 'accuracy' in description_lower or 'precise' in description_lower:
            requirements.append('high_accuracy')
        
        if 'fast' in description_lower or 'quick' in description_lower:
            requirements.append('fast_training')
        
        if 'explain' in description_lower or 'interpret' in description_lower:
            requirements.append('interpretable')
        
        return requirements
    
    def build_pipeline(self, description: str) -> Dict[str, Any]:
        """Build complete ML pipeline from natural language"""
        # Parse description
        parsed = self.parse_description(description)
        
        # Generate pipeline code
        if self.agent:
            pipeline_prompt = self._create_pipeline_prompt(parsed, description)
            result = self.agent.build(pipeline_prompt)
            
            if result.get('success'):
                return {
                    'success': True,
                    'pipeline_code': result['code'],
                    'parsed_description': parsed,
                    'steps': self.pipeline_templates.get(parsed['task_type'], {}).get('steps', [])
                }
        
        # Fallback to template
        return {
            'success': True,
            'pipeline_code': self._generate_template_pipeline(parsed),
            'parsed_description': parsed,
            'steps': self.pipeline_templates.get(parsed['task_type'], {}).get('steps', [])
        }
    
    def _create_pipeline_prompt(self, parsed: Dict, description: str) -> str:
        """Create prompt for AI Agent"""
        prompt = f"Create a complete ML pipeline for: {description}\n\n"
        prompt += f"Task type: {parsed['task_type']}\n"
        prompt += f"Data type: {parsed['data_type']}\n"
        if parsed['target']:
            prompt += f"Target: {parsed['target']}\n"
        if parsed['features']:
            prompt += f"Features: {', '.join(parsed['features'])}\n"
        if parsed['requirements']:
            prompt += f"Requirements: {', '.join(parsed['requirements'])}\n"
        prompt += "\nGenerate complete pipeline code with all steps."
        
        return prompt
    
    def _generate_template_pipeline(self, parsed: Dict) -> str:
        """Generate template pipeline code"""
        task_type = parsed['task_type']
        
        template = f"""from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Load your data here
# X = your_features
# y = your_labels

# Preprocess data
# X_processed = toolbox.data.preprocess(X)

# Train model
result = toolbox.fit(X, y, task_type='{task_type}')

# Evaluate model
print(f"Model trained! Accuracy: {{result.get('accuracy', 0):.2%}}")

# Save model (optional)
# toolbox.register_model(result['model'], 'my_model')
"""
        return template
    
    def execute_pipeline(self, description: str) -> Dict[str, Any]:
        """Build and execute pipeline from natural language"""
        pipeline_result = self.build_pipeline(description)
        
        if not pipeline_result['success']:
            return pipeline_result
        
        # Execute pipeline code
        try:
            exec(pipeline_result['pipeline_code'], {'MLToolbox': self.toolbox.__class__, 'toolbox': self.toolbox})
            pipeline_result['execution_success'] = True
        except Exception as e:
            pipeline_result['execution_success'] = False
            pipeline_result['execution_error'] = str(e)
        
        return pipeline_result


# Global instance
_global_nlp_pipeline = None

def get_natural_language_pipeline() -> NaturalLanguagePipeline:
    """Get global natural language pipeline instance"""
    global _global_nlp_pipeline
    if _global_nlp_pipeline is None:
        _global_nlp_pipeline = NaturalLanguagePipeline()
    return _global_nlp_pipeline
