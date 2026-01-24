"""
Few-Shot Learning - Provide examples to improve LLM performance
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FewShotLearner:
    """
    Few-Shot Learning
    
    Manages examples and creates few-shot prompts
    """
    
    def __init__(self):
        self.examples = {}  # Task type -> list of examples
        self.example_quality = {}  # Track which examples work best
    
    def add_example(self, task_type: str, input_text: str, output_text: str, 
                   quality_score: Optional[float] = None):
        """
        Add example for few-shot learning
        
        Parameters
        ----------
        task_type : str
            Type of task
        input_text : str
            Input example
        output_text : str
            Output example
        quality_score : float, optional
            Quality score (0-1)
        """
        if task_type not in self.examples:
            self.examples[task_type] = []
        
        example = {
            'input': input_text,
            'output': output_text,
            'quality_score': quality_score or 0.5
        }
        
        self.examples[task_type].append(example)
        
        # Sort by quality
        self.examples[task_type].sort(key=lambda x: x['quality_score'], reverse=True)
    
    def get_best_examples(self, task_type: str, n: int = 3) -> List[Dict[str, str]]:
        """
        Get best examples for task type
        
        Parameters
        ----------
        task_type : str
            Type of task
        n : int
            Number of examples to return
            
        Returns
        -------
        examples : list of dict
            Best examples
        """
        if task_type not in self.examples:
            return []
        
        return self.examples[task_type][:n]
    
    def create_few_shot_prompt(self, task_type: str, current_input: str, 
                              n_examples: int = 3) -> str:
        """
        Create few-shot prompt
        
        Parameters
        ----------
        task_type : str
            Type of task
        current_input : str
            Current input to process
        n_examples : int
            Number of examples to include
            
        Returns
        -------
        prompt : str
            Few-shot prompt
        """
        examples = self.get_best_examples(task_type, n=n_examples)
        
        if not examples:
            return f"Task: {current_input}\n\nOutput:"
        
        prompt = "Examples:\n\n"
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Now, for this input:\n"
        prompt += f"Input: {current_input}\n"
        prompt += f"Output:"
        
        return prompt
    
    def update_example_quality(self, task_type: str, example_index: int, 
                             quality_score: float):
        """
        Update example quality based on performance
        
        Parameters
        ----------
        task_type : str
            Type of task
        example_index : int
            Index of example
        quality_score : float
            New quality score (0-1)
        """
        if task_type in self.examples and 0 <= example_index < len(self.examples[task_type]):
            self.examples[task_type][example_index]['quality_score'] = quality_score
            # Re-sort
            self.examples[task_type].sort(key=lambda x: x['quality_score'], reverse=True)
    
    def initialize_ml_examples(self):
        """Initialize common ML examples"""
        
        # Classification example
        self.add_example(
            'classification',
            "Classify customer data into churn/no-churn",
            "I'll use a Random Forest classifier. First, I'll preprocess the data by handling missing values and encoding categorical variables. Then I'll train the model with 100 estimators and evaluate using cross-validation.",
            0.9
        )
        
        # Regression example
        self.add_example(
            'regression',
            "Predict house prices from features",
            "I'll use a Gradient Boosting Regressor. First, I'll standardize the features and handle outliers. Then I'll train with early stopping and tune hyperparameters using grid search.",
            0.9
        )
        
        # Feature engineering example
        self.add_example(
            'feature_engineering',
            "Engineer features for customer segmentation",
            "I'll create: 1) Interaction features (age * income), 2) Binning (age groups), 3) Polynomial features for non-linear relationships, 4) Feature selection using variance threshold.",
            0.85
        )
