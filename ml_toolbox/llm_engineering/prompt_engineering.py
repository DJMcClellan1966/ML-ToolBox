"""
Prompt Engineering - Best Practices from LLM Engineer's Handbook

Implements:
- Prompt templates
- Prompt optimization
- Few-shot prompting
- Chain-of-thought prompting
- Role-based prompting
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Prompt template with variable substitution
    
    Supports:
    - Variable placeholders: {variable_name}
    - Default values
    - Type validation
    """
    
    def __init__(self, template: str, variables: Optional[Dict[str, Any]] = None):
        """
        Initialize prompt template
        
        Parameters
        ----------
        template : str
            Template string with {variable} placeholders
        variables : dict, optional
            Default variable values
        """
        self.template = template
        self.variables = variables or {}
    
    def format(self, **kwargs) -> str:
        """
        Format template with variables
        
        Parameters
        ----------
        **kwargs
            Variable values to substitute
            
        Returns
        -------
        formatted_prompt : str
            Formatted prompt string
        """
        # Merge default variables with provided ones
        merged_vars = {**self.variables, **kwargs}
        
        try:
            return self.template.format(**merged_vars)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            # Try with available variables only
            available_vars = {k: v for k, v in merged_vars.items() if f"{{{k}}}" in self.template}
            return self.template.format(**available_vars)
    
    def add_variable(self, name: str, value: Any):
        """Add or update a variable"""
        self.variables[name] = value


class PromptEngineer:
    """
    Prompt Engineering - Optimize prompts for better LLM performance
    
    Features:
    - Prompt templates
    - Prompt optimization
    - Few-shot examples
    - Chain-of-thought reasoning
    - Role-based prompting
    """
    
    def __init__(self):
        self.templates = {}
        self.few_shot_examples = {}
        self.optimization_history = []
        
        # Initialize common templates
        self._init_common_templates()
    
    def _init_common_templates(self):
        """Initialize common prompt templates"""
        
        # Classification template
        self.templates['classification'] = PromptTemplate(
            """You are an expert machine learning engineer. Your task is to classify data.

Task: {task_description}
Data: {data_info}
Target: {target_info}

Instructions:
1. Analyze the data characteristics
2. Select the best classification algorithm
3. Explain your reasoning
4. Provide the classification result

Classification:"""
        )
        
        # Regression template
        self.templates['regression'] = PromptTemplate(
            """You are an expert machine learning engineer. Your task is to predict continuous values.

Task: {task_description}
Data: {data_info}
Target: {target_info}

Instructions:
1. Analyze the data characteristics
2. Select the best regression algorithm
3. Explain your reasoning
4. Provide the prediction result

Prediction:"""
        )
        
        # Feature engineering template
        self.templates['feature_engineering'] = PromptTemplate(
            """You are an expert data scientist. Your task is to engineer features.

Data: {data_info}
Target: {target_info}

Instructions:
1. Analyze feature distributions
2. Identify missing values and outliers
3. Suggest feature transformations
4. Recommend feature selection

Feature Engineering Suggestions:"""
        )
        
        # Code generation template
        self.templates['code_generation'] = PromptTemplate(
            """You are an expert Python developer specializing in machine learning.

Task: {task_description}
Context: {context}

Instructions:
1. Generate clean, efficient Python code
2. Use best practices
3. Include error handling
4. Add comments for clarity

Code:"""
        )
    
    def create_prompt(self, task_type: str, **kwargs) -> str:
        """
        Create optimized prompt for task
        
        Parameters
        ----------
        task_type : str
            Type of task (classification, regression, etc.)
        **kwargs
            Variables for template
            
        Returns
        -------
        prompt : str
            Optimized prompt
        """
        if task_type not in self.templates:
            # Default template
            template = PromptTemplate(
                "Task: {task_description}\n\nInstructions:\n{instructions}\n\nResult:"
            )
            return template.format(
                task_description=kwargs.get('task_description', ''),
                instructions=kwargs.get('instructions', 'Complete the task')
            )
        
        template = self.templates[task_type]
        return template.format(**kwargs)
    
    def add_few_shot_examples(self, task_type: str, examples: List[Dict[str, str]]):
        """
        Add few-shot examples for a task type
        
        Parameters
        ----------
        task_type : str
            Type of task
        examples : list of dict
            Examples with 'input' and 'output' keys
        """
        self.few_shot_examples[task_type] = examples
    
    def create_few_shot_prompt(self, task_type: str, current_input: str, **kwargs) -> str:
        """
        Create few-shot prompt with examples
        
        Parameters
        ----------
        task_type : str
            Type of task
        current_input : str
            Current input to process
        **kwargs
            Additional variables
            
        Returns
        -------
        prompt : str
            Few-shot prompt
        """
        base_prompt = self.create_prompt(task_type, **kwargs)
        
        if task_type in self.few_shot_examples:
            examples = self.few_shot_examples[task_type]
            few_shot_section = "\n\nExamples:\n"
            for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
                few_shot_section += f"\nExample {i}:\n"
                few_shot_section += f"Input: {example.get('input', '')}\n"
                few_shot_section += f"Output: {example.get('output', '')}\n"
            
            return base_prompt + few_shot_section + f"\n\nCurrent Input: {current_input}\nOutput:"
        
        return base_prompt + f"\n\nInput: {current_input}\nOutput:"
    
    def create_chain_of_thought_prompt(self, task: str, reasoning_steps: List[str]) -> str:
        """
        Create chain-of-thought prompt
        
        Parameters
        ----------
        task : str
            Task description
        reasoning_steps : list of str
            Steps for reasoning
            
        Returns
        -------
        prompt : str
            Chain-of-thought prompt
        """
        prompt = f"Task: {task}\n\n"
        prompt += "Let's think step by step:\n\n"
        
        for i, step in enumerate(reasoning_steps, 1):
            prompt += f"Step {i}: {step}\n"
        
        prompt += "\nBased on the above reasoning, the answer is:"
        
        return prompt
    
    def optimize_prompt(self, original_prompt: str, task_type: str, 
                       optimization_strategy: str = 'clarity') -> str:
        """
        Optimize prompt for better performance
        
        Parameters
        ----------
        original_prompt : str
            Original prompt
        task_type : str
            Type of task
        optimization_strategy : str
            Strategy: 'clarity', 'brevity', 'detail', 'structure'
            
        Returns
        -------
        optimized_prompt : str
            Optimized prompt
        """
        if optimization_strategy == 'clarity':
            # Add clear instructions
            optimized = f"Clear instructions:\n{original_prompt}\n\nBe specific and precise."
        elif optimization_strategy == 'brevity':
            # Make more concise
            optimized = original_prompt.replace("  ", " ").strip()
        elif optimization_strategy == 'detail':
            # Add more context
            optimized = f"Detailed context:\n{original_prompt}\n\nProvide comprehensive analysis."
        elif optimization_strategy == 'structure':
            # Add structure
            optimized = f"Structured task:\n{original_prompt}\n\nFollow this structure:\n1. Understand\n2. Analyze\n3. Execute\n4. Verify"
        else:
            optimized = original_prompt
        
        # Store optimization
        self.optimization_history.append({
            'original': original_prompt,
            'optimized': optimized,
            'strategy': optimization_strategy,
            'task_type': task_type
        })
        
        return optimized
    
    def add_role(self, prompt: str, role: str) -> str:
        """
        Add role-based context to prompt
        
        Parameters
        ----------
        prompt : str
            Original prompt
        role : str
            Role (e.g., 'expert', 'beginner', 'researcher')
            
        Returns
        -------
        role_prompt : str
            Prompt with role context
        """
        role_contexts = {
            'expert': "You are an expert in this field with years of experience.",
            'beginner': "You are explaining this to someone new to the field.",
            'researcher': "You are a researcher conducting a detailed analysis.",
            'engineer': "You are a practical engineer focused on implementation.",
            'analyst': "You are a data analyst examining patterns and trends."
        }
        
        role_context = role_contexts.get(role.lower(), f"You are a {role}.")
        
        return f"{role_context}\n\n{prompt}"
