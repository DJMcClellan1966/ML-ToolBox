"""
Compartment 3: Algorithms
Machine learning models, evaluation, tuning, and ensembles

Optimizations:
- Parallel cross-validation
- Cached model evaluation
- Big O optimizations
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import functools

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimizations
try:
    from .optimizations import cache_result, get_global_monitor, ParallelProcessor
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    print("Warning: Optimizations not available")


class AlgorithmsCompartment:
    """
    Compartment 3: Algorithms
    
    Components for machine learning algorithms:
    - ML Evaluation: Cross-validation, metrics, overfitting detection
    - Hyperparameter Tuning: Grid search, random search
    - Ensemble Learning: Voting, bagging, boosting, stacking
    - Model utilities and wrappers
    """
    
    def __init__(self, n_jobs: Optional[int] = None):
        self.components = {}
        self.n_jobs = n_jobs or -1  # -1 means use all cores
        self._monitor = get_global_monitor() if OPTIMIZATIONS_AVAILABLE else None
        self._parallel = ParallelProcessor(n_workers=None) if OPTIMIZATIONS_AVAILABLE else None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all algorithms compartment components"""
        
        # ML Evaluation
        try:
            from ml_evaluation import MLEvaluator, HyperparameterTuner, PreprocessorOptimizer
            self.components['MLEvaluator'] = MLEvaluator
            self.components['HyperparameterTuner'] = HyperparameterTuner
            self.components['PreprocessorOptimizer'] = PreprocessorOptimizer
        except ImportError as e:
            print(f"Warning: Could not import ML evaluation: {e}")
        
        # Ensemble Learning
        try:
            from ensemble_learning import EnsembleLearner, PreprocessorEnsemble
            self.components['EnsembleLearner'] = EnsembleLearner
            self.components['PreprocessorEnsemble'] = PreprocessorEnsemble
        except ImportError as e:
            print(f"Warning: Could not import ensemble learning: {e}")
        
        # Statistical Learning Methods
        try:
            from statistical_learning import (
                StatisticalEvaluator,
                StatisticalValidator,
                BayesianOptimizer,
                StatisticalFeatureSelector
            )
            self.components['StatisticalEvaluator'] = StatisticalEvaluator
            self.components['StatisticalValidator'] = StatisticalValidator
            self.components['BayesianOptimizer'] = BayesianOptimizer
            self.components['StatisticalFeatureSelector'] = StatisticalFeatureSelector
        except ImportError as e:
            print(f"Warning: Could not import statistical learning: {e}")
        
        # Add component descriptions
        self.component_descriptions = {
            'MLEvaluator': {
                'description': 'Comprehensive ML model evaluation',
                'features': [
                    'Cross-validation',
                    'Multiple metrics (accuracy, precision, recall, F1, MSE, MAE, R²)',
                    'Overfitting detection',
                    'Learning curves',
                    'Train/test splits'
                ],
                'location': 'ml_evaluation.py',
                'category': 'Evaluation'
            },
            'HyperparameterTuner': {
                'description': 'Hyperparameter optimization',
                'features': [
                    'Grid search',
                    'Random search',
                    'Model parameter tuning',
                    'Preprocessor parameter tuning'
                ],
                'location': 'ml_evaluation.py',
                'category': 'Tuning'
            },
            'EnsembleLearner': {
                'description': 'Ensemble learning methods',
                'features': [
                    'Voting ensembles',
                    'Bagging',
                    'Boosting',
                    'Stacking',
                    'Multiple algorithms'
                ],
                'location': 'ensemble_learning.py',
                'category': 'Ensemble'
            },
            'PreprocessorEnsemble': {
                'description': 'Ensemble of preprocessing strategies',
                'features': [
                    'Multiple preprocessors',
                    'Voting on preprocessing',
                    'Combined preprocessing'
                ],
                'location': 'ensemble_learning.py',
                'category': 'Ensemble'
            },
            'StatisticalEvaluator': {
                'description': 'Statistical evaluation with uncertainty quantification',
                'features': [
                    'Confidence intervals for predictions',
                    'Prediction intervals',
                    'Bootstrap-based uncertainty',
                    'Uncertainty scores'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'StatisticalValidator': {
                'description': 'Statistical validation methods',
                'features': [
                    'Permutation tests for model comparison',
                    'Bootstrap validation',
                    'Hypothesis testing',
                    'Statistical significance'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'BayesianOptimizer': {
                'description': 'Bayesian hyperparameter optimization',
                'features': [
                    'Gaussian Process-based optimization',
                    'More efficient than grid/random search',
                    'Handles uncertainty',
                    'Better exploration/exploitation'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'StatisticalFeatureSelector': {
                'description': 'Statistical feature selection methods',
                'features': [
                    'Mutual information selection',
                    'Chi-square tests',
                    'F-tests',
                    'Statistical significance'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            }
        }
    
    def get_evaluator(self):
        """
        Get an ML Evaluator instance
        
        Returns:
            MLEvaluator instance
        """
        if 'MLEvaluator' in self.components:
            return self.components['MLEvaluator']()
        else:
            raise ImportError("ML Evaluator not available")
    
    def get_tuner(self):
        """
        Get a Hyperparameter Tuner instance
        
        Returns:
            HyperparameterTuner instance
        """
        if 'HyperparameterTuner' in self.components:
            return self.components['HyperparameterTuner']()
        else:
            raise ImportError("Hyperparameter Tuner not available")
    
    def get_ensemble(self):
        """
        Get an Ensemble Learner instance
        
        Returns:
            EnsembleLearner instance
        """
        if 'EnsembleLearner' in self.components:
            return self.components['EnsembleLearner']()
        else:
            raise ImportError("Ensemble Learner not available")
    
    def get_statistical_evaluator(self, n_bootstrap: int = 1000):
        """
        Get statistical evaluator instance
        
        Args:
            n_bootstrap: Number of bootstrap samples for uncertainty quantification
            
        Returns:
            StatisticalEvaluator instance
        """
        if 'StatisticalEvaluator' in self.components:
            return self.components['StatisticalEvaluator'](n_bootstrap=n_bootstrap)
        else:
            raise ImportError("StatisticalEvaluator not available. Install scipy: pip install scipy")
    
    def get_statistical_validator(self):
        """
        Get statistical validator instance
        
        Returns:
            StatisticalValidator instance
        """
        if 'StatisticalValidator' in self.components:
            return self.components['StatisticalValidator']()
        else:
            raise ImportError("StatisticalValidator not available. Install scipy: pip install scipy")
    
    def get_bayesian_optimizer(self, n_calls: int = 50):
        """
        Get Bayesian optimizer instance
        
        Args:
            n_calls: Number of optimization iterations
            
        Returns:
            BayesianOptimizer instance
        """
        if 'BayesianOptimizer' in self.components:
            return self.components['BayesianOptimizer'](n_calls=n_calls)
        else:
            raise ImportError("BayesianOptimizer not available. Install scikit-optimize: pip install scikit-optimize")
    
    def get_statistical_feature_selector(self):
        """
        Get statistical feature selector instance
        
        Returns:
            StatisticalFeatureSelector instance
        """
        if 'StatisticalFeatureSelector' in self.components:
            return self.components['StatisticalFeatureSelector']()
        else:
            raise ImportError("StatisticalFeatureSelector not available")
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("COMPARTMENT 3: ALGORITHMS")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Algorithms Compartment',
            'description': 'ML models, evaluation, tuning, and ensembles',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
