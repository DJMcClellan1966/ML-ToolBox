"""
Andrew Ng Machine Learning Technical Strategy
Implementing systematic approaches from Andrew Ng's ML methodology

Key Strategies:
1. Systematic Error Analysis
2. Bias/Variance Diagnosis
3. Learning Curves Analysis
4. Performance Optimization
5. Feature Importance Analysis
6. Ablation Studies
7. Model Debugging Framework
8. Systematic Model Selection
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import required libraries
try:
    from sklearn.model_selection import (
        learning_curve, validation_curve, train_test_split,
        cross_val_score, KFold, StratifiedKFold
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        mean_squared_error, mean_absolute_error, r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"Warning: sklearn not available: {e}")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # matplotlib is optional, don't print warning


class ErrorAnalyzer:
    """
    Systematic Error Analysis - Core of Andrew Ng's ML Strategy
    
    Analyze model errors to identify:
    - Most common error types
    - Features causing errors
    - Data quality issues
    - Model limitations
    """
    
    def __init__(self):
        self.error_logs = []
        self.error_patterns = defaultdict(list)
    
    def analyze_classification_errors(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Systematic error analysis for classification
        
        Returns detailed breakdown of errors:
        - Confusion matrix
        - Error types by class
        - Feature contributions
        - Misclassification patterns
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Identify error patterns
        errors = []
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                error = {
                    'index': i,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'features': X[i].tolist() if feature_names else None
                }
                errors.append(error)
                self.error_logs.append(error)
                
                # Track error patterns
                error_type = f"{true_label}->{pred_label}"
                self.error_patterns[error_type].append(i)
        
        # Feature importance in errors (if tree-based model)
        feature_importance_in_errors = None
        if hasattr(model, 'feature_importances_'):
            feature_importance_in_errors = model.feature_importances_.tolist()
        
        return {
            'confusion_matrix': cm.tolist(),
            'error_count': len(errors),
            'error_rate': len(errors) / len(y_true),
            'errors_by_class': self._errors_by_class(errors, y_true, class_names),
            'common_error_patterns': dict(self.error_patterns),
            'precision_by_class': precision.tolist(),
            'recall_by_class': recall.tolist(),
            'f1_by_class': f1.tolist(),
            'feature_importance': feature_importance_in_errors,
            'feature_names': feature_names,
            'recommendations': self._generate_recommendations(errors, cm, precision, recall)
        }
    
    def _errors_by_class(
        self,
        errors: List[Dict],
        y_true: np.ndarray,
        class_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Categorize errors by true class"""
        errors_by_class = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        
        for error in errors:
            true_label = error['true_label']
            pred_label = error['pred_label']
            
            class_name = class_names[true_label] if class_names else str(true_label)
            errors_by_class[class_name] += 1
            
            # False positives: predicted as this class but wasn't
            pred_class_name = class_names[pred_label] if class_names else str(pred_label)
            false_positives[pred_class_name] += 1
            
            # False negatives: should be this class but wasn't predicted
            false_negatives[class_name] += 1
        
        return {
            'errors_per_class': dict(errors_by_class),
            'false_positives': dict(false_positives),
            'false_negatives': dict(false_negatives)
        }
    
    def _generate_recommendations(
        self,
        errors: List[Dict],
        confusion_matrix: np.ndarray,
        precision: np.ndarray,
        recall: np.ndarray
    ) -> List[str]:
        """Generate actionable recommendations based on error analysis"""
        recommendations = []
        
        if len(errors) > len(errors) * 0.1:  # More than 10% error rate
            recommendations.append("High error rate - consider collecting more data or improving features")
        
        # Check for class imbalance issues
        if np.std(precision) > 0.2:
            recommendations.append("Large precision variance - possible class imbalance, try class weighting")
        
        if np.std(recall) > 0.2:
            recommendations.append("Large recall variance - some classes poorly learned, add more examples")
        
        # Check for systematic errors
        if len(self.error_patterns) < len(errors) * 0.5:
            recommendations.append("Systematic error patterns detected - model may need different features")
        
        # Check precision vs recall tradeoff
        low_precision_classes = np.where(precision < 0.5)[0]
        low_recall_classes = np.where(recall < 0.5)[0]
        
        if len(low_precision_classes) > 0:
            recommendations.append(f"Low precision for classes {low_precision_classes.tolist()} - too many false positives")
        
        if len(low_recall_classes) > 0:
            recommendations.append(f"Low recall for classes {low_recall_classes.tolist()} - too many false negatives")
        
        return recommendations


class BiasVarianceDiagnostic:
    """
    Bias/Variance Diagnosis - Andrew Ng's Framework
    
    Diagnose model issues:
    - High Bias (underfitting): Model too simple
    - High Variance (overfitting): Model too complex
    - Just Right: Good balance
    """
    
    def diagnose(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Diagnose bias and variance
        
        Returns:
        - Training performance
        - Validation performance
        - Gap analysis
        - Diagnosis (high bias, high variance, or good)
        - Recommendations
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        if task_type == 'classification':
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_error = 1 - train_score
            val_error = 1 - val_score
            
            # Get detailed metrics
            train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
        else:  # regression
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_error = mean_squared_error(y_train, train_pred)
            val_error = mean_squared_error(y_val, val_pred)
        
        # Calculate gap
        gap = abs(train_score - val_score) if task_type == 'classification' else abs(train_error - val_error)
        
        # Diagnose
        diagnosis = self._diagnose_bias_variance(
            train_error if task_type == 'classification' else train_error,
            val_error if task_type == 'classification' else val_error,
            gap,
            task_type
        )
        
        return {
            'train_score': float(train_score),
            'val_score': float(val_score),
            'train_error': float(train_error),
            'val_error': float(val_error),
            'gap': float(gap),
            'diagnosis': diagnosis['type'],
            'explanation': diagnosis['explanation'],
            'recommendations': diagnosis['recommendations'],
            'task_type': task_type
        }
    
    def _diagnose_bias_variance(
        self,
        train_error: float,
        val_error: float,
        gap: float,
        task_type: str
    ) -> Dict[str, Any]:
        """Diagnose bias vs variance"""
        
        # Thresholds (Andrew Ng's framework)
        HIGH_BIAS_THRESHOLD = 0.15 if task_type == 'classification' else 100.0
        HIGH_VARIANCE_THRESHOLD = 0.10 if task_type == 'classification' else 50.0
        
        if train_error > HIGH_BIAS_THRESHOLD:
            # High bias (underfitting)
            return {
                'type': 'high_bias',
                'explanation': 'Model is too simple - training error is high. Model cannot fit the data well.',
                'recommendations': [
                    'Try a more complex model (more features, higher polynomial degree, deeper network)',
                    'Add more features',
                    'Reduce regularization',
                    'Train longer (if neural network)'
                ]
            }
        elif gap > HIGH_VARIANCE_THRESHOLD:
            # High variance (overfitting)
            return {
                'type': 'high_variance',
                'explanation': 'Model is too complex - large gap between train and validation error. Model memorizes training data.',
                'recommendations': [
                    'Get more training data',
                    'Add regularization (L1, L2, dropout)',
                    'Reduce model complexity',
                    'Feature selection to reduce features',
                    'Early stopping'
                ]
            }
        else:
            # Good balance
            return {
                'type': 'good',
                'explanation': 'Good balance between bias and variance. Model generalizes well.',
                'recommendations': [
                    'Model is performing well',
                    'Consider ensemble methods for further improvement',
                    'Fine-tune hyperparameters for marginal gains'
                ]
            }


class LearningCurvesAnalyzer:
    """
    Learning Curves Analysis - Andrew Ng's Approach
    
    Analyze how model performance changes with training set size
    """
    
    def analyze(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: Optional[str] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Generate learning curves
        
        Returns:
        - Training scores at different sizes
        - Validation scores at different sizes
        - Analysis of learning curve shape
        - Recommendations
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        if scoring is None:
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # Generate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        # Analyze curve shape
        analysis = self._analyze_curve_shape(
            train_sizes_abs,
            train_scores_mean,
            val_scores_mean,
            task_type
        )
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'val_scores_mean': val_scores_mean.tolist(),
            'val_scores_std': val_scores_std.tolist(),
            'analysis': analysis,
            'task_type': task_type
        }
    
    def _analyze_curve_shape(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Analyze learning curve shape for insights"""
        # Calculate trends
        train_trend = train_scores[-1] - train_scores[0]
        val_trend = val_scores[-1] - val_scores[0]
        final_gap = train_scores[-1] - val_scores[-1]
        
        # Determine if more data would help
        val_slope = (val_scores[-1] - val_scores[-3]) / 2 if len(val_scores) >= 3 else 0
        
        would_more_data_help = val_slope > 0.01 if task_type == 'classification' else val_slope > -1.0
        
        analysis = {
            'train_trend': float(train_trend),
            'val_trend': float(val_trend),
            'final_gap': float(final_gap),
            'val_slope': float(val_slope),
            'would_more_data_help': would_more_data_help,
            'recommendations': []
        }
        
        # Generate recommendations
        if would_more_data_help:
            analysis['recommendations'].append('More training data would likely improve performance')
        else:
            analysis['recommendations'].append('More training data may not help significantly - focus on features/model')
        
        if final_gap > 0.1:
            analysis['recommendations'].append('Large gap indicates overfitting - consider regularization')
        
        if val_trend < 0:
            analysis['recommendations'].append('Validation score decreasing - model may be overfitting as data increases')
        
        return analysis


class ModelDebugger:
    """
    Model Debugging Framework - Andrew Ng's Systematic Approach
    
    Systematic debugging checklist:
    1. Data quality
    2. Feature engineering
    3. Model selection
    4. Hyperparameters
    5. Training process
    """
    
    def debug(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Systematic model debugging
        
        Returns comprehensive debugging report
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        debug_report = {
            'data_quality': self._check_data_quality(X_train, y_train, X_val, y_val),
            'feature_analysis': self._analyze_features(X_train, X_val),
            'model_performance': self._check_model_performance(model, X_train, y_train, X_val, y_val, task_type),
            'training_issues': self._check_training_issues(model, X_train, y_train, task_type),
            'recommendations': []
        }
        
        # Aggregate recommendations
        for section in debug_report.values():
            if isinstance(section, dict) and 'recommendations' in section:
                debug_report['recommendations'].extend(section['recommendations'])
        
        return debug_report
    
    def _check_data_quality(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Check data quality issues"""
        issues = []
        recommendations = []
        
        # Check for NaN values
        if np.isnan(X_train).any():
            issues.append('NaN values in training data')
            recommendations.append('Handle missing values (imputation, removal)')
        
        if np.isnan(X_val).any():
            issues.append('NaN values in validation data')
        
        # Check for infinite values
        if np.isinf(X_train).any():
            issues.append('Infinite values in training data')
            recommendations.append('Handle infinite values')
        
        # Check for class imbalance (classification)
        if len(np.unique(y_train)) < 10:  # Likely classification
            class_counts = np.bincount(y_train.astype(int))
            if np.std(class_counts) > np.mean(class_counts) * 0.5:
                issues.append('Class imbalance detected')
                recommendations.append('Use class weighting or resampling techniques')
        
        # Check data size
        if len(X_train) < 100:
            issues.append('Small training set')
            recommendations.append('Collect more training data')
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'feature_count': X_train.shape[1]
        }
    
    def _analyze_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature issues"""
        issues = []
        recommendations = []
        
        # Check for constant features
        train_std = np.std(X_train, axis=0)
        constant_features = np.where(train_std < 1e-6)[0]
        if len(constant_features) > 0:
            issues.append(f'Constant features detected: {constant_features.tolist()}')
            recommendations.append('Remove constant features')
        
        # Check for feature scale differences
        train_mean = np.mean(np.abs(X_train), axis=0)
        if np.max(train_mean) / (np.min(train_mean) + 1e-8) > 100:
            issues.append('Large feature scale differences')
            recommendations.append('Consider feature scaling (StandardScaler, MinMaxScaler)')
        
        # Check for correlation with target (if available)
        # This would require y_train, but keeping it simple for now
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'constant_features': constant_features.tolist() if len(constant_features) > 0 else [],
            'feature_variance': train_std.tolist()
        }
    
    def _check_model_performance(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Check model performance issues"""
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        gap = abs(train_score - val_score)
        
        issues = []
        recommendations = []
        
        if train_score < 0.7 and task_type == 'classification':
            issues.append('Low training performance')
            recommendations.append('Model may be too simple - increase complexity')
        
        if gap > 0.15:
            issues.append('Large train/validation gap')
            recommendations.append('Overfitting detected - add regularization or get more data')
        
        return {
            'train_score': float(train_score),
            'val_score': float(val_score),
            'gap': float(gap),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_training_issues(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Check training process issues"""
        issues = []
        recommendations = []
        
        # Check if model supports partial_fit (online learning)
        if hasattr(model, 'partial_fit'):
            recommendations.append('Model supports online learning - can train incrementally')
        
        # Check for convergence issues (for iterative models)
        if hasattr(model, 'n_iter_'):
            if model.n_iter_ == model.max_iter:
                issues.append('Model reached max iterations - may not have converged')
                recommendations.append('Increase max_iter or adjust learning rate')
        
        return {
            'issues': issues,
            'recommendations': recommendations
        }


class SystematicModelSelector:
    """
    Systematic Model Selection - Andrew Ng's Approach
    
    Select best model systematically:
    1. Try simple models first
    2. Evaluate with proper validation
    3. Compare systematically
    4. Consider bias/variance tradeoff
    """
    
    def __init__(self):
        self.comparison_results = []
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Systematically compare multiple models
        
        Returns:
        - Performance comparison
        - Best model
        - Recommendations
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if scoring is None:
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        results = []
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            # Train on full data for final evaluation
            model.fit(X, y)
            
            result = {
                'model_name': name,
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist(),
                'model': model
            }
            results.append(result)
            self.comparison_results.append(result)
        
        # Find best model
        best_idx = np.argmax([r['cv_mean'] for r in results])
        best_result = results[best_idx]
        
        # Generate recommendations
        recommendations = self._generate_model_recommendations(results, task_type)
        
        return {
            'comparison': results,
            'best_model': {
                'name': best_result['model_name'],
                'cv_mean': best_result['cv_mean'],
                'cv_std': best_result['cv_std']
            },
            'recommendations': recommendations,
            'task_type': task_type
        }
    
    def _generate_model_recommendations(
        self,
        results: List[Dict],
        task_type: str
    ) -> List[str]:
        """Generate recommendations based on model comparison"""
        recommendations = []
        
        if len(results) == 0:
            return recommendations
        
        # Check for overfitting (high variance in CV scores)
        for result in results:
            if result['cv_std'] > 0.1:
                recommendations.append(f"{result['model_name']} has high variance - consider regularization")
        
        # Check if simple models perform well
        simple_model_names = ['linear', 'logistic', 'naive']
        simple_results = [r for r in results if any(name in r['model_name'].lower() for name in simple_model_names)]
        complex_results = [r for r in results if r not in simple_results]
        
        if simple_results and complex_results:
            simple_best = max(simple_results, key=lambda x: x['cv_mean'])
            complex_best = max(complex_results, key=lambda x: x['cv_mean'])
            
            if simple_best['cv_mean'] > complex_best['cv_mean'] * 0.9:
                recommendations.append('Simple models perform nearly as well - prefer simplicity for interpretability')
        
        return recommendations


class AndrewNgMLStrategy:
    """
    Complete Andrew Ng ML Technical Strategy
    
    Combines all components for systematic ML development
    """
    
    def __init__(self):
        self.error_analyzer = ErrorAnalyzer()
        self.bias_variance = BiasVarianceDiagnostic()
        self.learning_curves = LearningCurvesAnalyzer()
        self.model_debugger = ModelDebugger()
        self.model_selector = SystematicModelSelector()
    
    def complete_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Complete systematic analysis using Andrew Ng's framework
        
        Returns comprehensive analysis report
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        print("[Andrew Ng ML Strategy] Running complete analysis...")
        
        # Initialize results dictionary
        results = {}
        
        # 1. Bias/Variance Diagnosis
        try:
            print("  [1/5] Bias/Variance Diagnosis...")
            bias_variance = self.bias_variance.diagnose(model, X_train, y_train, X_val, y_val, task_type)
            if 'error' not in bias_variance:
                results['bias_variance_diagnosis'] = bias_variance
            else:
                results['bias_variance_diagnosis'] = {'error': bias_variance.get('error', 'Unknown error')}
        except Exception as e:
            print(f"    Warning: Bias/Variance diagnosis failed: {e}")
            results['bias_variance_diagnosis'] = {'error': str(e)}
        
        # 2. Learning Curves
        try:
            print("  [2/5] Learning Curves Analysis...")
            X_all = np.vstack([X_train, X_val])
            y_all = np.hstack([y_train, y_val])
            learning_curves = self.learning_curves.analyze(model, X_all, y_all, task_type=task_type)
            if 'error' not in learning_curves:
                results['learning_curves'] = learning_curves
            else:
                results['learning_curves'] = {'error': learning_curves.get('error', 'Unknown error')}
        except Exception as e:
            print(f"    Warning: Learning curves analysis failed: {e}")
            results['learning_curves'] = {'error': str(e)}
        
        # 3. Error Analysis
        try:
            print("  [3/5] Error Analysis...")
            # Create fresh model instance for error analysis
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                model_for_analysis = type(model)(**model_params)
            else:
                model_for_analysis = type(model)()
            model_for_analysis.fit(X_train, y_train)
            
            if task_type == 'classification':
                error_analysis = self.error_analyzer.analyze_classification_errors(
                    model_for_analysis, X_val, y_val
                )
            else:
                error_analysis = {}
            
            if error_analysis and 'error' not in error_analysis:
                results['error_analysis'] = error_analysis
            elif error_analysis:
                results['error_analysis'] = {'error': error_analysis.get('error', 'Unknown error')}
            else:
                results['error_analysis'] = {}
        except Exception as e:
            print(f"    Warning: Error analysis failed: {e}")
            results['error_analysis'] = {'error': str(e)}
        
        # 4. Model Debugging
        try:
            print("  [4/5] Model Debugging...")
            # Use same model instance
            if 'model_for_analysis' not in locals():
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                    model_for_analysis = type(model)(**model_params)
                else:
                    model_for_analysis = type(model)()
            
            debug_report = self.model_debugger.debug(model_for_analysis, X_train, y_train, X_val, y_val, task_type)
            if 'error' not in debug_report:
                results['debug_report'] = debug_report
            else:
                results['debug_report'] = {'error': debug_report.get('error', 'Unknown error')}
        except Exception as e:
            print(f"    Warning: Model debugging failed: {e}")
            results['debug_report'] = {'error': str(e)}
        
        # 5. Recommendations
        print("  [5/5] Generating Recommendations...")
        all_recommendations = []
        
        if 'bias_variance_diagnosis' in results and 'error' not in results['bias_variance_diagnosis']:
            all_recommendations.extend(results['bias_variance_diagnosis'].get('recommendations', []))
        
        if 'learning_curves' in results and 'error' not in results['learning_curves']:
            all_recommendations.extend(results['learning_curves'].get('analysis', {}).get('recommendations', []))
        
        if 'error_analysis' in results and 'error' not in results['error_analysis']:
            all_recommendations.extend(results['error_analysis'].get('recommendations', []))
        
        if 'debug_report' in results and 'error' not in results['debug_report']:
            all_recommendations.extend(results['debug_report'].get('recommendations', []))
        
        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))
        results['prioritized_recommendations'] = unique_recommendations
        
        # Generate summary
        try:
            results['summary'] = self._generate_summary(
                results.get('bias_variance_diagnosis', {}),
                results.get('learning_curves', {}),
                results.get('error_analysis', {}),
                results.get('debug_report', {})
            )
        except Exception as e:
            print(f"    Warning: Summary generation failed: {e}")
            results['summary'] = {'error': str(e)}
        
        return results
    
    def _generate_summary(
        self,
        bias_variance: Dict,
        learning_curves: Dict,
        error_analysis: Dict,
        debug_report: Dict
    ) -> Dict[str, Any]:
        """Generate executive summary"""
        # Handle error cases gracefully
        bv_diagnosis = bias_variance.get('diagnosis', 'unknown') if 'error' not in bias_variance else 'error'
        bv_train = bias_variance.get('train_score', 0) if 'error' not in bias_variance else 0
        bv_val = bias_variance.get('val_score', 0) if 'error' not in bias_variance else 0
        bv_gap = bias_variance.get('gap', 0) if 'error' not in bias_variance else 0
        
        error_rate = error_analysis.get('error_rate', 0) if error_analysis and 'error' not in error_analysis else 0
        
        data_issues = 0
        feature_issues = 0
        if debug_report and 'error' not in debug_report:
            data_issues = len(debug_report.get('data_quality', {}).get('issues', []))
            feature_issues = len(debug_report.get('feature_analysis', {}).get('issues', []))
        
        return {
            'diagnosis': bv_diagnosis,
            'train_score': bv_train,
            'val_score': bv_val,
            'gap': bv_gap,
            'error_rate': error_rate,
            'data_quality_issues': data_issues,
            'feature_issues': feature_issues
        }
