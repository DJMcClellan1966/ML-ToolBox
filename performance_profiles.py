"""
Kuhn/Johnson Performance Profiles
Visual and statistical comparison of model performance

Features:
- Performance profile plots
- Boxplots of CV scores
- Statistical significance testing
- Model rankings
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import cross_val_score
    from scipy import stats
    SKLEARN_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False
    print("Warning: sklearn/scipy not available")

# Try to import matplotlib (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PerformanceProfile:
    """
    Create performance profiles for model comparison
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: Any = None,
        scoring: Optional[str] = None,
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {name: model} pairs
            X: Features
            y: Labels
            cv: Cross-validation splitter
            scoring: Scoring metric
            n_repeats: Number of CV repeats (for repeated k-fold)
            
        Returns:
            Dictionary with comparison results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        if cv is None:
            if len(np.unique(y)) < 10:  # Classification
                cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=self.random_state)
            else:
                cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=self.random_state)
        
        results = {}
        
        for model_name, model in models.items():
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            
            results[model_name] = {
                'cv_scores': cv_scores,
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'min': np.min(cv_scores),
                'max': np.max(cv_scores),
                'median': np.median(cv_scores),
                'q25': np.percentile(cv_scores, 25),
                'q75': np.percentile(cv_scores, 75)
            }
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['mean'])
        
        # Statistical significance testing
        significance_tests = self._significance_tests(results, X, y, cv, scoring)
        
        # Rankings
        rankings = self._rank_models(results)
        
        return {
            'models': results,
            'best_model': best_model,
            'best_score': results[best_model]['mean'],
            'rankings': rankings,
            'significance_tests': significance_tests,
            'scoring': scoring,
            'n_repeats': n_repeats
        }
    
    def _significance_tests(
        self,
        results: Dict[str, Dict],
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        scoring: str
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        model_names = list(results.keys())
        n_models = len(model_names)
        
        # Pairwise comparisons
        pairwise_tests = {}
        
        for i, model1_name in enumerate(model_names):
            for j, model2_name in enumerate(model_names[i+1:], i+1):
                scores1 = results[model1_name]['cv_scores']
                scores2 = results[model2_name]['cv_scores']
                
                # Paired t-test (scores are paired across CV folds)
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                # Mann-Whitney U test (non-parametric alternative)
                u_stat, p_value_mw = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                
                pairwise_tests[f"{model1_name}_vs_{model2_name}"] = {
                    't_test': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'mann_whitney': {
                        'u_statistic': float(u_stat),
                        'p_value': float(p_value_mw),
                        'significant': p_value_mw < 0.05
                    }
                }
        
        return pairwise_tests
    
    def _rank_models(self, results: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """Rank models by mean score"""
        rankings = sorted(
            results.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        return [(name, result['mean']) for name, result in rankings]
    
    def plot_comparison(
        self,
        comparison_results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot model comparison
        
        Args:
            comparison_results: Results from compare_models()
            save_path: Path to save figure (optional)
            show: Whether to display plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available for plotting")
            return
        
        models_data = comparison_results['models']
        model_names = list(models_data.keys())
        
        # Prepare data for boxplot
        scores_data = [models_data[name]['cv_scores'] for name in model_names]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Boxplot
        bp = axes[0].boxplot(scores_data, labels=model_names, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[0].set_title('Model Comparison - CV Scores Distribution')
        axes[0].set_ylabel(f"Score ({comparison_results['scoring']})")
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Bar plot of means with error bars
        means = [models_data[name]['mean'] for name in model_names]
        stds = [models_data[name]['std'] for name in model_names]
        
        bars = axes[1].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Highlight best model
        best_model = comparison_results['best_model']
        best_idx = model_names.index(best_model)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        axes[1].set_title('Model Comparison - Mean Scores with Error Bars')
        axes[1].set_ylabel(f"Mean Score ({comparison_results['scoring']})")
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self, comparison_results: Dict[str, Any]):
        """Print summary of model comparison"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        models_data = comparison_results['models']
        best_model = comparison_results['best_model']
        
        print(f"\nScoring Metric: {comparison_results['scoring']}")
        print(f"Best Model: {best_model} (Score: {models_data[best_model]['mean']:.4f} ± {models_data[best_model]['std']:.4f})")
        
        print("\nModel Rankings:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        
        rankings = comparison_results['rankings']
        for rank, (name, mean) in enumerate(rankings, 1):
            result = models_data[name]
            print(f"{rank:<6} {name:<30} {mean:<12.4f} {result['std']:<12.4f} "
                  f"{result['min']:<12.4f} {result['max']:<12.4f}")
        
        # Significance tests summary
        sig_tests = comparison_results.get('significance_tests', {})
        if sig_tests and 'error' not in sig_tests:
            print("\nStatistical Significance Tests (p < 0.05 = significant):")
            print("-" * 80)
            for comparison, tests in sig_tests.items():
                t_test = tests['t_test']
                mw_test = tests['mann_whitney']
                print(f"{comparison}:")
                print(f"  T-test: p={t_test['p_value']:.4f} ({'Significant' if t_test['significant'] else 'Not significant'})")
                print(f"  Mann-Whitney: p={mw_test['p_value']:.4f} ({'Significant' if mw_test['significant'] else 'Not significant'})")
        
        print("="*80 + "\n")
