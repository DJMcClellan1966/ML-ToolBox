"""
Dependency Manager
Checks and manages all dependencies with clear reporting

Improvement: Clean dependency management instead of warning spam
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib

sys.path.insert(0, str(Path(__file__).parent))


class DependencyManager:
    """
    Dependency Manager
    
    Improvement: Clean dependency checking and reporting
    """
    
    def __init__(self):
        self.core_dependencies = {
            'numpy': 'numpy',
            'ml_toolbox': 'ml_toolbox'
        }
        
        self.optional_dependencies = {
            'sklearn': 'scikit-learn',
            'sentence_transformers': 'sentence-transformers',
            'psutil': 'psutil',
            'py_cpuinfo': 'py-cpuinfo',
            'imbalanced_learn': 'imbalanced-learn',
            'statsmodels': 'statsmodels',
            'pgmpy': 'pgmpy',
            'hmmlearn': 'hmmlearn',
            'shap': 'shap',
            'lime': 'lime',
            'h5py': 'h5py'
        }
        
        self.feature_dependencies = {
            'advanced_preprocessing': ['sentence_transformers'],
            'gpu_acceleration': ['cupy'],  # If using GPU
            'monitoring': ['psutil'],
            'architecture_optimization': ['py_cpuinfo'],
            'statistical_learning': ['statsmodels', 'sklearn'],
            'bayesian_networks': ['pgmpy'],
            'hmm': ['hmmlearn'],
            'interpretability': ['shap', 'lime'],
            'model_persistence': ['h5py']
        }
    
    def check_all(self) -> Dict[str, Any]:
        """Check all dependencies"""
        core_status = self.check_core()
        optional_status = self.check_optional()
        feature_status = self.check_features()
        
        return {
            'core': core_status,
            'optional': optional_status,
            'features': feature_status,
            'summary': self._generate_summary(core_status, optional_status, feature_status)
        }
    
    def check_core(self) -> Dict[str, bool]:
        """Check core dependencies"""
        status = {}
        for name, module in self.core_dependencies.items():
            try:
                importlib.import_module(module)
                status[name] = True
            except ImportError:
                status[name] = False
        return status
    
    def check_optional(self) -> Dict[str, bool]:
        """Check optional dependencies"""
        status = {}
        for name, module in self.optional_dependencies.items():
            try:
                importlib.import_module(module)
                status[name] = True
            except ImportError:
                status[name] = False
        return status
    
    def check_features(self) -> Dict[str, Dict[str, Any]]:
        """Check feature dependencies"""
        feature_status = {}
        
        for feature, deps in self.feature_dependencies.items():
            available = []
            missing = []
            
            for dep in deps:
                if dep in self.optional_dependencies:
                    module = self.optional_dependencies[dep]
                    try:
                        importlib.import_module(module)
                        available.append(dep)
                    except ImportError:
                        missing.append(dep)
            
            feature_status[feature] = {
                'available': len(available) == len(deps),
                'available_deps': available,
                'missing_deps': missing
            }
        
        return feature_status
    
    def _generate_summary(self, core: Dict, optional: Dict, features: Dict) -> Dict[str, Any]:
        """Generate dependency summary"""
        core_missing = [name for name, status in core.items() if not status]
        optional_missing = [name for name, status in optional.items() if not status]
        features_disabled = [name for name, status in features.items() if not status['available']]
        
        return {
            'core_missing': core_missing,
            'optional_missing': optional_missing,
            'features_disabled': features_disabled,
            'all_core_available': len(core_missing) == 0,
            'install_suggestions': self._get_install_suggestions(optional_missing)
        }
    
    def _get_install_suggestions(self, missing: List[str]) -> List[str]:
        """Get pip install suggestions"""
        suggestions = []
        if missing:
            packages = [self.optional_dependencies.get(name, name) for name in missing]
            suggestions.append(f"pip install {' '.join(packages)}")
        return suggestions
    
    def print_summary(self, check_result: Optional[Dict] = None):
        """Print dependency summary"""
        if check_result is None:
            check_result = self.check_all()
        
        print("="*80)
        print("DEPENDENCY STATUS")
        print("="*80)
        
        # Core dependencies
        core = check_result['core']
        print("\nCore Dependencies:")
        for name, status in core.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {name}")
        
        if not check_result['summary']['all_core_available']:
            print("\n⚠️  WARNING: Missing core dependencies!")
            print("   The toolbox may not work correctly.")
        
        # Optional dependencies
        optional = check_result['optional']
        missing_optional = [name for name, status in optional.items() if not status]
        
        if missing_optional:
            print(f"\nOptional Dependencies ({len(missing_optional)} missing):")
            for name in missing_optional[:5]:  # Show first 5
                print(f"  ⚠️  {name} (optional)")
            
            if len(missing_optional) > 5:
                print(f"  ... and {len(missing_optional) - 5} more")
        
        # Features
        features = check_result['features']
        disabled_features = [name for name, status in features.items() if not status['available']]
        
        if disabled_features:
            print(f"\nDisabled Features ({len(disabled_features)}):")
            for name in disabled_features[:5]:
                print(f"  ⚠️  {name}")
        
        # Install suggestions
        suggestions = check_result['summary']['install_suggestions']
        if suggestions:
            print("\n💡 To enable more features, install:")
            for suggestion in suggestions:
                print(f"   {suggestion}")
        
        print("="*80)


# Global instance
_global_dependency_manager = None

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager"""
    global _global_dependency_manager
    if _global_dependency_manager is None:
        _global_dependency_manager = DependencyManager()
    return _global_dependency_manager
