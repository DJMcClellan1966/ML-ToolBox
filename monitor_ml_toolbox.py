"""
Monitor ML Toolbox Resource Usage
Comprehensive monitoring of CPU and memory usage
"""
import sys
from pathlib import Path
import time
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from ml_monitor import ResourceMonitor, MonitoredMLToolbox

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


def monitor_data_preprocessing():
    """Monitor data preprocessing operations"""
    print("Monitoring Data Preprocessing...")
    
    monitor = ResourceMonitor(sample_interval=0.1)
    monitor.start_monitoring()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000),
            'feature3': np.random.rand(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Monitor preprocessing
        @monitor.monitor_function
        def preprocess_data():
            # Try to get preprocessor
            try:
                from data_preprocessor import AdvancedDataPreprocessor
                preprocessor = AdvancedDataPreprocessor()
                cleaned = preprocessor.clean_data(data.copy())
                transformed = preprocessor.transform_data(cleaned.copy())
                return transformed
            except Exception as e:
                print(f"Error: {e}")
                return data
        
        result = preprocess_data()
        time.sleep(0.5)  # Let monitoring collect samples
        
        monitor.stop_monitoring()
        return monitor
    
    except Exception as e:
        print(f"Error monitoring preprocessing: {e}")
        monitor.stop_monitoring()
        return monitor


def monitor_model_training():
    """Monitor model training operations"""
    print("Monitoring Model Training...")
    
    monitor = ResourceMonitor(sample_interval=0.1)
    monitor.start_monitoring()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        
        # Monitor training
        @monitor.monitor_function
        def train_model():
            simple_ml = toolbox.algorithms.get_simple_ml_tasks()
            result = simple_ml.train_classifier(X, y, model_type='random_forest')
            return result
        
        result = train_model()
        time.sleep(0.5)
        
        monitor.stop_monitoring()
        return monitor
    
    except Exception as e:
        print(f"Error monitoring training: {e}")
        monitor.stop_monitoring()
        return monitor


def monitor_full_pipeline():
    """Monitor full ML pipeline"""
    print("Monitoring Full ML Pipeline...")
    
    monitor = ResourceMonitor(sample_interval=0.1)
    monitor.start_monitoring()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        data = pd.DataFrame(X)
        data['target'] = y
        
        # Monitor full pipeline
        @monitor.monitor_function
        def run_pipeline():
            # Preprocessing
            try:
                from data_preprocessor import AdvancedDataPreprocessor
                preprocessor = AdvancedDataPreprocessor()
                cleaned = preprocessor.clean_data(data.copy())
            except:
                cleaned = data
            
            # Training
            simple_ml = toolbox.algorithms.get_simple_ml_tasks()
            X_train = cleaned.drop(columns=['target']).values
            y_train = cleaned['target'].values
            result = simple_ml.train_classifier(X_train, y_train, model_type='random_forest')
            
            # Prediction
            predictions = result['model'].predict(X_train[:100])
            
            return result
        
        result = run_pipeline()
        time.sleep(0.5)
        
        monitor.stop_monitoring()
        return monitor
    
    except Exception as e:
        print(f"Error monitoring pipeline: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop_monitoring()
        return monitor


def run_comprehensive_monitoring():
    """Run comprehensive monitoring of ML Toolbox"""
    print("="*80)
    print("ML TOOLBOX COMPREHENSIVE RESOURCE MONITORING")
    print("="*80)
    print()
    
    all_monitors = {}
    
    # Monitor each component
    components = [
        ('data_preprocessing', monitor_data_preprocessing),
        ('model_training', monitor_model_training),
        ('full_pipeline', monitor_full_pipeline)
    ]
    
    for name, monitor_func in components:
        try:
            monitor = monitor_func()
            all_monitors[name] = monitor
            print(f"[OK] {name} monitoring complete")
        except Exception as e:
            print(f"[ERROR] {name} monitoring failed: {e}")
        print()
    
    # Combine monitoring data
    combined_monitor = ResourceMonitor()
    
    for name, monitor in all_monitors.items():
        # Merge function statistics
        for func_name, cpu_data in monitor.function_cpu.items():
            combined_monitor.function_cpu[func_name].extend(cpu_data)
        
        for func_name, mem_data in monitor.function_memory.items():
            combined_monitor.function_memory[func_name].extend(mem_data)
    
    # Generate comprehensive report
    print("="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    print()
    
    report = combined_monitor.generate_report('ml_toolbox_monitoring_report.txt')
    print(report)
    
    # Export data
    combined_monitor.export_data('ml_toolbox_monitoring_data.json')
    print("\n[OK] Monitoring data exported to ml_toolbox_monitoring_data.json")
    
    # Show bottlenecks
    bottlenecks = combined_monitor.identify_resource_bottlenecks()
    if bottlenecks:
        print("\n" + "="*80)
        print("RESOURCE BOTTLENECKS")
        print("="*80)
        for i, bottleneck in enumerate(bottlenecks[:10], 1):
            print(f"\n{i}. {bottleneck['type'].upper()} Bottleneck [{bottleneck.get('severity', 'unknown').upper()}]")
            if bottleneck.get('level') == 'function':
                print(f"   Function: {bottleneck.get('function', 'unknown')}")
            print(f"   Value: {bottleneck.get('value', 0):.2f}")
            print(f"   Recommendation: {bottleneck.get('recommendation', 'N/A')}")
    
    return combined_monitor


if __name__ == '__main__':
    monitor = run_comprehensive_monitoring()
