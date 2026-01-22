"""
MLOps Examples
Complete examples for Compartment 4: MLOps & Production

Demonstrates:
- Model monitoring
- Model deployment
- A/B testing
- Experiment tracking
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import required components
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")
    exit(1)

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Warning: ML Toolbox not available")
    exit(1)


def example_model_monitoring():
    """Example: Model monitoring and drift detection"""
    print("=" * 60)
    print("Example 1: Model Monitoring & Drift Detection")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Create model monitor
    monitor = toolbox.mlops.get_model_monitor(
        model=model,
        reference_data=X_train,
        reference_labels=y_train,
        baseline_performance=0.9,
        model_name='example_model'
    )
    
    # Monitor test data
    print("\nMonitoring test data...")
    results = monitor.monitor(
        X_test,
        y_test,
        check_data_drift=True,
        check_concept_drift=True,
        check_performance=True
    )
    
    print(f"\nData Drift: {results['data_drift']['has_drift']}")
    print(f"Concept Drift: {results['concept_drift']['has_drift']}")
    print(f"Performance Metrics: {results['performance']['metrics']}")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\nMonitoring Summary:")
    print(f"  Model: {summary['model_name']}")
    print(f"  Alerts: {summary['n_alerts']}")
    print(f"  Performance: {summary['performance_summary']}")
    
    print("\n✓ Model monitoring example complete\n")


def example_model_deployment():
    """Example: Model deployment and serving"""
    print("=" * 60)
    print("Example 2: Model Deployment & Serving")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Create model registry
    registry = toolbox.mlops.get_model_registry()
    
    # Register model
    print("\nRegistering model...")
    registry.register(model, version='v1.0.0', set_active=True, metadata={
        'accuracy': 0.95,
        'trained_at': '2024-01-01'
    })
    
    print(f"Active version: {registry.active_version}")
    print(f"Registered versions: {[v['version'] for v in registry.list_versions()]}")
    
    # Test batch inference
    print("\nTesting batch inference...")
    batch_inference = toolbox.mlops.get_batch_inference(model)
    predictions = batch_inference.predict_batch(X_test[:10], batch_size=5)
    print(f"Batch predictions: {predictions[:5]}")
    
    # Test real-time inference
    print("\nTesting real-time inference...")
    realtime = toolbox.mlops.get_realtime_inference(model)
    prediction = realtime.predict(X_test[0])
    print(f"Real-time prediction: {prediction}")
    
    print("\n✓ Model deployment example complete")
    print("  Note: To start API server, use:")
    print("  server = toolbox.mlops.get_model_server(registry)")
    print("  server.run(host='0.0.0.0', port=8000)\n")


def example_ab_testing():
    """Example: A/B testing for ML models"""
    print("=" * 60)
    print("Example 3: A/B Testing")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train two models
    model_a = RandomForestClassifier(n_estimators=10, random_state=42)
    model_b = RandomForestClassifier(n_estimators=20, random_state=42)
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Create A/B test
    print("\nCreating A/B test...")
    ab_test = toolbox.mlops.get_ab_test(
        test_name='model_comparison',
        variants={'model_a': model_a, 'model_b': model_b},
        traffic_split={'model_a': 0.5, 'model_b': 0.5}
    )
    
    # Simulate requests
    print("\nSimulating requests...")
    for i in range(20):
        variant = ab_test.route_request(request_id=f'request_{i}')
        model = ab_test.variants[variant]
        prediction = model.predict(X_test[i:i+1])[0]
        ab_test.record_prediction(variant, prediction, y_test[i])
    
    # Compare variants
    print("\nComparing variants...")
    comparison = ab_test.compare_variants(metric_name='accuracy')
    print(f"Variant A accuracy: {comparison['metrics_a']['mean']:.4f}")
    print(f"Variant B accuracy: {comparison['metrics_b']['mean']:.4f}")
    print(f"Better variant: {comparison['better_variant']}")
    print(f"Significant difference: {comparison['is_significant']}")
    print(f"Recommendation: {comparison['recommendation']}")
    
    print("\n✓ A/B testing example complete\n")


def example_experiment_tracking():
    """Example: Experiment tracking"""
    print("=" * 60)
    print("Example 4: Experiment Tracking")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Create experiment tracker
    tracker = toolbox.mlops.get_experiment_tracker(storage_dir='experiments_example')
    
    # Create experiment
    print("\nCreating experiment...")
    experiment = tracker.create_experiment(
        experiment_name='random_forest_tuning',
        description='Testing different n_estimators values'
    )
    
    # Log parameters
    experiment.log_parameters({
        'n_estimators': 10,
        'max_depth': 10,
        'random_state': 42
    })
    
    # Train and evaluate model
    model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    # Log metrics
    experiment.log_metrics({
        'accuracy': accuracy,
        'f1': 0.92  # Example
    })
    
    # Save model
    experiment.save_model(model, model_name='rf_model.pkl')
    
    # Add tags and notes
    experiment.add_tag('random_forest')
    experiment.add_tag('classification')
    experiment.add_note('Initial experiment with default parameters')
    
    # Complete experiment
    experiment.complete()
    
    # Save experiment
    tracker.save_experiment(experiment.experiment_id)
    
    # List experiments
    print("\nListing experiments...")
    experiments = tracker.list_experiments()
    print(f"Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['experiment_name']}: {exp['status']}")
    
    # Get best experiment
    print("\nFinding best experiment...")
    best = tracker.get_best_experiment('accuracy', maximize=True)
    if best:
        print(f"Best experiment: {best.experiment_name}")
        print(f"Best accuracy: {best.get_latest_metrics().get('accuracy')}")
    
    print("\n✓ Experiment tracking example complete\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MLOps Examples - Compartment 4: MLOps & Production")
    print("=" * 60 + "\n")
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required dependencies not available. Please install:")
        print("  pip install scikit-learn")
        exit(1)
    
    try:
        example_model_monitoring()
        example_model_deployment()
        example_ab_testing()
        example_experiment_tracking()
        
        print("=" * 60)
        print("All MLOps examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
