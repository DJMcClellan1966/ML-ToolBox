"""
Andrew Ng ML Strategy Demo
Demonstrates systematic ML development approach
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("="*80)
print("ANDREW NG ML STRATEGY DEMO")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

# Generate sample data
print("\n[1] Generating sample data...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")

# Get Andrew Ng ML Strategy
print("\n[2] Initializing Andrew Ng ML Strategy...")
strategy = toolbox.algorithms.get_andrew_ng_strategy()

# Complete Analysis
print("\n[3] Running Complete Analysis...")
model = RandomForestClassifier(n_estimators=50, random_state=42)

results = strategy.complete_analysis(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    task_type='classification'
)

# Display Results
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

# Check for errors
if 'error' in results:
    print(f"\n[ERROR] {results['error']}")
    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)
    exit(0)

# Bias/Variance Diagnosis
if 'bias_variance_diagnosis' in results:
    print("\n[BIAS/VARIANCE DIAGNOSIS]")
    bv = results['bias_variance_diagnosis']
    if 'error' not in bv:
        print(f"  Diagnosis: {bv.get('diagnosis', 'unknown')}")
        print(f"  Training Score: {bv.get('train_score', 0):.4f}")
        print(f"  Validation Score: {bv.get('val_score', 0):.4f}")
        print(f"  Gap: {bv.get('gap', 0):.4f}")
        print(f"  Explanation: {bv.get('explanation', 'N/A')}")
        print(f"\n  Recommendations:")
        for rec in bv.get('recommendations', []):
            print(f"    - {rec}")
    else:
        print(f"  Error: {bv.get('error', 'Unknown error')}")

# Learning Curves
if 'learning_curves' in results:
    print("\n[LEARNING CURVES]")
    lc = results['learning_curves']
    if 'error' not in lc and 'analysis' in lc:
        analysis = lc['analysis']
        print(f"  Would more data help: {analysis.get('would_more_data_help', False)}")
        print(f"  Final gap: {analysis.get('final_gap', 0):.4f}")
        print(f"  Validation trend: {analysis.get('val_trend', 0):.4f}")
        print(f"\n  Recommendations:")
        for rec in analysis.get('recommendations', []):
            print(f"    - {rec}")
    else:
        print(f"  Error: {lc.get('error', 'Unknown error')}")

# Error Analysis (if classification)
if 'error_analysis' in results and results['error_analysis']:
    print("\n[ERROR ANALYSIS]")
    ea = results['error_analysis']
    if 'error' not in ea:
        print(f"  Error Rate: {ea.get('error_rate', 0):.4f}")
        print(f"  Error Count: {ea.get('error_count', 0)}")
        print(f"\n  Recommendations:")
        for rec in ea.get('recommendations', [])[:5]:  # Top 5
            print(f"    - {rec}")
    else:
        print(f"  Error: {ea.get('error', 'Unknown error')}")

# Debug Report
if 'debug_report' in results:
    print("\n[DEBUG REPORT]")
    dr = results['debug_report']
    if 'error' not in dr:
        print(f"  Data Quality Issues: {len(dr.get('data_quality', {}).get('issues', []))}")
        print(f"  Feature Issues: {len(dr.get('feature_analysis', {}).get('issues', []))}")
        print(f"  Model Performance Issues: {len(dr.get('model_performance', {}).get('issues', []))}")
    else:
        print(f"  Error: {dr.get('error', 'Unknown error')}")

# Summary
print("\n[SUMMARY]")
summary = results['summary']
print(f"  Overall Diagnosis: {summary['diagnosis']}")
print(f"  Training Score: {summary['train_score']:.4f}")
print(f"  Validation Score: {summary['val_score']:.4f}")
print(f"  Gap: {summary['gap']:.4f}")
print(f"  Error Rate: {summary['error_rate']:.4f}")

# Prioritized Recommendations
print("\n[PRIORITIZED RECOMMENDATIONS]")
for i, rec in enumerate(results['prioritized_recommendations'][:10], 1):
    print(f"  {i}. {rec}")

print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)

# Individual Component Examples
print("\n[4] Individual Component Examples...")

# Error Analyzer
print("\n  [Error Analyzer]")
error_analyzer = toolbox.algorithms.get_error_analyzer()
model.fit(X_train, y_train)
error_results = error_analyzer.analyze_classification_errors(model, X_val, y_val)
if 'error' not in error_results:
    print(f"    Error Rate: {error_results['error_rate']:.4f}")
    print(f"    Common Patterns: {len(error_results['common_error_patterns'])}")

# Bias/Variance Diagnostic
print("\n  [Bias/Variance Diagnostic]")
bv_diagnostic = toolbox.algorithms.get_bias_variance_diagnostic()
bv_results = bv_diagnostic.diagnose(model, X_train, y_train, X_val, y_val)
print(f"    Diagnosis: {bv_results['diagnosis']}")

# Learning Curves
print("\n  [Learning Curves Analyzer]")
lc_analyzer = toolbox.algorithms.get_learning_curves_analyzer()
X_all = np.vstack([X_train, X_val])
y_all = np.hstack([y_train, y_val])
lc_results = lc_analyzer.analyze(model, X_all, y_all, task_type='classification')
print(f"    Would more data help: {lc_results['analysis']['would_more_data_help']}")

# Model Selector
print("\n  [Systematic Model Selector]")
model_selector = toolbox.algorithms.get_systematic_model_selector()
models = {
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}
comparison = model_selector.compare_models(models, X_train, y_train, task_type='classification')
print(f"    Best Model: {comparison['best_model']['name']}")
print(f"    Best CV Score: {comparison['best_model']['cv_mean']:.4f}")

print("\n" + "="*80)
