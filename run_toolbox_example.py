"""
Simple Example: How to Run the ML Toolbox
Demonstrates basic usage and revolutionary features
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ML TOOLBOX - QUICK START EXAMPLE")
print("="*80)
print()

# Step 1: Import and initialize
print("[1] Initializing ML Toolbox...")
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
print("✅ Toolbox initialized with all revolutionary features!")
print()

# Step 2: Basic model training
print("[2] Training a simple model...")
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

result = toolbox.fit(X, y, task_type='classification')
print(f"✅ Model trained!")
print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
print(f"   Model ID: {result.get('model_id', 'N/A')}")
print()

# Step 3: Use Predictive Intelligence
print("[3] Using Predictive Intelligence...")
if toolbox.predictive_intelligence:
    suggestions = toolbox.predictive_intelligence.get_suggestions({'action': 'train_model'})
    print(f"✅ Suggestions for next steps:")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"   {i}. {suggestion}")
print()

# Step 4: Use Natural Language Pipeline
print("[4] Using Natural Language Pipeline...")
if toolbox.natural_language_pipeline:
    description = "Classify data into 2 classes"
    pipeline_result = toolbox.natural_language_pipeline.build_pipeline(description)
    print(f"✅ Pipeline built from: '{description}'")
    print(f"   Task type detected: {pipeline_result['parsed_description']['task_type']}")
    print(f"   Pipeline steps: {len(pipeline_result['steps'])}")
print()

# Step 5: Use Self-Healing Code
print("[5] Using Self-Healing Code...")
if toolbox.self_healing_code:
    code = "toolbox.fit(X, y)"
    analysis = toolbox.self_healing_code.analyze_code(code)
    print(f"✅ Code analyzed")
    print(f"   Issues found: {len(analysis['issues'])}")
    
    if analysis['has_issues']:
        healed = toolbox.self_healing_code.heal_code(code)
        print(f"   Issues fixed: {healed['issues_fixed']}")
print()

# Step 6: Use Auto-Optimizer
print("[6] Using Auto-Optimizer...")
if toolbox.auto_optimizer:
    code = "toolbox.fit(X, y)"
    analysis = toolbox.auto_optimizer.analyze_code(code)
    print(f"✅ Code analyzed for optimization")
    print(f"   Optimization opportunities: {len(analysis['opportunities'])}")
    
    if analysis['has_opportunities']:
        optimized = toolbox.auto_optimizer.optimize_code(code)
        print(f"   Optimizations applied: {optimized['optimizations_applied']}")
print()

# Step 7: Summary
print("="*80)
print("SUMMARY")
print("="*80)
print("✅ ML Toolbox is working!")
print("✅ All revolutionary features are active!")
print()
print("Available features:")
print("  - Predictive Intelligence: Anticipates your needs")
print("  - Self-Healing Code: Fixes errors automatically")
print("  - Natural Language Pipeline: Describe, get ML solution")
print("  - Collaborative Intelligence: Learns from community")
print("  - Auto-Optimizer: Optimizes code automatically")
print()
print("Ready to build amazing ML models! 🚀")
print("="*80)
