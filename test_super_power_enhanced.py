"""
Test Enhanced Super Power Agent - All Quick Wins and Phases
"""
import sys
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ENHANCED SUPER POWER AGENT TEST")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    print("\n[OK] ML Toolbox imported")
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)

# Generate test data
np.random.seed(42)
X = np.random.randn(1000, 20).astype(np.float64)
y = np.random.randint(0, 2, 1000)

print(f"\nTest data: {X.shape}, {len(y)} samples")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

# Test Enhanced Super Power Agent
if toolbox.super_power_agent:
    print("\n[OK] Super Power Agent available")
    agent = toolbox.super_power_agent
    
    # Test 1: Enhanced Error Messages
    print("\n1. ENHANCED ERROR MESSAGES")
    print("-"*80)
    response = toolbox.chat("Classify this data", X, None)  # Missing target
    print(f"Response: {response.get('message', 'No message')}")
    if 'suggestions' in response:
        print("Suggestions:")
        for sug in response['suggestions']:
            print(f"  - {sug}")
    
    # Test 2: Specialist Agent Integration
    print("\n2. SPECIALIST AGENT INTEGRATION")
    print("-"*80)
    response = toolbox.chat("Classify this data", X, y)
    print(f"Response: {response.get('message', 'No message')}")
    if 'result' in response and 'metrics' in response['result']:
        print(f"Metrics: {response['result']['metrics']}")
    
    # Test 3: Context Management
    print("\n3. CONTEXT MANAGEMENT")
    print("-"*80)
    context = {'previous_tasks': ['classification'], 'preferences': {'speed': True}}
    response = toolbox.chat("Improve it", X, y, context=context)
    print(f"Response: {response.get('message', 'No message')}")
    
    # Test 4: Enhanced NLU
    print("\n4. ENHANCED NATURAL LANGUAGE UNDERSTANDING")
    print("-"*80)
    variations = [
        "What can you tell me about this data?",
        "Analyze this dataset",
        "Examine the data",
        "Tell me about the data"
    ]
    for var in variations:
        intent = agent.understand_intent(var)
        print(f"'{var}' -> Task: {intent.task_type.value}")
    
    # Test 5: Multi-Agent Workflow
    print("\n5. MULTI-AGENT WORKFLOW")
    print("-"*80)
    if agent.orchestrator:
        workflow_result = agent.orchestrator.execute_workflow(
            workflow_type=agent.orchestrator.WorkflowType.ADAPTIVE,
            task_description="Build a classification model",
            data=X,
            target=y
        )
        print(f"Workflow type: {workflow_result.get('workflow_type')}")
        print(f"Agents used: {workflow_result.get('agents_used')}")
        print(f"Status: {workflow_result.get('status')}")
    
    # Test 6: Learning System
    print("\n6. LEARNING SYSTEM")
    print("-"*80)
    # Execute a task
    response = toolbox.chat("Predict values from this data", X, y)
    # Check learned patterns
    patterns = agent.get_learned_patterns()
    print(f"Learned patterns: {len(patterns)}")
    if patterns:
        for pattern_key, pattern_data in list(patterns.items())[:3]:
            print(f"  - {pattern_key}: success_rate={pattern_data.get('success_rate', 0):.2f}")
    
    # Test 7: Pattern Suggestions
    print("\n7. PATTERN SUGGESTIONS")
    print("-"*80)
    suggestion = agent.suggest_best_approach("Predict house prices")
    print(f"Suggestion: {suggestion.get('suggestion')}")
    print(f"Confidence: {suggestion.get('confidence', 0):.2f}")
    
    # Test 8: MLOps Integration
    print("\n8. MLOPS INTEGRATION")
    print("-"*80)
    # Train a model first
    result = toolbox.fit(X, y, task_type='classification')
    if isinstance(result, dict) and 'model' in result:
        model = result['model']
        # Test deployment
        response = toolbox.chat(
            "Deploy this model to production",
            model=model,
            model_name="test_classifier",
            version="1.0"
        )
        print(f"Deployment response: {response.get('message', 'No message')}")
        if 'result' in response and 'recommendations' in response['result']:
            print("Recommendations:")
            for rec in response['result']['recommendations']:
                print(f"  - {rec}")
    
    print("\n" + "="*80)
    print("ENHANCED SUPER POWER AGENT TEST COMPLETE")
    print("="*80)
else:
    print("\n[SKIP] Super Power Agent not available")
