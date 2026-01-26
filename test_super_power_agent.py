"""
Test Super Power Agent - Natural Language ML Interface
"""
import sys
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("SUPER POWER AGENT TEST")
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

# Test Super Power Agent
if toolbox.super_power_agent:
    print("\n[OK] Super Power Agent available")
    
    # Test 1: Classification
    print("\n1. CLASSIFICATION TASK")
    print("-"*80)
    response = toolbox.chat("Classify this data", X, y)
    print(f"Response: {response.get('message', 'No message')}")
    if 'result' in response:
        result = response['result']
        print(f"Task: {result.get('task', 'Unknown')}")
        if 'metrics' in result:
            print(f"Metrics: {result['metrics']}")
    
    # Test 2: Regression
    print("\n2. REGRESSION TASK")
    print("-"*80)
    y_reg = np.random.randn(1000)
    response = toolbox.chat("Predict values from this data", X, y_reg)
    print(f"Response: {response.get('message', 'No message')}")
    if 'result' in response and 'metrics' in response['result']:
        print(f"Metrics: {response['result']['metrics']}")
    
    # Test 3: Feature Engineering
    print("\n3. FEATURE ENGINEERING TASK")
    print("-"*80)
    response = toolbox.chat("Engineer features from this data", X, y)
    print(f"Response: {response.get('message', 'No message')}")
    if 'result' in response:
        result = response['result']
        if 'engineered_shape' in result:
            print(f"Original shape: {result['original_shape']}")
            print(f"Engineered shape: {result['engineered_shape']}")
    
    # Test 4: Analysis
    print("\n4. DATA ANALYSIS TASK")
    print("-"*80)
    response = toolbox.chat("Analyze this data", X, y)
    print(f"Response: {response.get('message', 'No message')}")
    if 'result' in response:
        result = response['result']
        if 'data_info' in result:
            print(f"Data info: Shape {result['data_shape']}")
    
    # Test 5: Natural Language Variations
    print("\n5. NATURAL LANGUAGE VARIATIONS")
    print("-"*80)
    variations = [
        "Predict house prices",
        "Train a classifier",
        "Build a regression model",
        "What can you tell me about this data?"
    ]
    
    for var in variations:
        intent = toolbox.super_power_agent.understand_intent(var)
        print(f"'{var}' -> Task: {intent.task_type.value}")
    
    print("\n" + "="*80)
    print("SUPER POWER AGENT TEST COMPLETE")
    print("="*80)
else:
    print("\n[SKIP] Super Power Agent not available")
