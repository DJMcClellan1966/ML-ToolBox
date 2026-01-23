"""
Test Third Eye Feature
Demonstrates the code oracle capabilities
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_third_eye():
    """Test Third Eye feature"""
    print("="*80)
    print("THIRD EYE FEATURE TEST")
    print("="*80)
    print()
    
    try:
        from revolutionary_features import get_third_eye
        
        third_eye = get_third_eye()
        
        # Test 1: Look into future of good code
        print("[1] Looking into future of good code...")
        good_code = """
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y, task_type='classification')
"""
        
        prediction = third_eye.look_into_future(good_code)
        print(f"  Will work: {prediction['will_work']}")
        print(f"  Will fail: {prediction['will_fail']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  Direction: {prediction['direction']['intended_use']}")
        print(f"  Issues: {len(prediction['issues'])}")
        print(f"  Suggestions: {len(prediction['suggestions'])}")
        print(f"  Alternative uses: {len(prediction['alternative_uses'])}")
        print()
        
        # Test 2: Look into future of bad code
        print("[2] Looking into future of bad code...")
        bad_code = """
toolbox.fit(X, y)
"""
        
        prediction = third_eye.look_into_future(bad_code)
        print(f"  Will work: {prediction['will_work']}")
        print(f"  Will fail: {prediction['will_fail']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  Issues found: {len(prediction['issues'])}")
        for issue in prediction['issues'][:3]:
            print(f"    - {issue['message']}")
        print()
        
        # Test 3: See different use
        print("[3] Seeing different use for code...")
        code = """
from ml_toolbox import MLToolbox
toolbox = MLToolbox()
result = toolbox.fit(X, y, task_type='classification')
"""
        
        alternative = third_eye.see_different_use(code, 'classification')
        print(f"  Intended use: {alternative['intended_use']}")
        print(f"  Alternative uses found: {len(alternative['alternative_uses'])}")
        for alt in alternative['alternative_uses'][:3]:
            print(f"    - {alt['use']}: {alt['description']}")
        print()
        
        # Test 4: Predict future issues
        print("[4] Predicting future issues...")
        code = """
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y)
predictions = toolbox.predict(result['model_id'], X)
"""
        
        future_issues = third_eye.predict_future_issues(code)
        print(f"  Future issues predicted: {len(future_issues)}")
        for issue in future_issues:
            print(f"    Step: {issue['step']}")
            print(f"    Issue: {issue['issue']}")
            print(f"    Probability: {issue['probability']:.1%}")
            print()
        
        print("="*80)
        print("THIRD EYE TEST COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_third_eye()
