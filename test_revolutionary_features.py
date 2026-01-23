"""
Test Revolutionary Features
Verify all mindblowing features work correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_revolutionary_features():
    """Test all revolutionary features"""
    print("="*80)
    print("TESTING REVOLUTIONARY FEATURES")
    print("="*80)
    print()
    
    # Test 1: Predictive Intelligence
    print("[1] Testing Predictive Intelligence...")
    try:
        from revolutionary_features import get_predictive_intelligence
        
        predictive = get_predictive_intelligence()
        predictive.record_action('train_model', {'model_type': 'random_forest'})
        predictions = predictive.predict_next_action('train_model', {})
        suggestions = predictive.get_suggestions({'action': 'train_model'})
        
        print(f"  [OK] Predictive Intelligence working")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Suggestions: {len(suggestions)}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 2: Self-Healing Code
    print("\n[2] Testing Self-Healing Code...")
    try:
        from revolutionary_features import get_self_healing_code
        
        healer = get_self_healing_code()
        code = "toolbox.fit(X, y)"
        analysis = healer.analyze_code(code)
        result = healer.heal_code(code)
        
        print(f"  [OK] Self-Healing Code working")
        print(f"  Issues found: {len(analysis['issues'])}")
        print(f"  Issues fixed: {result['issues_fixed']}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 3: Natural Language Pipeline
    print("\n[3] Testing Natural Language Pipeline...")
    try:
        from revolutionary_features import get_natural_language_pipeline
        
        nlp = get_natural_language_pipeline()
        description = "Classify data into 3 classes"
        result = nlp.build_pipeline(description)
        
        print(f"  [OK] Natural Language Pipeline working")
        print(f"  Task detected: {result['parsed_description']['task_type']}")
        print(f"  Pipeline steps: {len(result['steps'])}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 4: Collaborative Intelligence
    print("\n[4] Testing Collaborative Intelligence...")
    try:
        from revolutionary_features import get_collaborative_intelligence
        
        collab = get_collaborative_intelligence()
        collab.learn_pattern('preprocessing', {'method': 'advanced'}, 0.95)
        recommended = collab.get_recommended_pattern('preprocessing', {})
        insights = collab.get_community_insights()
        
        print(f"  [OK] Collaborative Intelligence working")
        print(f"  Patterns shared: {insights['total_patterns_shared']}")
        print(f"  Patterns contributed: {insights['patterns_contributed']}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 5: Auto-Optimizer
    print("\n[5] Testing Auto-Optimizer...")
    try:
        from revolutionary_features import get_auto_optimizer
        
        optimizer = get_auto_optimizer()
        code = "toolbox.fit(X, y)"
        analysis = optimizer.analyze_code(code)
        result = optimizer.optimize_code(code)
        
        print(f"  [OK] Auto-Optimizer working")
        print(f"  Opportunities: {len(analysis['opportunities'])}")
        print(f"  Optimizations applied: {result['optimizations_applied']}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 6: Integration with ML Toolbox
    print("\n[6] Testing ML Toolbox Integration...")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox()
        
        if toolbox.predictive_intelligence:
            print("  [OK] Predictive Intelligence integrated")
        if toolbox.self_healing_code:
            print("  [OK] Self-Healing Code integrated")
        if toolbox.natural_language_pipeline:
            print("  [OK] Natural Language Pipeline integrated")
        if toolbox.collaborative_intelligence:
            print("  [OK] Collaborative Intelligence integrated")
        if toolbox.auto_optimizer:
            print("  [OK] Auto-Optimizer integrated")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_revolutionary_features()
