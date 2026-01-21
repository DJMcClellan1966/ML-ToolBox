"""
Test Runner for ML Toolbox
Runs all ML toolbox related tests
"""
import sys
from pathlib import Path
import importlib.util
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test files to run
TEST_FILES = [
    'tests/test_ml_evaluation.py',
    'tests/test_ensemble_learning.py',
    'tests/test_preprocessor_comparison.py',
    'tests/test_compression.py',
    'tests/test_bag_of_words_comparison.py',
    'tests/test_regression_analysis.py',
    'tests/test_clustering_analysis.py',
]

def run_test_file(test_file: str) -> tuple[bool, str]:
    """Run a test file and return (success, message)"""
    try:
        print(f"\n{'='*80}")
        print(f"Running: {test_file}")
        print('='*80)
        
        # Load and execute the test file
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        if spec is None or spec.loader is None:
            return False, f"Could not load {test_file}"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for test functions
        test_functions = [name for name in dir(module) if name.startswith('test_')]
        
        if test_functions:
            for test_func_name in test_functions:
                test_func = getattr(module, test_func_name)
                if callable(test_func):
                    print(f"\n[Running] {test_func_name}")
                    test_func()
        else:
            # Try to find a main function or run_all_tests
            if hasattr(module, 'run_all_tests'):
                module.run_all_tests()
            elif hasattr(module, 'main'):
                module.main()
            else:
                # Just execute the module
                pass
        
        return True, f"[PASS] {test_file} passed"
        
    except Exception as e:
        error_msg = f"[FAIL] {test_file} failed: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg

def main():
    """Run all ML toolbox tests"""
    print("="*80)
    print("ML TOOLBOX TEST SUITE")
    print("="*80)
    
    results = []
    passed = 0
    failed = 0
    
    for test_file in TEST_FILES:
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"\n[SKIP] {test_file} not found")
            continue
        
        success, message = run_test_file(str(test_path))
        results.append((test_file, success, message))
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print("\nDetailed Results:")
    for test_file, success, message in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status}: {test_file}")
        if not success:
            print(f"    {message}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} TEST(S) FAILED")
    
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
