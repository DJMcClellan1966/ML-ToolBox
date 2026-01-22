"""
Run Rigorous Test Suite
Comprehensive testing for correctness, errors, and performance
"""
import sys
from pathlib import Path
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent))

def run_tests():
    """Run all rigorous tests"""
    print("=" * 80)
    print("ML Toolbox - Rigorous Test Suite")
    print("=" * 80)
    print()
    
    test_files = [
        'tests/test_correctness_benchmarks.py',
        'tests/test_performance_benchmarks.py',
        'tests/test_error_handling.py',
        'tests/test_integration.py'
    ]
    
    results = {}
    
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Running: {test_file}")
        print(f"{'=' * 80}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_path), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test file
            )
            
            elapsed = time.time() - start_time
            
            results[test_file] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed': elapsed
            }
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED ({elapsed:.2f}s)")
            else:
                print(f"❌ {test_file} - FAILED ({elapsed:.2f}s)")
                print("\nOutput:")
                print(result.stdout)
                if result.stderr:
                    print("\nErrors:")
                    print(result.stderr)
        
        except subprocess.TimeoutExpired:
            print(f"⏱️  {test_file} - TIMEOUT (>5 minutes)")
            results[test_file] = {'timeout': True}
        except Exception as e:
            print(f"❌ {test_file} - ERROR: {e}")
            results[test_file] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r.get('returncode') == 0)
    failed = sum(1 for r in results.values() if r.get('returncode', 0) != 0 and 'returncode' in r)
    errors = sum(1 for r in results.values() if 'error' in r or 'timeout' in r)
    
    print(f"\nTotal test files: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Errors/Timeouts: {errors}")
    
    if failed == 0 and errors == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed or had errors")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
