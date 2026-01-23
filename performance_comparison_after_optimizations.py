"""
Performance Comparison: Before vs After Optimizations
"""
import json
from pathlib import Path
from typing import Dict, List, Any

def compare_performance():
    """Compare performance before and after optimizations"""
    
    # Previous results (from earlier test)
    previous_results = {
        'simple_tests': {
            'binary_classification': {'toolbox': 0.1232, 'sklearn': 0.0179},
            'multiclass_classification': {'toolbox': 0.2585, 'sklearn': 0.0290},
            'simple_regression': {'toolbox': 0.1741, 'sklearn': 0.0228},
            'basic_clustering': {'toolbox': 2.3372, 'sklearn': 0.0450}
        },
        'medium_tests': {
            'high_dim_classification': {'toolbox': 0.3625, 'sklearn': 0.0545},
            'imbalanced_classification': {'toolbox': 0.2146, 'sklearn': 0.0256},
            'time_series_regression': {'toolbox': 0.1592, 'sklearn': 0.0215},
            'multi_output_regression': {'toolbox': 0.1493, 'sklearn': 0.0103},
            'feature_selection': {'toolbox': 0.0056, 'sklearn': 0.0042}
        },
        'hard_tests': {
            'very_high_dim': {'toolbox': 0.8566, 'sklearn': 0.0616},
            'nonlinear_patterns': {'toolbox': 0.2048, 'sklearn': 0.0245},
            'sparse_data': {'toolbox': 0.2227, 'sklearn': 0.0311},
            'noisy_data': {'toolbox': 0.2011, 'sklearn': 0.0306},
            'ensemble': {'toolbox': 0.2149, 'sklearn': 0.0486}
        }
    }
    
    # Current results (from latest test)
    current_results = {
        'simple_tests': {
            'binary_classification': {'toolbox': 0.2213, 'sklearn': 0.0190},
            'multiclass_classification': {'toolbox': 0.2009, 'sklearn': 0.0332},
            'simple_regression': {'toolbox': 0.1598, 'sklearn': 0.0247},
            'basic_clustering': {'toolbox': 2.9877, 'sklearn': 0.0546}
        },
        'medium_tests': {
            'high_dim_classification': {'toolbox': 0.4148, 'sklearn': 0.0568},
            'imbalanced_classification': {'toolbox': 0.2241, 'sklearn': 0.0286},
            'time_series_regression': {'toolbox': 0.1367, 'sklearn': 0.0072},
            'multi_output_regression': {'toolbox': 0.1931, 'sklearn': 0.0209},
            'feature_selection': {'toolbox': 0.0203, 'sklearn': 0.0006}
        },
        'hard_tests': {
            'very_high_dim': {'toolbox': 0.9264, 'sklearn': 0.1442},
            'nonlinear_patterns': {'toolbox': 0.2418, 'sklearn': 0.0296},
            'sparse_data': {'toolbox': 0.1739, 'sklearn': 0.0109},
            'noisy_data': {'toolbox': 0.1475, 'sklearn': 0.0283},
            'ensemble': {'toolbox': 0.3368, 'sklearn': 0.0338}
        }
    }
    
    print("="*80)
    print("PERFORMANCE COMPARISON: BEFORE vs AFTER OPTIMIZATIONS")
    print("="*80)
    print()
    
    improvements = []
    regressions = []
    
    for category in ['simple_tests', 'medium_tests', 'hard_tests']:
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-"*80)
        print(f"{'Test':<30} {'Before':<12} {'After':<12} {'Change':<12} {'Speedup':<10}")
        print("-"*80)
        
        for test_name in previous_results[category].keys():
            if test_name not in current_results[category]:
                continue
            
            prev_time = previous_results[category][test_name]['toolbox']
            curr_time = current_results[category][test_name]['toolbox']
            
            if prev_time > 0:
                change = ((prev_time - curr_time) / prev_time) * 100
                speedup = prev_time / curr_time if curr_time > 0 else 0
                
                if change > 0:
                    improvements.append({
                        'category': category,
                        'test': test_name,
                        'before': prev_time,
                        'after': curr_time,
                        'improvement': change,
                        'speedup': speedup
                    })
                    print(f"{test_name:<30} {prev_time:<12.4f} {curr_time:<12.4f} {change:<11.1f}% {speedup:<10.2f}x")
                else:
                    regressions.append({
                        'category': category,
                        'test': test_name,
                        'before': prev_time,
                        'after': curr_time,
                        'regression': abs(change)
                    })
                    print(f"{test_name:<30} {prev_time:<12.4f} {curr_time:<12.4f} {change:<11.1f}% (slower)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nImprovements: {len(improvements)} tests")
    if improvements:
        avg_improvement = sum(i['improvement'] for i in improvements) / len(improvements)
        avg_speedup = sum(i['speedup'] for i in improvements) / len(improvements)
        print(f"  Average improvement: {avg_improvement:.1f}%")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        
        print("\nTop Improvements:")
        for imp in sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:5]:
            print(f"  - {imp['test']}: {imp['improvement']:.1f}% faster ({imp['speedup']:.2f}x)")
    
    print(f"\nRegressions: {len(regressions)} tests")
    if regressions:
        avg_regression = sum(r['regression'] for r in regressions) / len(regressions)
        print(f"  Average regression: {avg_regression:.1f}%")
        
        print("\nRegressions:")
        for reg in sorted(regressions, key=lambda x: x['regression'], reverse=True):
            print(f"  - {reg['test']}: {reg['regression']:.1f}% slower")
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    # Calculate averages
    prev_avg = {}
    curr_avg = {}
    
    for category in ['simple_tests', 'medium_tests', 'hard_tests']:
        prev_times = [previous_results[category][t]['toolbox'] for t in previous_results[category].keys()]
        curr_times = [current_results[category][t]['toolbox'] for t in current_results[category].keys() if t in previous_results[category]]
        
        if prev_times and curr_times:
            prev_avg[category] = sum(prev_times) / len(prev_times)
            curr_avg[category] = sum(curr_times) / len(curr_times)
    
    print(f"\n{'Category':<20} {'Before Avg':<15} {'After Avg':<15} {'Change':<15}")
    print("-"*80)
    
    for category in prev_avg.keys():
        change = ((prev_avg[category] - curr_avg[category]) / prev_avg[category]) * 100
        print(f"{category.replace('_', ' '):<20} {prev_avg[category]:<15.4f} {curr_avg[category]:<15.4f} {change:<14.1f}%")
    
    overall_prev = sum(prev_avg.values()) / len(prev_avg) if prev_avg else 0
    overall_curr = sum(curr_avg.values()) / len(curr_avg) if curr_avg else 0
    
    if overall_prev > 0:
        overall_change = ((overall_prev - overall_curr) / overall_prev) * 100
        print(f"\n{'Overall Average':<20} {overall_prev:<15.4f} {overall_curr:<15.4f} {overall_change:<14.1f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if len(improvements) > len(regressions):
        print("✅ Optimizations show overall improvement!")
        print(f"   {len(improvements)} tests improved vs {len(regressions)} regressions")
    elif len(improvements) == len(regressions):
        print("⚠️  Mixed results - some improvements, some regressions")
    else:
        print("⚠️  More regressions than improvements - may need further optimization")


if __name__ == '__main__':
    compare_performance()
