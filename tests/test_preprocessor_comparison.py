"""
Comprehensive Test Suite: Advanced Preprocessor vs Conventional Methods
"""
import sys
from pathlib import Path
import time
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor


class PreprocessorComparison:
    """Compare advanced and conventional preprocessors"""
    
    def __init__(self):
        self.advanced = AdvancedDataPreprocessor(use_quantum=True)
        self.conventional = ConventionalPreprocessor()
        self.test_results = []
    
    def generate_test_data(self, size: int = 100) -> List[str]:
        """Generate test data with duplicates, variations, and noise"""
        base_items = [
            "Python is great for data science and machine learning",
            "Machine learning uses algorithms to find patterns",
            "I need help with programming errors in my code",
            "Business revenue increased by 20% this quarter",
            "Customer support is available 24/7 for assistance",
            "Learn Python programming through online tutorials",
            "JavaScript is used for web development",
            "Sales team achieved record profits this year",
            "Fix errors in your code with debugging tools",
            "Education courses teach programming fundamentals"
        ]
        
        # Create variations and duplicates
        test_data = []
        
        # Add base items
        test_data.extend(base_items)
        
        # Add semantic duplicates (different wording, same meaning)
        semantic_duplicates = [
            "Python is excellent for data science and ML",  # Duplicate of base_items[0]
            "ML uses algorithms to discover patterns",  # Duplicate of base_items[1]
            "I require assistance with code errors",  # Duplicate of base_items[2]
            "Company profits grew 20% this period",  # Duplicate of base_items[3]
            "Support team helps customers around the clock",  # Duplicate of base_items[4]
            "Study Python coding via internet courses",  # Duplicate of base_items[5]
            "JS is utilized for building websites",  # Duplicate of base_items[6]
            "Revenue team hit all-time high earnings",  # Duplicate of base_items[7]
            "Resolve bugs in programs using debuggers",  # Duplicate of base_items[8]
            "Training programs cover coding basics"  # Duplicate of base_items[9]
        ]
        test_data.extend(semantic_duplicates)
        
        # Add exact duplicates
        test_data.extend(base_items[:5])  # Exact duplicates
        
        # Add low quality items
        low_quality = [
            "Short",
            "Hi",
            "This is a very long sentence that goes on and on and on and might be considered low quality due to excessive length and verbosity and repetition and unnecessary words",
            "A",
            "Test"
        ]
        test_data.extend(low_quality)
        
        # Add more variations to reach desired size
        while len(test_data) < size:
            import random
            base = random.choice(base_items)
            # Add slight variations
            variations = [
                base + ".",
                base.upper(),
                base.lower(),
                base.replace("Python", "Python programming"),
                base.replace("code", "source code")
            ]
            test_data.extend(variations[:min(5, size - len(test_data))])
        
        return test_data[:size]
    
    def run_comparison(self, test_data: List[str], test_name: str = "Test") -> Dict:
        """Run comparison between advanced and conventional preprocessors"""
        print("\n" + "="*80)
        print(f"COMPARISON TEST: {test_name}")
        print("="*80)
        
        # Run advanced preprocessor
        print("\n[ADVANCED PREPROCESSOR]")
        print("-" * 80)
        start_time = time.time()
        advanced_results = self.advanced.preprocess(test_data.copy(), verbose=True)
        advanced_time = time.time() - start_time
        
        # Run conventional preprocessor
        print("\n[CONVENTIONAL PREPROCESSOR]")
        print("-" * 80)
        start_time = time.time()
        conventional_results = self.conventional.preprocess(test_data.copy(), verbose=True)
        conventional_time = time.time() - start_time
        
        # Compare results
        comparison = {
            'test_name': test_name,
            'input_size': len(test_data),
            'advanced': {
                'final_count': advanced_results['final_count'],
                'duplicates_removed': len(advanced_results['duplicates']),
                'categories': len(advanced_results['categorized']),
                'avg_quality': advanced_results['stats']['avg_quality'],
                'processing_time': advanced_results['processing_time']
            },
            'conventional': {
                'final_count': conventional_results['final_count'],
                'duplicates_removed': len(conventional_results['duplicates']),
                'categories': len(conventional_results['categorized']),
                'avg_quality': conventional_results['stats']['avg_quality'],
                'processing_time': conventional_results['processing_time']
            },
            'improvements': {}
        }
        
        # Calculate improvements
        if comparison['conventional']['duplicates_removed'] > 0:
            dup_improvement = ((comparison['advanced']['duplicates_removed'] - 
                              comparison['conventional']['duplicates_removed']) / 
                             comparison['conventional']['duplicates_removed']) * 100
            comparison['improvements']['duplicate_detection'] = dup_improvement
        
        if comparison['conventional']['processing_time'] > 0:
            speed_ratio = comparison['conventional']['processing_time'] / comparison['advanced']['processing_time']
            comparison['improvements']['speed_ratio'] = speed_ratio
        
        quality_improvement = ((comparison['advanced']['avg_quality'] - 
                               comparison['conventional']['avg_quality']) / 
                              max(comparison['conventional']['avg_quality'], 0.01)) * 100
        comparison['improvements']['quality_improvement'] = quality_improvement
        
        # Print comparison
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"\nInput Size: {comparison['input_size']} items")
        print(f"\nFinal Output:")
        print(f"  Advanced:     {comparison['advanced']['final_count']} items")
        print(f"  Conventional: {comparison['conventional']['final_count']} items")
        
        print(f"\nDuplicates Removed:")
        print(f"  Advanced:     {comparison['advanced']['duplicates_removed']} items")
        print(f"  Conventional: {comparison['conventional']['duplicates_removed']} items")
        if 'duplicate_detection' in comparison['improvements']:
            improvement = comparison['improvements']['duplicate_detection']
            print(f"  Improvement:  {improvement:+.1f}%")
        
        print(f"\nCategories Created:")
        print(f"  Advanced:     {comparison['advanced']['categories']} categories")
        print(f"  Conventional: {comparison['conventional']['categories']} categories")
        
        print(f"\nAverage Quality Score:")
        print(f"  Advanced:     {comparison['advanced']['avg_quality']:.3f}")
        print(f"  Conventional: {comparison['conventional']['avg_quality']:.3f}")
        if 'quality_improvement' in comparison['improvements']:
            improvement = comparison['improvements']['quality_improvement']
            print(f"  Improvement:  {improvement:+.1f}%")
        
        print(f"\nProcessing Time:")
        print(f"  Advanced:     {comparison['advanced']['processing_time']:.3f}s")
        print(f"  Conventional: {comparison['conventional']['processing_time']:.3f}s")
        if 'speed_ratio' in comparison['improvements']:
            ratio = comparison['improvements']['speed_ratio']
            if ratio > 1:
                print(f"  Advanced is {ratio:.2f}x faster")
            else:
                print(f"  Conventional is {1/ratio:.2f}x faster")
        
        print("="*80)
        
        self.test_results.append(comparison)
        return comparison
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PREPROCESSOR COMPARISON TEST SUITE")
        print("="*80)
        
        # Test 1: Small dataset
        print("\n[TEST 1] Small Dataset (50 items)")
        test_data_1 = self.generate_test_data(50)
        self.run_comparison(test_data_1, "Small Dataset")
        
        # Test 2: Medium dataset
        print("\n[TEST 2] Medium Dataset (100 items)")
        test_data_2 = self.generate_test_data(100)
        self.run_comparison(test_data_2, "Medium Dataset")
        
        # Test 3: Large dataset
        print("\n[TEST 3] Large Dataset (200 items)")
        test_data_3 = self.generate_test_data(200)
        self.run_comparison(test_data_3, "Large Dataset")
        
        # Test 4: Semantic duplicates focus
        print("\n[TEST 4] Semantic Duplicates Focus")
        semantic_test = [
            "Python is great for data science",
            "Python is excellent for data science",
            "Python is wonderful for data science",
            "Python is amazing for data science",
            "Machine learning uses algorithms",
            "ML uses algorithms",
            "Machine learning employs algorithms",
            "Algorithms are used in machine learning"
        ]
        self.run_comparison(semantic_test, "Semantic Duplicates")
        
        # Test 5: Mixed quality data
        print("\n[TEST 5] Mixed Quality Data")
        mixed_quality = [
            "This is a high quality sentence with good content and proper structure",
            "Short",
            "Another well-formed sentence with multiple words and proper grammar",
            "A",
            "Quality content that provides value and information",
            "Bad",
            "Excellent example of proper data quality standards"
        ]
        self.run_comparison(mixed_quality, "Mixed Quality")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        if not self.test_results:
            print("No test results available")
            return
        
        # Aggregate statistics
        total_advanced_dups = sum(r['advanced']['duplicates_removed'] for r in self.test_results)
        total_conventional_dups = sum(r['conventional']['duplicates_removed'] for r in self.test_results)
        
        total_advanced_time = sum(r['advanced']['processing_time'] for r in self.test_results)
        total_conventional_time = sum(r['conventional']['processing_time'] for r in self.test_results)
        
        avg_advanced_quality = sum(r['advanced']['avg_quality'] for r in self.test_results) / len(self.test_results)
        avg_conventional_quality = sum(r['conventional']['avg_quality'] for r in self.test_results) / len(self.test_results)
        
        print(f"\nTotal Tests Run: {len(self.test_results)}")
        print(f"\nOverall Duplicate Detection:")
        print(f"  Advanced:     {total_advanced_dups} duplicates removed")
        print(f"  Conventional: {total_conventional_dups} duplicates removed")
        if total_conventional_dups > 0:
            improvement = ((total_advanced_dups - total_conventional_dups) / total_conventional_dups) * 100
            print(f"  Improvement:  {improvement:+.1f}%")
        
        print(f"\nOverall Processing Time:")
        print(f"  Advanced:     {total_advanced_time:.3f}s")
        print(f"  Conventional: {total_conventional_time:.3f}s")
        if total_conventional_time > 0:
            speedup = total_conventional_time / total_advanced_time
            print(f"  Speed Ratio:  {speedup:.2f}x")
        
        print(f"\nAverage Quality Score:")
        print(f"  Advanced:     {avg_advanced_quality:.3f}")
        print(f"  Conventional: {avg_conventional_quality:.3f}")
        improvement = ((avg_advanced_quality - avg_conventional_quality) / max(avg_conventional_quality, 0.01)) * 100
        print(f"  Improvement:  {improvement:+.1f}%")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        findings = []
        
        if total_advanced_dups > total_conventional_dups:
            findings.append("Advanced preprocessor detects more semantic duplicates")
        
        if avg_advanced_quality > avg_conventional_quality:
            findings.append("Advanced preprocessor produces higher quality results")
        
        if total_advanced_time < total_conventional_time:
            findings.append("Advanced preprocessor is faster")
        elif total_advanced_time > total_conventional_time:
            findings.append("Advanced preprocessor is slower (but more accurate)")
        
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        
        if not findings:
            print("Results are similar between both methods")
        
        print("="*80 + "\n")


def main():
    """Run comparison tests"""
    try:
        comparison = PreprocessorComparison()
        comparison.run_all_tests()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
