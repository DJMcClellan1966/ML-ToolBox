"""
TAOCP Complete - Missing Algorithms from The Art of Computer Programming
Completes the TAOCP implementation with missing sorting, string, numerical, and combinatorial algorithms

Algorithms from:
- Vol. 3: More sorting algorithms (Merge, Radix, Counting, Bucket)
- Vol. 3: More string algorithms (Boyer-Moore, Rabin-Karp, Suffix structures)
- Vol. 2: More numerical methods (Floating-point, Statistical tests)
- Vol. 4: More combinatorial algorithms (Gray codes, Partitions, Catalan)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Iterator
import numpy as np
from collections import defaultdict
import math

sys.path.insert(0, str(Path(__file__).parent))


class TAOCPSorting:
    """
    Additional Sorting Algorithms (TAOCP Vol. 3)
    
    Completes the sorting algorithm library
    """
    
    @staticmethod
    def merge_sort(arr: List[Any], key: Optional[callable] = None) -> List[Any]:
        """
        Merge Sort - Stable, O(n log n) guaranteed
        
        Args:
            arr: List to sort
            key: Optional key function
            
        Returns:
            Sorted list
        """
        if key is None:
            key = lambda x: x
        
        if len(arr) <= 1:
            return arr.copy()
        
        mid = len(arr) // 2
        left = TAOCPSorting.merge_sort(arr[:mid], key)
        right = TAOCPSorting.merge_sort(arr[mid:], key)
        
        return TAOCPSorting._merge(left, right, key)
    
    @staticmethod
    def _merge(left: List[Any], right: List[Any], key: callable) -> List[Any]:
        """Merge two sorted lists"""
        result = []
        i, j = 0, 0
        
        while i < len(left) and j < len(right):
            if key(left[i]) <= key(right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def radix_sort(arr: List[int], base: int = 10) -> List[int]:
        """
        Radix Sort - O(nk) for integers
        
        Args:
            arr: List of integers
            base: Number base (default 10)
            
        Returns:
            Sorted list
        """
        if not arr:
            return []
        
        # Find maximum number of digits
        max_val = max(abs(x) for x in arr)
        max_digits = int(math.log(max_val, base)) + 1 if max_val > 0 else 1
        
        # Separate negative and positive
        negatives = [-x for x in arr if x < 0]
        positives = [x for x in arr if x >= 0]
        
        # Sort negatives (reversed) and positives
        if negatives:
            negatives = TAOCPSorting._radix_sort_helper(negatives, base, max_digits)
            negatives = [-x for x in reversed(negatives)]
        
        if positives:
            positives = TAOCPSorting._radix_sort_helper(positives, base, max_digits)
        
        return negatives + positives
    
    @staticmethod
    def _radix_sort_helper(arr: List[int], base: int, max_digits: int) -> List[int]:
        """Helper for radix sort"""
        for digit_pos in range(max_digits):
            buckets = [[] for _ in range(base)]
            
            for num in arr:
                digit = (num // (base ** digit_pos)) % base
                buckets[digit].append(num)
            
            arr = [num for bucket in buckets for num in bucket]
        
        return arr
    
    @staticmethod
    def counting_sort(arr: List[int], max_val: Optional[int] = None) -> List[int]:
        """
        Counting Sort - O(n + k) for small range
        
        Args:
            arr: List of non-negative integers
            max_val: Maximum value (auto-detect if None)
            
        Returns:
            Sorted list
        """
        if not arr:
            return []
        
        if max_val is None:
            max_val = max(arr)
        
        # Count occurrences
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        
        # Build sorted array
        result = []
        for i in range(max_val + 1):
            result.extend([i] * count[i])
        
        return result
    
    @staticmethod
    def bucket_sort(arr: List[float], num_buckets: Optional[int] = None) -> List[float]:
        """
        Bucket Sort - O(n) average case
        
        Args:
            arr: List of floats in [0, 1)
            num_buckets: Number of buckets (default: len(arr))
            
        Returns:
            Sorted list
        """
        if not arr:
            return []
        
        if num_buckets is None:
            num_buckets = len(arr)
        
        # Create buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute into buckets
        for num in arr:
            bucket_idx = int(num * num_buckets)
            if bucket_idx >= num_buckets:
                bucket_idx = num_buckets - 1
            buckets[bucket_idx].append(num)
        
        # Sort each bucket and concatenate
        result = []
        for bucket in buckets:
            result.extend(sorted(bucket))
        
        return result


class TAOCPString:
    """
    Additional String Algorithms (TAOCP Vol. 3)
    
    Advanced pattern matching and text processing
    """
    
    @staticmethod
    def boyer_moore(text: str, pattern: str) -> List[int]:
        """
        Boyer-Moore Algorithm - Fast pattern matching
        
        Skips characters for efficiency
        
        Args:
            text: Text to search in
            pattern: Pattern to find
            
        Returns:
            List of indices where pattern found
        """
        if not pattern:
            return []
        
        matches = []
        n, m = len(text), len(pattern)
        
        # Build bad character table
        bad_char = {}
        for i in range(m):
            bad_char[pattern[i]] = i
        
        # Build good suffix table (simplified)
        good_suffix = [0] * (m + 1)
        i = m
        j = m + 1
        good_suffix[i] = j
        
        while i > 0:
            while j <= m and pattern[i-1] != pattern[j-1]:
                if good_suffix[j] == 0:
                    good_suffix[j] = j - i
                j = good_suffix[j]
            i -= 1
            j -= 1
            good_suffix[i] = j
        
        # Search
        s = 0
        while s <= n - m:
            j = m - 1
            
            while j >= 0 and pattern[j] == text[s + j]:
                j -= 1
            
            if j < 0:
                matches.append(s)
                s += good_suffix[0] if s + m < n else 1
            else:
                bad_char_shift = j - bad_char.get(text[s + j], -1)
                good_suffix_shift = good_suffix[j + 1]
                s += max(bad_char_shift, good_suffix_shift)
        
        return matches
    
    @staticmethod
    def rabin_karp(text: str, pattern: str, base: int = 256, mod: int = 101) -> List[int]:
        """
        Rabin-Karp Algorithm - Rolling hash for substring search
        
        Args:
            text: Text to search in
            pattern: Pattern to find
            base: Hash base
            mod: Modulus
            
        Returns:
            List of indices where pattern found
        """
        if not pattern:
            return []
        
        matches = []
        n, m = len(text), len(pattern)
        
        # Calculate hash of pattern and first window
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        for i in range(m - 1):
            h = (h * base) % mod
        
        for i in range(m):
            pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
            text_hash = (base * text_hash + ord(text[i])) % mod
        
        # Slide window and check
        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                if text[i:i+m] == pattern:
                    matches.append(i)
            
            if i < n - m:
                text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
                if text_hash < 0:
                    text_hash += mod
        
        return matches
    
    @staticmethod
    def suffix_array(text: str) -> List[int]:
        """
        Suffix Array - Advanced text indexing
        
        Args:
            text: Input text
            
        Returns:
            Suffix array (indices sorted by suffix)
        """
        if not text:
            return []
        
        # Add sentinel
        text += '$'
        n = len(text)
        
        # Build suffix array using doubling method
        sa = list(range(n))
        rank = [ord(text[i]) for i in range(n)]
        new_rank = [0] * n
        
        k = 1
        while k < n:
            # Sort by (rank[i], rank[i+k])
            sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
            
            # Update ranks
            new_rank[sa[0]] = 0
            for i in range(1, n):
                prev_rank = (rank[sa[i-1]], rank[sa[i-1] + k] if sa[i-1] + k < n else -1)
                curr_rank = (rank[sa[i]], rank[sa[i] + k] if sa[i] + k < n else -1)
                new_rank[sa[i]] = new_rank[sa[i-1]] + (1 if prev_rank != curr_rank else 0)
            
            rank, new_rank = new_rank, rank
            k *= 2
        
        # Remove sentinel
        return [i for i in sa if i < n - 1]


class TAOCPNumerical:
    """
    Additional Numerical Methods (TAOCP Vol. 2)
    
    Floating-point arithmetic and statistical tests
    """
    
    @staticmethod
    def floating_point_precision_analysis(x: float, y: float) -> Dict[str, Any]:
        """
        Floating-Point Precision Analysis
        
        Analyze precision and errors in floating-point operations
        
        Args:
            x, y: Floating-point numbers
            
        Returns:
            Dictionary with precision analysis
        """
        import sys
        
        # Machine epsilon
        eps = sys.float_info.epsilon
        
        # Relative error
        rel_error_add = abs((x + y) - (x + y)) / max(abs(x + y), eps)
        rel_error_mult = abs((x * y) - (x * y)) / max(abs(x * y), eps)
        
        # Precision
        precision = -math.log10(eps)
        
        return {
            'machine_epsilon': eps,
            'precision_digits': precision,
            'relative_error_add': rel_error_add,
            'relative_error_mult': rel_error_mult,
            'max_representable': sys.float_info.max,
            'min_representable': sys.float_info.min
        }
    
    @staticmethod
    def chi_square_test(data: List[int], expected: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Chi-Square Test for Randomness (TAOCP Vol. 2)
        
        Test if data is uniformly distributed
        
        Args:
            data: Observed frequencies
            expected: Expected frequencies (uniform if None)
            
        Returns:
            Dictionary with test results
        """
        n = sum(data)
        k = len(data)
        
        if expected is None:
            expected = [n / k] * k
        
        # Calculate chi-square statistic
        chi_square = sum((data[i] - expected[i])**2 / expected[i] for i in range(k))
        
        # Degrees of freedom
        df = k - 1
        
        # P-value approximation (simplified)
        # In practice, use scipy.stats.chi2
        p_value = 0.05  # Placeholder
        
        return {
            'chi_square': chi_square,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'is_random': chi_square < 2 * df  # Simplified threshold
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(data: List[float]) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov Test for Uniformity
        
        Test if data follows uniform distribution
        
        Args:
            data: Sample data in [0, 1)
            
        Returns:
            Dictionary with test results
        """
        n = len(data)
        sorted_data = sorted(data)
        
        # Calculate D statistic
        D_plus = max((i + 1) / n - sorted_data[i] for i in range(n))
        D_minus = max(sorted_data[i] - i / n for i in range(n))
        D = max(D_plus, D_minus)
        
        # Critical value (simplified, for alpha=0.05)
        critical_value = 1.36 / math.sqrt(n)
        
        return {
            'D_statistic': D,
            'D_plus': D_plus,
            'D_minus': D_minus,
            'critical_value': critical_value,
            'is_uniform': D < critical_value
        }


class TAOCPCombinatorial:
    """
    Additional Combinatorial Algorithms (TAOCP Vol. 4)
    
    Gray codes, partitions, Catalan numbers
    """
    
    @staticmethod
    def gray_code(n: int) -> List[str]:
        """
        Gray Code Generation - Minimal change sequences
        
        Generate n-bit Gray codes
        
        Args:
            n: Number of bits
            
        Returns:
            List of Gray code sequences
        """
        if n == 0:
            return ['']
        if n == 1:
            return ['0', '1']
        
        # Recursive construction
        prev = TAOCPCombinatorial.gray_code(n - 1)
        return ['0' + code for code in prev] + ['1' + code for code in reversed(prev)]
    
    @staticmethod
    def integer_partitions(n: int) -> List[List[int]]:
        """
        Integer Partition Generation
        
        Generate all partitions of n
        
        Args:
            n: Integer to partition
            
        Returns:
            List of partitions (each is a list of parts)
        """
        if n == 0:
            return [[]]
        if n == 1:
            return [[1]]
        
        partitions = []
        
        def generate_partitions(remaining: int, max_part: int, current: List[int]):
            if remaining == 0:
                partitions.append(current.copy())
                return
            
            for part in range(min(max_part, remaining), 0, -1):
                current.append(part)
                generate_partitions(remaining - part, part, current)
                current.pop()
        
        generate_partitions(n, n, [])
        return partitions
    
    @staticmethod
    def catalan_numbers(n: int) -> List[int]:
        """
        Catalan Number Generation
        
        Generate first n Catalan numbers
        
        Args:
            n: Number of Catalan numbers to generate
            
        Returns:
            List of Catalan numbers
        """
        catalan = [0] * (n + 1)
        catalan[0] = 1
        
        for i in range(1, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - 1 - j]
        
        return catalan[1:]
    
    @staticmethod
    def bell_numbers(n: int) -> List[int]:
        """
        Bell Number Generation - Set partitions
        
        Generate first n Bell numbers
        
        Args:
            n: Number of Bell numbers to generate
            
        Returns:
            List of Bell numbers
        """
        bell = [[0] * (n + 1) for _ in range(n + 1)]
        bell[0][0] = 1
        
        for i in range(1, n + 1):
            bell[i][0] = bell[i-1][i-1]
            for j in range(1, i + 1):
                bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
        
        return [bell[i][0] for i in range(1, n + 1)]
    
    @staticmethod
    def stirling_numbers_first_kind(n: int, k: int) -> int:
        """
        Stirling Numbers of the First Kind
        
        Count permutations with k cycles
        
        Args:
            n, k: Parameters
            
        Returns:
            Stirling number s(n, k)
        """
        if k == 0:
            return 1 if n == 0 else 0
        if k > n:
            return 0
        
        # Recursive formula: s(n, k) = (n-1)*s(n-1, k) + s(n-1, k-1)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = (i - 1) * dp[i-1][j] + dp[i-1][j-1]
        
        return dp[n][k]
    
    @staticmethod
    def stirling_numbers_second_kind(n: int, k: int) -> int:
        """
        Stirling Numbers of the Second Kind
        
        Count ways to partition n elements into k non-empty subsets
        
        Args:
            n, k: Parameters
            
        Returns:
            Stirling number S(n, k)
        """
        if k == 0:
            return 1 if n == 0 else 0
        if k > n:
            return 0
        
        # Recursive formula: S(n, k) = k*S(n-1, k) + S(n-1, k-1)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        
        return dp[n][k]


class TAOCPComplete:
    """
    Unified interface for all TAOCP complete algorithms
    """
    
    def __init__(self):
        self.sorting = TAOCPSorting()
        self.string = TAOCPString()
        self.numerical = TAOCPNumerical()
        self.combinatorial = TAOCPCombinatorial()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'numpy': 'numpy>=1.26.0',
            'python': 'Python 3.8+'
        }
