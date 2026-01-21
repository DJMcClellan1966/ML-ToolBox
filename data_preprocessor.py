"""
Advanced Data Preprocessor
Combines Quantum Kernel + PocketFence Kernel for superior data preprocessing
"""
import sys
from pathlib import Path
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import re
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from quantum_kernel import get_kernel, KernelConfig
import requests
import numpy as np


class AdvancedDataPreprocessor:
    """
    Advanced data preprocessor using Quantum Kernel + PocketFence Kernel
    
    Features:
    - Safety filtering (PocketFence)
    - Semantic deduplication (Quantum)
    - Intelligent categorization (Quantum)
    - Quality scoring (Quantum)
    - Standardization (Quantum)
    """
    
    def __init__(self, pocketfence_url: str = "http://localhost:5000", 
                 dedup_threshold: float = 0.9,
                 use_quantum: bool = True):
        self.quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True)) if use_quantum else None
        self.pocketfence_url = pocketfence_url
        self.dedup_threshold = dedup_threshold
        self.use_quantum = use_quantum
        self.pocketfence_available = self._check_pocketfence()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'unsafe_filtered': 0,
            'duplicates_removed': 0,
            'categories_created': 0,
            'processing_times': []
        }
    
    def _check_pocketfence(self) -> bool:
        """Check if PocketFence service is available"""
        try:
            response = requests.get(f"{self.pocketfence_url}/api/kernel/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            raw_data: List of raw text items to preprocess
            verbose: Print detailed progress
            
        Returns:
            Dictionary with preprocessing results
        """
        start_time = time.time()
        
        if verbose:
            print(f"[Preprocessing] Input: {len(raw_data)} items")
        
        results = {
            'original_count': len(raw_data),
            'safe_data': [],
            'unsafe_data': [],
            'deduplicated': [],
            'duplicates': [],
            'categorized': {},
            'quality_scores': [],
            'final_count': 0,
            'processing_time': 0.0,
            'stats': {}
        }
        
        # Stage 1: Safety filtering
        if verbose:
            print("[Stage 1] Safety Filtering (PocketFence Kernel)")
        safe_data, unsafe_data = self._safety_filter(raw_data, verbose)
        results['safe_data'] = safe_data
        results['unsafe_data'] = unsafe_data
        self.stats['unsafe_filtered'] += len(unsafe_data)
        
        # Stage 2: Semantic deduplication
        if verbose:
            print(f"[Stage 2] Semantic Deduplication (Quantum Kernel)")
        unique_data, duplicates = self._deduplicate_semantic(safe_data, verbose)
        results['deduplicated'] = unique_data
        results['duplicates'] = duplicates
        self.stats['duplicates_removed'] += len(duplicates)
        
        # Stage 3: Categorization
        if verbose:
            print("[Stage 3] Categorization (Quantum Kernel)")
        categorized = self._categorize(unique_data, verbose)
        results['categorized'] = categorized
        self.stats['categories_created'] = len(categorized)
        
        # Stage 4: Quality scoring
        if verbose:
            print("[Stage 4] Quality Scoring (Quantum Kernel)")
        quality_scores = self._quality_score(unique_data)
        results['quality_scores'] = quality_scores
        
        # Final results
        results['final_count'] = len(unique_data)
        results['processing_time'] = time.time() - start_time
        results['stats'] = {
            'unsafe_filtered': len(unsafe_data),
            'duplicates_removed': len(duplicates),
            'categories': len(categorized),
            'avg_quality': sum(s['score'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
        }
        
        self.stats['total_processed'] += len(raw_data)
        self.stats['processing_times'].append(results['processing_time'])
        
        return results
    
    def _safety_filter(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Stage 1: Filter unsafe content using PocketFence"""
        safe = []
        unsafe = []
        
        if not self.pocketfence_available:
            if verbose:
                print("  [Note] PocketFence service not available, skipping safety filter")
            return data, []
        
        for item in data:
            try:
                response = requests.post(
                    f"{self.pocketfence_url}/api/filter/content",
                    json={"content": item},
                    timeout=1
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get('isBlocked', False) or not result.get('isChildSafe', True):
                        unsafe.append(item)
                    else:
                        safe.append(item)
                else:
                    safe.append(item)
            except:
                safe.append(item)
        
        if verbose:
            print(f"  Safe: {len(safe)}, Unsafe: {len(unsafe)}")
        
        return safe, unsafe
    
    def _deduplicate_semantic(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Stage 2: Remove semantic duplicates using Quantum Kernel"""
        if not self.use_quantum or not self.quantum_kernel:
            # Fallback to exact matching
            return self._deduplicate_exact(data, verbose)
        
        unique = []
        duplicates = []
        seen_embeddings = []
        
        for item in data:
            embedding = self.quantum_kernel.embed(item)
            
            # Check similarity to seen items
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = float(np.abs(np.dot(embedding, seen_emb)))
                if similarity >= self.dedup_threshold:
                    duplicates.append(item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_embeddings.append(embedding)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _deduplicate_exact(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Fallback: Exact duplicate removal"""
        unique = []
        duplicates = []
        seen = set()
        
        for item in data:
            item_lower = item.lower().strip()
            if item_lower in seen:
                duplicates.append(item)
            else:
                unique.append(item)
                seen.add(item_lower)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _categorize(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Stage 3: Categorize by semantic similarity"""
        if not self.use_quantum or not self.quantum_kernel:
            return self._categorize_keyword(data, verbose)
        
        categories = defaultdict(list)
        
        # Define category examples
        category_examples = {
            'technical': ['programming', 'code', 'algorithm', 'software', 'development', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business', 'money'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support', 'troubleshoot'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education', 'study'],
            'general': ['hello', 'thanks', 'information', 'question', 'general']
        }
        
        for item in data:
            best_category = 'general'
            best_score = 0.0
            
            for category, examples in category_examples.items():
                similarities = [
                    self.quantum_kernel.similarity(item, example)
                    for example in examples
                ]
                avg_similarity = sum(similarities) / len(similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
            for cat, items in categories.items():
                print(f"    - {cat}: {len(items)} items")
        
        return dict(categories)
    
    def _categorize_keyword(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Fallback: Keyword-based categorization"""
        categories = defaultdict(list)
        
        category_keywords = {
            'technical': ['code', 'programming', 'algorithm', 'software', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education'],
            'general': []
        }
        
        for item in data:
            item_lower = item.lower()
            best_category = 'general'
            best_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for kw in keywords if kw in item_lower)
                if matches > best_matches:
                    best_matches = matches
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
        
        return dict(categories)
    
    def _quality_score(self, data: List[str]) -> List[Dict[str, Any]]:
        """Stage 4: Score data quality"""
        scored = []
        
        for item in data:
            length = len(item)
            word_count = len(item.split())
            
            # Length score
            if 20 <= length <= 500:
                length_score = 1.0
            elif length < 20:
                length_score = length / 20.0
            else:
                length_score = max(0.5, 1.0 - (length - 500) / 1000.0)
            
            # Completeness score
            completeness_score = min(word_count / 10.0, 1.0)
            
            # Combined quality
            quality = (length_score * 0.4 + completeness_score * 0.6)
            
            scored.append({
                'item': item,
                'score': quality,
                'length': length,
                'word_count': word_count
            })
        
        return scored
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            'total_processed': self.stats['total_processed'],
            'unsafe_filtered': self.stats['unsafe_filtered'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'categories_created': self.stats['categories_created'],
            'avg_processing_time': avg_time
        }


class ConventionalPreprocessor:
    """
    Conventional data preprocessor using standard methods
    
    Features:
    - Basic safety filtering (keyword-based)
    - Exact duplicate removal
    - Keyword-based categorization
    - Simple quality scoring
    """
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'unsafe_filtered': 0,
            'duplicates_removed': 0,
            'categories_created': 0,
            'processing_times': []
        }
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Conventional preprocessing pipeline"""
        start_time = time.time()
        
        if verbose:
            print(f"[Preprocessing] Input: {len(raw_data)} items")
        
        results = {
            'original_count': len(raw_data),
            'safe_data': [],
            'unsafe_data': [],
            'deduplicated': [],
            'duplicates': [],
            'categorized': {},
            'quality_scores': [],
            'final_count': 0,
            'processing_time': 0.0,
            'stats': {}
        }
        
        # Stage 1: Basic safety filtering
        if verbose:
            print("[Stage 1] Basic Safety Filtering")
        safe_data, unsafe_data = self._safety_filter(raw_data, verbose)
        results['safe_data'] = safe_data
        results['unsafe_data'] = unsafe_data
        self.stats['unsafe_filtered'] += len(unsafe_data)
        
        # Stage 2: Exact duplicate removal
        if verbose:
            print("[Stage 2] Exact Duplicate Removal")
        unique_data, duplicates = self._deduplicate_exact(safe_data, verbose)
        results['deduplicated'] = unique_data
        results['duplicates'] = duplicates
        self.stats['duplicates_removed'] += len(duplicates)
        
        # Stage 3: Keyword-based categorization
        if verbose:
            print("[Stage 3] Keyword-Based Categorization")
        categorized = self._categorize_keyword(unique_data, verbose)
        results['categorized'] = categorized
        self.stats['categories_created'] = len(categorized)
        
        # Stage 4: Simple quality scoring
        if verbose:
            print("[Stage 4] Simple Quality Scoring")
        quality_scores = self._quality_score(unique_data)
        results['quality_scores'] = quality_scores
        
        # Final results
        results['final_count'] = len(unique_data)
        results['processing_time'] = time.time() - start_time
        results['stats'] = {
            'unsafe_filtered': len(unsafe_data),
            'duplicates_removed': len(duplicates),
            'categories': len(categorized),
            'avg_quality': sum(s['score'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
        }
        
        self.stats['total_processed'] += len(raw_data)
        self.stats['processing_times'].append(results['processing_time'])
        
        return results
    
    def _safety_filter(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Basic keyword-based safety filtering"""
        unsafe_keywords = ['spam', 'scam', 'hack', 'virus']  # Simplified
        safe = []
        unsafe = []
        
        for item in data:
            item_lower = item.lower()
            if any(kw in item_lower for kw in unsafe_keywords):
                unsafe.append(item)
            else:
                safe.append(item)
        
        if verbose:
            print(f"  Safe: {len(safe)}, Unsafe: {len(unsafe)}")
        
        return safe, unsafe
    
    def _deduplicate_exact(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Exact duplicate removal"""
        unique = []
        duplicates = []
        seen = set()
        
        for item in data:
            item_lower = item.lower().strip()
            if item_lower in seen:
                duplicates.append(item)
            else:
                unique.append(item)
                seen.add(item_lower)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _categorize_keyword(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Keyword-based categorization"""
        categories = defaultdict(list)
        
        category_keywords = {
            'technical': ['code', 'programming', 'algorithm', 'software', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education'],
            'general': []
        }
        
        for item in data:
            item_lower = item.lower()
            best_category = 'general'
            best_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for kw in keywords if kw in item_lower)
                if matches > best_matches:
                    best_matches = matches
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
        
        return dict(categories)
    
    def _quality_score(self, data: List[str]) -> List[Dict[str, Any]]:
        """Simple quality scoring"""
        scored = []
        
        for item in data:
            length = len(item)
            word_count = len(item.split())
            
            # Simple scoring
            length_score = min(length / 100.0, 1.0)
            word_score = min(word_count / 10.0, 1.0)
            quality = (length_score + word_score) / 2.0
            
            scored.append({
                'item': item,
                'score': quality,
                'length': length,
                'word_count': word_count
            })
        
        return scored
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            'total_processed': self.stats['total_processed'],
            'unsafe_filtered': self.stats['unsafe_filtered'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'categories_created': self.stats['categories_created'],
            'avg_processing_time': avg_time
        }
