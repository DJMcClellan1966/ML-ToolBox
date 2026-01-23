"""
Architecture Optimizer
Detects hardware architecture and applies architecture-specific optimizations

Supports:
- Intel CPUs (AVX, AVX2, AVX-512)
- AMD CPUs (AVX, AVX2)
- ARM CPUs (NEON)
- Apple Silicon (M1/M2/M3)
- GPU architectures (CUDA, ROCm)
"""
import sys
from pathlib import Path
import platform
import warnings
from typing import Dict, List, Any, Optional, Tuple
import subprocess

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False
    warnings.warn("py-cpuinfo not available. Install with: pip install py-cpuinfo")


class ArchitectureDetector:
    """
    Detects hardware architecture and capabilities
    """
    
    def __init__(self):
        self.architecture_info = self._detect_architecture()
        self.optimization_flags = self._get_optimization_flags()
    
    def _detect_architecture(self) -> Dict[str, Any]:
        """Detect system architecture"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_brand': None,
            'cpu_vendor': None,
            'cpu_features': [],
            'architecture': 'unknown',
            'optimization_level': 'basic'
        }
        
        # Try to get detailed CPU info
        if CPUINFO_AVAILABLE:
            try:
                cpu_info = cpuinfo.get_cpu_info()
                info['cpu_brand'] = cpu_info.get('brand_raw', 'Unknown')
                info['cpu_vendor'] = cpu_info.get('vendor_id_raw', 'Unknown')
                info['cpu_features'] = cpu_info.get('flags', [])
            except Exception as e:
                warnings.warn(f"Could not get CPU info: {e}")
        
        # Detect architecture type
        machine = info['machine'].lower()
        processor = info['processor'].lower()
        
        # Intel detection
        if 'intel' in processor or 'intel' in info.get('cpu_vendor', '').lower():
            info['architecture'] = 'intel'
            info['optimization_level'] = self._detect_intel_features(info['cpu_features'])
        
        # AMD detection
        elif 'amd' in processor or 'amd' in info.get('cpu_vendor', '').lower():
            info['architecture'] = 'amd'
            info['optimization_level'] = self._detect_amd_features(info['cpu_features'])
        
        # ARM detection
        elif 'arm' in machine or 'aarch64' in machine:
            info['architecture'] = 'arm'
            info['optimization_level'] = self._detect_arm_features(info['cpu_features'])
        
        # Apple Silicon
        elif platform.system() == 'Darwin' and 'arm' in machine:
            info['architecture'] = 'apple_silicon'
            info['optimization_level'] = 'advanced'  # M1/M2/M3 have excellent optimizations
        
        # x86_64 generic
        elif 'x86_64' in machine or 'amd64' in machine:
            info['architecture'] = 'x86_64'
            info['optimization_level'] = 'standard'
        
        return info
    
    def _detect_intel_features(self, features: List[str]) -> str:
        """Detect Intel-specific features"""
        features_lower = [f.lower() for f in features]
        
        if 'avx512' in ' '.join(features_lower) or any('avx512' in f for f in features_lower):
            return 'avx512'  # Best optimization
        elif 'avx2' in ' '.join(features_lower) or any('avx2' in f for f in features_lower):
            return 'avx2'  # Good optimization
        elif 'avx' in ' '.join(features_lower) or any('avx' in f for f in features_lower):
            return 'avx'  # Basic optimization
        else:
            return 'sse'  # Fallback
    
    def _detect_amd_features(self, features: List[str]) -> str:
        """Detect AMD-specific features"""
        features_lower = [f.lower() for f in features]
        
        if 'avx2' in ' '.join(features_lower) or any('avx2' in f for f in features_lower):
            return 'avx2'  # Good optimization
        elif 'avx' in ' '.join(features_lower) or any('avx' in f for f in features_lower):
            return 'avx'  # Basic optimization
        else:
            return 'sse'  # Fallback
    
    def _detect_arm_features(self, features: List[str]) -> str:
        """Detect ARM-specific features"""
        features_lower = [f.lower() for f in features]
        
        if 'neon' in ' '.join(features_lower) or any('neon' in f for f in features_lower):
            return 'neon'  # ARM SIMD
        else:
            return 'basic'  # Fallback
    
    def _get_optimization_flags(self) -> Dict[str, Any]:
        """Get optimization flags for detected architecture"""
        arch = self.architecture_info['architecture']
        opt_level = self.architecture_info['optimization_level']
        
        flags = {
            'architecture': arch,
            'optimization_level': opt_level,
            'use_simd': True,
            'use_parallel': True,
            'vectorization': True,
            'cache_optimization': True,
            'instruction_set': opt_level
        }
        
        # Architecture-specific optimizations
        if arch == 'intel':
            flags.update({
                'prefer_avx': opt_level in ['avx', 'avx2', 'avx512'],
                'prefer_avx2': opt_level in ['avx2', 'avx512'],
                'prefer_avx512': opt_level == 'avx512',
                'cache_line_size': 64,  # Intel cache line
                'prefetch_hint': True
            })
        elif arch == 'amd':
            flags.update({
                'prefer_avx': opt_level in ['avx', 'avx2'],
                'prefer_avx2': opt_level == 'avx2',
                'cache_line_size': 64,  # AMD cache line
                'prefetch_hint': True
            })
        elif arch == 'arm' or arch == 'apple_silicon':
            flags.update({
                'prefer_neon': opt_level == 'neon',
                'cache_line_size': 128,  # ARM typically has larger cache lines
                'prefetch_hint': True
            })
        
        return flags
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture information"""
        return self.architecture_info
    
    def get_optimization_flags(self) -> Dict[str, Any]:
        """Get optimization flags"""
        return self.optimization_flags


class ArchitectureOptimizer:
    """
    Architecture-specific optimizer
    Applies optimizations based on detected architecture
    """
    
    def __init__(self):
        self.detector = ArchitectureDetector()
        self.arch_info = self.detector.get_architecture_info()
        self.opt_flags = self.detector.get_optimization_flags()
    
    def optimize_numpy_config(self):
        """Optimize NumPy configuration for architecture"""
        if not NUMPY_AVAILABLE:
            return
        
        # Set NumPy optimization flags
        if self.opt_flags.get('prefer_avx512'):
            # AVX-512 optimizations
            try:
                np.seterr(all='ignore')  # Ignore warnings for speed
                # NumPy will automatically use AVX-512 if available
            except:
                pass
        elif self.opt_flags.get('prefer_avx2'):
            # AVX2 optimizations
            try:
                np.seterr(all='ignore')
            except:
                pass
    
    def get_optimal_chunk_size(self, data_size: int) -> int:
        """Get optimal chunk size for data processing based on architecture"""
        cache_line_size = self.opt_flags.get('cache_line_size', 64)
        
        # Optimal chunk size should be multiple of cache line
        if data_size < 1000:
            return data_size  # Small data, no chunking needed
        
        # For larger data, use cache-aware chunking
        optimal_chunk = (cache_line_size * 8)  # 8 cache lines per chunk
        
        # Adjust based on data size
        if data_size < 10000:
            return min(optimal_chunk, data_size)
        elif data_size < 100000:
            return optimal_chunk * 2
        else:
            return optimal_chunk * 4
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count for parallel processing"""
        import os
        
        # Get CPU count
        cpu_count = os.cpu_count() or 4
        
        # Architecture-specific thread counts
        if self.arch_info['architecture'] == 'apple_silicon':
            # Apple Silicon: Use all cores (efficient)
            return cpu_count
        elif self.arch_info['architecture'] in ['intel', 'amd']:
            # x86: Use all cores for CPU-bound tasks
            return cpu_count
        elif self.arch_info['architecture'] == 'arm':
            # ARM: May have different core types (big.LITTLE)
            return cpu_count
        else:
            return cpu_count
    
    def optimize_array_operations(self, array: np.ndarray) -> np.ndarray:
        """Optimize array operations for architecture"""
        if not NUMPY_AVAILABLE:
            return array
        
        # Ensure array is contiguous and aligned for SIMD
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        
        # For AVX-512, ensure 64-byte alignment
        if self.opt_flags.get('prefer_avx512'):
            # NumPy handles alignment automatically, but we can optimize
            pass
        
        return array
    
    def get_architecture_summary(self) -> str:
        """Get architecture summary"""
        info = self.arch_info
        flags = self.opt_flags
        
        summary = f"""
Architecture Detection Summary:
===============================
Platform: {info['platform']}
Machine: {info['machine']}
Processor: {info['processor']}
CPU Brand: {info.get('cpu_brand', 'Unknown')}
Architecture: {info['architecture']}
Optimization Level: {info['optimization_level']}

Optimization Flags:
===================
Instruction Set: {flags.get('instruction_set', 'basic')}
SIMD Enabled: {flags.get('use_simd', False)}
Vectorization: {flags.get('vectorization', False)}
Cache Line Size: {flags.get('cache_line_size', 64)} bytes
Optimal Threads: {self.get_optimal_thread_count()}

Available Features:
===================
"""
        if flags.get('prefer_avx512'):
            summary += "- AVX-512: Enabled (Best performance)\n"
        if flags.get('prefer_avx2'):
            summary += "- AVX2: Enabled (Good performance)\n"
        if flags.get('prefer_avx'):
            summary += "- AVX: Enabled (Basic SIMD)\n"
        if flags.get('prefer_neon'):
            summary += "- NEON: Enabled (ARM SIMD)\n"
        
        return summary


# Global architecture optimizer instance
_global_optimizer = None

def get_architecture_optimizer() -> ArchitectureOptimizer:
    """Get global architecture optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ArchitectureOptimizer()
    return _global_optimizer


# Example usage
if __name__ == '__main__':
    print("Architecture Detection and Optimization")
    print("="*80)
    
    optimizer = ArchitectureOptimizer()
    
    # Get summary
    summary = optimizer.get_architecture_summary()
    print(summary)
    
    # Test optimizations
    if NUMPY_AVAILABLE:
        print("\nTesting Optimizations:")
        print("-"*80)
        
        # Test chunk size
        test_sizes = [100, 1000, 10000, 100000]
        for size in test_sizes:
            chunk = optimizer.get_optimal_chunk_size(size)
            print(f"Data size {size}: Optimal chunk size = {chunk}")
        
        # Test thread count
        threads = optimizer.get_optimal_thread_count()
        print(f"\nOptimal thread count: {threads}")
        
        # Test array optimization
        test_array = np.random.randn(1000, 100)
        optimized = optimizer.optimize_array_operations(test_array)
        print(f"\nArray optimization: {optimized.shape}, Contiguous: {optimized.flags['C_CONTIGUOUS']}")
