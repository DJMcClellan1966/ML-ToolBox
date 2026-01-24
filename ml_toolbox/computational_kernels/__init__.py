"""
Computational Kernels - Fortran/Julia-like Performance Without Those Languages

This module provides high-performance computational kernels that mimic the
performance characteristics of Fortran (vectorization, array operations)
and Julia (JIT compilation, modern numerical computing) without requiring
those languages.

Similar to quantum-inspired methods that provide quantum-like benefits
without actual quantum hardware.
"""

from .fortran_like_kernel import FortranLikeKernel
from .julia_like_kernel import JuliaLikeKernel
from .unified_computational_kernel import UnifiedComputationalKernel

__all__ = [
    'FortranLikeKernel',
    'JuliaLikeKernel',
    'UnifiedComputationalKernel',
]
