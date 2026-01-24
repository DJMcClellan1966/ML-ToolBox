"""
Setup script for building Cython extensions
Similar to scikit-learn's setup.py
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path

# Get include directories
numpy_include = numpy.get_include()

# Define extensions
extensions = [
    # Data preprocessing extensions
    Extension(
        "ml_toolbox.compartment1_data._preprocessors",
        ["ml_toolbox/compartment1_data/_preprocessors.pyx"],
        include_dirs=[numpy_include],
        extra_compile_args=['-O3', '-march=native'],  # Optimize
        language="c"
    ),
    
    # Algorithm extensions
    Extension(
        "ml_toolbox.compartment3_algorithms._classifiers",
        ["ml_toolbox/compartment3_algorithms/_classifiers.pyx"],
        include_dirs=[numpy_include],
        extra_compile_args=['-O3', '-march=native'],
        language="c"
    ),
    
    Extension(
        "ml_toolbox.compartment3_algorithms._regressors",
        ["ml_toolbox/compartment3_algorithms/_regressors.pyx"],
        include_dirs=[numpy_include],
        extra_compile_args=['-O3', '-march=native'],
        language="c"
    ),
    
    Extension(
        "ml_toolbox.compartment3_algorithms._clustering",
        ["ml_toolbox/compartment3_algorithms/_clustering.pyx"],
        include_dirs=[numpy_include],
        extra_compile_args=['-O3', '-march=native'],
        language="c"
    ),
]

setup(
    name='ml_toolbox',
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,  # Faster, but less safe
            'wraparound': False,   # Faster indexing
            'initializedcheck': False,  # Faster
            'cdivision': True,     # Faster division
        }
    ),
    zip_safe=False,
)
