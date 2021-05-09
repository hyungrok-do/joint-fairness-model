import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='joint-fairness-model',
    ext_modules=cythonize("models/solvers.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)