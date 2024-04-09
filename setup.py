from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='KMeansCython',
    ext_modules=cythonize("kmeans_cy.pyx"),
    include_dirs=[numpy.get_include()]
)
