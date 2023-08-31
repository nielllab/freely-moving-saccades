"""
setup.py

Setup file for saccadeAnalysis package.

Written by DMM, 2022
"""


import setuptools


setuptools.setup(
    name = 'saccadeAnalysis',
    packages = setuptools.find_packages(),
    description = 'Analysis for eye and head movements in freely moving mice.',
    author = 'DMM',
    version = 0.1,
)