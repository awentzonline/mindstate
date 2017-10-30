#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='mindstate',
    version='0.0.1',
    description='A toolkit for compressed-network-search-based models.',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/mindstate/',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'six',
        'redis',
    ]
)
