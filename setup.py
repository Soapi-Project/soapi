# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="soapi",
    version="0.1.0",
    description="Auto cleans csv data for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="",
    author="Team Soap",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(), 
    include_package_data=True,
    install_requires=[
        "numpy>=1.22", 
        "pandas>=1.3.5", 
        "pandas_profiling>=3.2.0",
        "scikit_learn>=0.21.3",
        "setuptools>=41.4.0"
    ]
)
