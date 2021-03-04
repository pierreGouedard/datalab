"""Builds the datalab package from the datalab folder.

To do so run the command below in the root folder:
pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="datalab",
    version="1.0",
    packages=find_packages(),
    author="Pierre Gouedard",
    author_email="pierre.gouedard@alumni.epfl.ch",
    description="Package for Data science projects"
)
