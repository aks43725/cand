# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cand',
    version='0.1.0',
    description='A Python package for generating high-quality candidate models',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Ching-Wei Cheng',
    author_email='aks43725@gmail.com',
    url='https://github.com/aks43725/cand',
    license=license,
    packages=find_packages(exclude=('docs'))
)

