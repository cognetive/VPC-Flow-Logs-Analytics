from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Analytics',
    version='1.0',  # Required
    setup_requires=['tensorflow'],
    py_modules=["Utilities"],
)
