from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Analytics',
    version='1.0',
    install_requires=["pandas", "tensorflow", "maxminddb-geolite2"],
    py_modules=["utilities", "data_preparation", "machine_learning", "anomaly_processing"]
)
