from setuptools import setup, find_namespace_packages

setup(
    name="building_reconstruction",
    version="25.11.04",
    packages=find_namespace_packages(include=["Tree_Segmentation*"]),
)