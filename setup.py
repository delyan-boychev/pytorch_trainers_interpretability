"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_namespace_packages

setup(
    name="pytorch_trainers_interpretability",
    author="Delyan Boychev",
    author_email="delyan.boychev05@gmail.com",
    url="https://github.com/delyan-boychev/pytorch_trainers_interpretability",
    description="",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", exclude=["tests"]),
    install_requires=[
        "torchvision>=0.13.1",
        "torch>=1.12.1",
        "tqdm",
        "numpy",
        "matplotlib",
        "timm",
    ],
)