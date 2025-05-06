from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="csdt",
    version="1.0.9",
    author="Çağla Mıdıklı, İlker Birbil, Doğanay Özese",
    author_email="midiklicagla@gmail.com",
    description="Custom Split Decision Tree",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sibirbil/CSDT.git",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.8.1",
        "numpy==1.26.0",
        "pandas==2.1.3",
        "graphviz",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6, <3.12",
)
