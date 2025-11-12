from setuptools import setup, find_packages


setup(
    name="famus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scikit-learn>=1.3.0",
        "biopython>=1.78",
        "tqdm>=4.66.2",
        "matplotlib>=3.10.1",
        "seaborn>=0.13.2",
        "pytorch>=2.2.0",
    ],
    author="Guy Shur",
    author_email="guyshur@gmail.com",
    description="Functional Annotation Method Using Siamese neural networks (FAMUS)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/burstein-lab/famus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.11",
)
