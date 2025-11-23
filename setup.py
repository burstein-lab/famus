from setuptools import setup, find_packages

setup(
    name="famus",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4,<2.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "biopython>=1.78",
        "tqdm>=4.66.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.2",
        "pyyaml>=5.0",
        # NOTE: PyTorch deliberately excluded - users install based on their system
    ],
    entry_points={
        "console_scripts": [
            "famus-train=famus.cli.train:main",
            "famus-classify=famus.cli.classify:main",
            "famus-convert-sdf=famus.cli.convert_sdf:main",
            "famus-install=famus.cli.install_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "famus": ["qmafft"],
    },
    author="Guy Shur",
    author_email="guyshur@gmail.com",
    description="Functional Annotation Method Using Siamese neural networks (FAMUS)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/burstein-lab/famus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
)
