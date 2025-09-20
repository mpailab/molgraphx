from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent
readme = (this_dir / "DESCRIPTION.md").read_text(encoding="utf-8")

setup(
    name="molgraphx",
    version="0.1.0",
    author="Grigoriy Bokov",
    author_email="bokovgrigoriy@gmail.com",
    description="Symmetry-sensitive analysis of molecular graph neural network models",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mpailab/molgraphx",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.8",
    ],
    extras_require={
        # Heavy/optional deps that users may prefer to install themselves
        "torch": ["torch>=1.12"],
        "torch-geometric": ["torch-geometric>=2.3"],
        "rdkit": ["rdkit-pypi>=2022.9.5"],
        "dev": ["pytest>=7", "ruff>=0.4", "mypy>=1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    project_urls={
        "Source": "https://github.com/mpailab/molgraphx",
        "Tracker": "https://github.com/mpailab/molgraphx/issues",
    },
)
