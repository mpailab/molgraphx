import setuptools

setuptools.setup(
    name="molgraphx",
    version="0.0.1",
    author="Grigoriy Bokov",
    author_email="bokovgrigoriy@gmail.com",
    description="Symmetry-sensitive analysis of molecular graph neural network models",
    long_description="file: DESCRIPTION.md",
    long_description_content_type="text/markdown",
    url="https://github.com/mpailab/molgraphx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)