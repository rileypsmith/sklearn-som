import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="sklearn-som",
    version="1.1.0",
    description="A simple planar self organizing map",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rileypsmith/sklearn-som",
    author="Riley Smith",
    author_email="",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["sklearn_som"],
    include_package_data=True,
    install_requires=["numpy"],
)
