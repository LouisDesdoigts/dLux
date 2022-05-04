import setuptools
import os
import codecs
import re

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

long_description = "Taking derivatives through Light"
    
here = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
# tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata']


setuptools.setup(
    name="dLux",
    version=find_version("src", "__init__.py"),
    description="A fully differentiable optical simulator build in Jax",
    long_description=long_description,
    # long_description_content_type="text/markdown",
    
    author="Louis Desdoigts",
    author_email="Louis.Desdoigts@sydney.edu.au",
    url="https://github.com/LouisDesdoigts/dLux",
    
    project_urls={
        "Bug Tracker": "https://github.com/LouisDesdoigts/dLux/issues",
    },
    
    package_dir={"dLux": "src"},
    packages=["dLux"],
    
    install_requires=install_requires,
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)