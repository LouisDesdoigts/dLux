import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dLux",
    version="0.1",
    author="Louis Desdoigts",
    author_email="Louis.Desdoigts@sydney.edu.au",
    description="A fully differentiable optical simulator build in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisDesdoigts/DeLux",
    project_urls={
        "Bug Tracker": "https://github.com/LouisDesdoigts/DeLux/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6", # Find actual minimum requirement
)