![alt text](docs/assets/logo.jpg?raw=true)

# ∂Lux
## Differentiable Light - _Optical systems as a neural network_
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/dLux/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/dLux/)

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan
Dennis](https://github.com/Jordan-Dennis), [Peter Tuthill](https://github.com/petertuthill)

## Installation

∂Lux is continuously being developed & improved. For that reason we suggest cloning the development version from source. To do so, clone this git repo, enter the directory, and run

```
pip install .
```
You can also install from PyPI, but this version is not updated very often and so may not be well represented in the documentation.

```
pip install dLux
```

## Use & Documentation

Documentation can be found at [here](https://louisdesdoigts.github.io/dLux/), and is under continuous development. To get started look at the [tutorial notebooks!](https://louisdesdoigts.github.io/dLux/notebooks/phase_retrieval_demo/)

## Collaboration & Development

We are always looking to collaborate and further develop this software! We have focused on flexibility and ease of development, so if you have a project you want to use ∂Lux for, but it currently does not have the required capabilities, don't hesitate to email me and we can discuss how to implement and merge it! louis.desdoigts@sydney.edu.au


## Windows/Google Colab Quickstart
`jaxlib` can be problematic on windows so we suggest users run our software on [Google Colab](https://research.google.com/colaboratory/).
There are a few extra steps to get setup. At the top of each colab file you will need
```
!git clone https://github.com/LouisDesdoigts/dLux.git # Download latest version
!cd dLux; pip install . -q # Navigate to ∂Lux and install from source
```

From here everything should work! You can also run the code on GPU to take full advantage of Jax, simply by switch to a GPU runtime environment, no extra steps necessary!


## Publications

We have a multitude of publications in the pipeline using dLux, however we do have a poster about this software available to view [here!](https://spie.org/astronomical-telescopes-instrumentation/presentation/Optical-design-analysis-and-calibration-using-Lux/12180-160)