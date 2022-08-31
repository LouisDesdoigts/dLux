![alt text](assets/logo.jpg?raw=true)

# ∂Lux Docs
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)

∂Lux: differentiating light 

_Optical systems as neural networks_

## Contributors

[Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan Dennis](https://github.com/Jordan-Dennis)

## Installation

The easiest way to install is from PyPI: just use

`pip install dLux`

To install from source: clone this git repo, enter the directory, and run

`pip install .`

## What is ∂Lux?

∂Lux is a full from-scratch rewrite of the ideas [morphine](https://github.com/benjaminpope/morphine), which is a fork of the popular optical simulation package '[poppy](https://github.com/mperrin/poppy)' using the autodiff library [Google Jax](https://github.com/google/jax) to do _derivatives_. We have built it from the ground up in [equinox](https://github.com/patrick-kidger/equinox), a powerful object-oriented library in Jax, to best take advantage of GPU acceleration and multi-device parallelization, and permit easy development and integration with neural networks.

The mathematical equivalence betweeen neural networks and optical systems means that optical systems can be efficiently modelled and optimized with the same automatic differentiation libraries that power deep learning. Using autodiff, we can infer *many* parameters at once - millions in the case of neural networks - whether by gradient descent, or with a Bayesian treatment using Hamiltonian Monte Carlo. Autodiff means that directly optimising physics-based forwards models with millions of parameters is not only possible, but practical without requiring vast computation power.


∂Lux has been built to be as simple and easy as possible for end-users, without abstracting them away from the underlying computations.

We are currently building examples and documentation! We currently have three tutorial notebooks, showing examples of 

- [Phase Retrieval](notebooks/phase_retrieval_demo.ipynb), inferring Zernike coefficients of aberrations in a simple asymmetric pupil
- [Phase Mask Design](notebooks/designing_a_mask.ipynb), to optimize the gradient energy of a pupil for astrometry
- [Pixel Level Calibration](notebooks/flatfield_calibration.ipynb) of the interpixel sensitivity variations (flat field), simultaneously with phase retrieval and positions of stars
- [Fisher Information](notebooks/fisher_information.ipynb) matrix entropy optimisation of a pupil

Please note that this software is still under development and so is subject to change.

## Windows/Google Colab Quickstart
`jaxlib` can be problematic on windows so we suggest users run our software on [Google Colab](https://research.google.com/colaboratory/).
There are a few extra steps to get setup
1. At the top of each colab file you will need 
``` 
!git clone https://github.com/LouisDesdoigts/dLux.git # Download latest version
!cd dLux; pip install . -q # Navigate to ∂Lux and install from source
``` 

**Tips and Tricks**
- You can read/write data from your own drive using
```
from google.colab import drive
drive.mount('/content/drive')
```
- View the files using the left sidebar to navigate
- Colab GPU works with the notebooks as is, you just have to change the Runtime mode

## Publications

We have a multitude of publications in the pipeline using dLux, however we do have a poster about this software available to view [here!](https://spie.org/astronomical-telescopes-instrumentation/presentation/Optical-design-analysis-and-calibration-using-Lux/12180-160)

