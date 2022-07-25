# ∂Lux Docs
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

∂Lux: Taking derivatives through Light - 'Optical systems as a Neural Network'

## Contributors 

[Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan Dennis](https://github.com/Jordan-Dennis)

## Installation

The easiest way to install is from PyPI: just use

`pip install dLux`

To install from source: clone this git repo, enter the directory, and run

`pip install .`

## What is ∂Lux?

∂Lux is a full from-scratch rewrite of the ideas [morphine](https://github.com/benjaminpope/morphine), which is a fork of the popular optical simulation package '[poppy](https://github.com/mperrin/poppy)' using the autodiff library [Google Jax](https://github.com/google/jax) to do _derivatives_. We have built it from the ground up in [equinox](https://github.com/patrick-kidger/equinox), a powerful object-oriented library in Jax, to best take advantage of GPU acceleration and multi-device parallelization, and permit easy development and integration with neural networks.

The goal of ∂Lux is to revolutionise the way in which optical modelling is approached. The mathematical equivalence betweeen neural networks and optical systems means that optical systems can be efficiently modelled and optimized with the same automatic differentiation libraries that power deep learning. Using autodiff, we can infer *many* parameters at once - millions in the case of neural networks - whether by gradient descent, or with a Bayesian treatment using Hamiltonian Monte Carlo. Autodiff means that directly optimising physics-based forwards models with millions of parameters is not only possible, but practical without requiring vast computation power. 

---

## Package Overview

∂Lux has been built to be as simple and easy as possible for end-users, without abstracting them away from the underlying computations. 

We are currently building examples and documentation! We currently have two tutorial notebooks, showing the basics of how to optimise simple and more complex models [here](https://github.com/LouisDesdoigts/dLux/tree/main/notebooks). Please note that this software is still under development and so is subject to change.

∂Lux is based on two main classes: the `OpticalSystem` and the layers. A ∂Lux optical model consists of a list of operations/transforms performed on the input wavefront, which is passed as an argument to the `OpticalSystem` class. Each transformation or operation is a single 'layer' in that list. For a very simple optical a typical list of layers would look something like this:

```
layers = [
  CreateWavefront(wf_npix, wf_size),
  TiltWavefront(),
  CircularrAperture(wf_npix),
  NormaliseWavefront(),
  MFT(det_npix, fl, det_pixsize)
]
```

This list of layers can then be turned into an optical system by calling `OpticalSystem(layers)`. We now have a fully differentiable optical model!
