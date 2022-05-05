# ∂Lux
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

∂Lux: Taking derivatives through Light - 'Optical systems as a Neural Network'

## Contributors 

[Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan Dennis](https://github.com/Jordan-Dennis)

## Installation

The easiest way to install is from PyPI: just use

`pip install dLux`

To install from source: clone this git repo, enter the directory, and run

`python setup.py install`

## Use

We are currently building examples and documentation! For now - see a [simple example](https://colab.research.google.com/drive/1Dz5NdRhtbGOzPl7jlIn5JvwNEQfaOq9Y?usp=sharing) in Google colab showing deployment onto GPU.

## History

∂Lux is a full from-scratch rewrite of the ideas [morphine](https://github.com/benjaminpope/morphine), which is a fork of the popular optical simulation package '[poppy](https://github.com/mperrin/poppy)' using the autodiff library [Google Jax](https://github.com/google/jax) to do _derivatives_. We have built it from the ground up in [equinox](https://github.com/patrick-kidger/equinox), a powerful object-oriented library in Jax, to best take advantage of new features and permit easy development and integration with neural networks.