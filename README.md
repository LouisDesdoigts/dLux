![alt text](docs/assets/logo.jpg?raw=true)

# ∂Lux
## Differentiable Light - _Optical systems as a neural network_
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/dLux/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/dLux/)

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan
Dennis](https://github.com/Jordan-Dennis), [Peter Tuthill](https://github.com/ptuthill)

## What is ∂Lux?

∂Lux is a differentiable optical modelling framework built using [Jax](https://github.com/google/jax) for automatic differentiation and [Equinox](https://github.com/patrick-kidger/equinox) for object-oriented construction! It is the intellectual successor to [morphine](https://github.com/benjaminpope/morphine), which is a fork of the popular optical simulation package '[poppy](https://github.com/mperrin/poppy), and has been built on a simple idea: neural-networks are structurally isomorphic to optical systems, ie optical models are a mathematical *subset* of neural-networks. This means that we can build our optical models within machine learning frameworks and gain access to wildly powerful and underutilised tools, primarily automatic-differentiaion. Employing these tools allows for novel approaches to historically pathological problems within optics and astronomy.


## Installation

∂Lux is hosted on PyPI, so simply pip install!
```
pip install dLux
```

You can also build from source. To do so, clone the git repo, enter the directory, and run

```
pip install .
```

We encourage the creation of a virtual enironment to run dLux to prevent software conflicts as we keep the software up to date with the lastest version of the core packages.


## Use & Documentation

Documentation can be found [here](https://louisdesdoigts.github.io/dLux/). To get started look, go to the Tutorials section and have a look!

## Collaboration & Development

We are always looking to collaborate and further develop this software! We have focused on flexibility and ease of development, so if you have a project you want to use ∂Lux for, but it currently does not have the required capabilities, don't hesitate to [email me](louis.desdoigts@sydney.edu.au) and we can discuss how to implement and merge it! Similarly you can take a look at the `CONTRIBUTING.md` file.


## Windows/Google Colab Quickstart
`jaxlib` is currently not supported by the jax team on windows, however there are two work-arounds! 

Firstly [here](https://github.com/cloudhan/jax-windows-builder) is some community built software to install jax on windows! We do not use this ourselves so have limited knowledge, but some users seems to have got everyting working fine! 

Secondly users can also run our software on [Google Colab](https://research.google.com/colaboratory/). If you want to instal from source in colab, run this at the start of your notebook!
```
!git clone https://github.com/LouisDesdoigts/dLux.git # Download latest version
!cd dLux; pip install . -q # Navigate to ∂Lux and install from source
```

From here everything should work! You can also run the code on GPU to take full advantage of Jax, simply by switch to a GPU runtime environment, no extra steps necessary!


## Publications

We have a multitude of publications in the pipeline using dLux, some built from our tutorials. To start we would recommend looking at [this invited talk](https://louisdesdoigts.github.io/diff_optics/#/0/3) on ∂Lux which gives a good overview and has an attatched recording of it being presented! We also have [this poster!](https://spie.org/astronomical-telescopes-instrumentation/presentation/Optical-design-analysis-and-calibration-using-Lux/12180-160)

