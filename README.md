![alt text](docs/assets/logo.jpg?raw=true)

# ∂Lux

[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/dLux/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/dLux/)

Differentiable Optical Models as _Parameterised Neural Networks_ in Jax using Zodiax

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan
Dennis](https://github.com/Jordan-Dennis), [Adam Taras](https://github.com/ataras2), [Peter Tuthill](https://github.com/ptuthill)

∂Lux is an open-source differentiable optical modelling framework harnessing the structural isomorphism between optical systems and neural networks, giving forwards models of optical system as a _parametric neural network_.

∂Lux is built in [Zodiax](https://github.com/LouisDesdoigts/zodiax) which is an open-source object-orineted [Jax](https://github.com/google/jax) framework built as an extension of [Equinox](https://github.com/patrick-kidger/equinox) for scientific programming. This framework allows for the creation of complex optical systems involving many planes, phase and amplitude screens in each, and propagate between them in the Fraunhofer or Fresnel regimes. This enables [fast phase retrieval](https://louisdesdoigts.github.io/dLux/notebooks/phase_retrieval_demo/), image deconvolution, and [hardware design in high dimensions](https://louisdesdoigts.github.io/dLux/notebooks/designing_a_mask/). Because ∂Lux models are fully differentiable, you can [optimize them by gradient descent over millions of parameters](https://louisdesdoigts.github.io/dLux/notebooks/flatfield_calibration/); or use [Hamiltonian Monte Carlo to accelerate MCMC sampling](https://louisdesdoigts.github.io/dLux/notebooks/HMC/). Our code is fully open-source under an MIT license, and we encourage you to use it and build on it to solve problems in astronomy and beyond.

The ∂Lux framwork is built in [Zodiax](https://github.com/LouisDesdoigts/zodiax), which gives it a deep range of capabilities from both [Jax](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox):

> - [Accelerated Numpy](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html): a Numpy like API that can run on GPU and TPU
>
> - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html): Allows for optimisation and inference in extremely high dimensional spaces
>
> - [Just-In-Time Compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html): Compliles code into XLA at runtime and optimising execution across hardware
>
> - [Automatic Vectorisation](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html): Allows for simple parallelism across hardware and asynchronys execution
>
> - [Object Oriented Jax](https://docs.kidger.site/equinox/all-of-equinox/): Allows for differentiable classes that are recognised as a valid _Jax_ type
>
> - [Inbuilt Neural Networks](https://docs.kidger.site/equinox/api/nn/linear/): Has pre-built neural network layers classes
>
> - [Path-Based Pytree Interface](docs/usage.md): Path based indexing allows for easy interfacing with large and highly nested physical models
>
> - [Leaf Manipulation Methods](docs/usage.md): Inbuilt methods allow for easy manipulation of Pytrees mirroring the _Jax_ Array API

For an overview of these capabilities and different optimisation methods in [Zodiax](https://github.com/LouisDesdoigts/zodiax), please go through this [Zodiax Tutorial](https://louisdesdoigts.github.io/zodiax/docs/usage/).

Doccumentataion: [https://louisdesdoigts.github.io/dLux/](https://louisdesdoigts.github.io/dLux/)

Installation: ```pip install dLux```

Requires: Python 3.8+, Jax 0.4.3+, Zodiax 0.3.1+

## Collaboration & Development

We are always looking to collaborate and further develop this software! We have focused on flexibility and ease of development, so if you have a project you want to use ∂Lux for, but it currently does not have the required capabilities, don't hesitate to [email me](louis.desdoigts@sydney.edu.au) and we can discuss how to implement and merge it! Similarly you can take a look at the `CONTRIBUTING.md` file.

## Publications

We have a multitude of publications in the pipeline using dLux, some built from our tutorials. To start we would recommend looking at [this invited talk](https://louisdesdoigts.github.io/diff_optics/#/0/3) on ∂Lux which gives a good overview and has an attatched recording of it being presented! We also have [this poster!](https://spie.org/astronomical-telescopes-instrumentation/presentation/Optical-design-analysis-and-calibration-using-Lux/12180-160)