![dLux logo](https://raw.githubusercontent.com/LouisDesdoigts/dLux/main/docs/assets/logo.jpg)

# ∂Lux

[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LouisDesdoigts/dLux/graph/badge.svg)](https://codecov.io/gh/LouisDesdoigts/dLux)
[![Documentation](https://github.com/LouisDesdoigts/dLux/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/dLux/)

dLux is an open-source differentiable optical modelling framework built within the [Jax](https://github.com/jax-ml/jax)+[Equinox](https://github.com/patrick-kidger/equinox)+[Zodiax](https://github.com/LouisDesdoigts/zodiax) ecosystem. All models are differentiable, GPU accelerated, and interface with the optimisation and statistical tools provided by these libraries. It is designed to be incredibly fast, flexible, and extensible, serving as a framework to solve real-world optical problems within, rather than just another optical simulator. It was built for the modelling of astronomical telescopes, but the tools it provides and its flexible construction make it applicable to optical problems well beyond this domain.

Under the hood, dLux takes inspiration from the isomorphic nature between optical system and neural networks, enabling the description of arbitrary optical systems as a series of layers that operate sequentially on a wavefront. This 'layers based' design provides near-complete flexibility to end users (since you can easily define your own layer!), while also enabling most optical systems to be composed from layers that are already implemented within dLux. Unlike other optical simulators, dLux aims to put tools in users hands and teach them how to use them, not provide a pre-built system that works in unknown ways. This formulation puts more power in the hands of users and also helps them learn optical modelling along the way!

## Documentation & Tutorials

dLux has extensive docummentation and a growing set of tutorials to help users get up and running. The documentation includes a detailed API reference, as well as guides on how to use the various features of dLux. The tutorials cover a range of topics, from basic usage to more advanced techniques, and are designed to help users learn how to use dLux effectively.

Documentation: [https://louisdesdoigts.github.io/dLux/](https://louisdesdoigts.github.io/dLux/)

The tutorials can be found in the documentation, but are also hosted on the [dLux tutorials](https://github.com/LouisDesdoigts/dLux_tutorials) repository, where the notebooks can be downloaded and run directly. This also enables users to easily contribute their own tutorials to the repository, which we encourage!


## Installation

dLux is hosted on PyPI and can be installed via pip: ```pip install dLux```

To run dLux on a GPU, simply install a compatible Jax GPU version.

Requires: Python 3.10+, Zodiax 0.5+

## Other Projects

dLux has a number of downstream and linked projects:

 - [Zodiax](https://github.com/LouisDesdoigts/zodiax): A differentiable physical modelling framework for general scientific programming, providing the core tools and utilities that dLux is built on.
 - [abcdLux](https://github.com/LouisDesdoigts/abcdLux): abcdLux provides differentiable Fresnel propagation for optical systems, and is presently being integrated into dLux as the core propagation engine.
 - [Amigo](https://github.com/LouisDesdoigts/Amigo): Amigo is the best-performing data calibration and analysis pipeline for the JWST interferometer, and is the recommended tool for the analysis of JWST AMI data.
 - [Dorito](https://github.com/maxecharles/dorito): Dorito is a high-performance image reconstruction algorithm for the JWST AMI, built on top of Amigo. 
 - [dLuxToliman](https://github.com/maxecharles/dLuxToliman): dLuxToliman is a high-fidelity simulator for the Toliman space telescope, built on top of dLux and designed to be used for the design and analysis of the Toliman mission.

If you have any other projects that use dLux and would like to be added to this list, please let us know!

## Publications

 - [Deep Calibration of Flat Field and Phase Retrieval with Automatic Differentiation](https://arxiv.org/abs/2406.08703)
 - [Optical Design Maximising Fisher Information](https://arxiv.org/abs/2406.08704)
 - [Amigo: a Data-Driven Calibration of the JWST Interferometer](https://arxiv.org/abs/2510.09806)
 - [Image reconstruction with the JWST Interferometer](https://arxiv.org/abs/2510.10924)

If you have any other papers that use dLux and would like to be added to this list, please let us know!


## Collaboration & Development

dLux is open source so we are always open to feedback, suggestions, and pull-requests! More details about contributing can be found in our [contributing guide](CONTRIBUTING.md).

## Citation

If you use dLux in your research, please cite
[Differentiable optics with dLux: I--deep calibration of flat field and phase
retrieval with automatic differentiation](https://doi.org/10.1117/1.JATIS.9.2.028007):

```bibtex
@article{desdoigts2023differentiable,
  author = {Desdoigts, Louis and Pope, Benjamin J. S. and Dennis, Jordan and Tuthill, Peter G.},
  title = {Differentiable optics with {$\partial$Lux}: I--deep calibration of flat field and phase retrieval with automatic differentiation},
  journal = {Journal of Astronomical Telescopes, Instruments, and Systems},
  volume = {9},
  number = {2},
  pages = {028007},
  year = {2023},
  publisher = {SPIE},
  doi = {10.1117/1.JATIS.9.2.028007}
}
```
