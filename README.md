# ∂Lux
[![PyPI version](https://badge.fury.io/py/dLux.svg)](https://badge.fury.io/py/dLux)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/dLux/actions/workflows/automated_tests.yml/badge.svg)](https://github.com/LouisDesdoigts/dLux/actions/workflows/tests.yml)

∂Lux: Taking derivatives through Light - 'Optical systems as a Neural Network'

## Contributors 

[Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Jordan Dennis](https://github.com/Jordan-Dennis)

## Installation

∂Lux is still a new package and so is continuously being developed. For that reason we suggest cloning from source!


> To install from source: clone this git repo, enter the directory, and run
>
>`pip install .`

You can also install from PyPI, but this version is updated less often.

> To install from PyPI run
>
>`pip install dLux`


## Use

Documentation can be found at: https://louisdesdoigts.github.io/dLux/
> Note the docs are mostly built but the package is continually being developed so they are subject to change!

We are currently building tutorial notebooks to help new users get started! We do have some demonstration notebooks that can be found [here!](https://github.com/LouisDesdoigts/dLux/tree/main/notebooks). These notebooks are designed to get users started with a few different uses cases of ∂Lux and its tool. 

> Note: You must first run the `basis_creation.ipynb` notebook to generate and save files that are used in the other notebooks. Then `GE_optimisation.ipynb` must be run to generate a pupil for use in all of the other notebooks.

<!--

## History

∂Lux is a full from-scratch rewrite of the ideas [morphine](https://github.com/benjaminpope/morphine), which is a fork of the popular optical simulation package '[poppy](https://github.com/mperrin/poppy)' using the autodiff library [Google Jax](https://github.com/google/jax) to do _derivatives_. We have built it from the ground up in [equinox](https://github.com/patrick-kidger/equinox), a powerful object-oriented library in Jax, to best take advantage of new features and permit easy development and integration with neural networks.

## The Basics

The goal of ∂Lux is to revolutionalise the way in which optical modelling is approached. We believe that the mathematical symmetry betweeen neural networks and optical systems means that the current state of optical modelling is stuck in the old ways, and that differentiable optical models that harness the power of automatic differention is imperative to pushing the bounds of what is possible. 

For the uninitiated, automatic differentaion (auto-diff) is the mathematical tool that underpins the revolution in machine learning. The power of auto-diff ultimately lies in its ability to divorce the time it takes to optimise a model from the number of parameters being optimised in that model. This represents a *fundamental paradigm shift* in the way in which problems can be approached. Much time and effort has been focused in the past on making problems in optical modelling computationally tracatable, forcing compromises on what is learnt. This is no longer the case, directly optimising physics-based forwards models with millions of parameters is not only possible, but practical without requiring vast computation power. 

We have built ∂Lux using Jax - googles numpy-like auto-diff library and Equinox. Together these two packages allow us to build an optical simulator that takes full advantage of the bleeding edge of computer science. For example each individual PSF calcualtion is natively performed in parallel across however many computational resources are available without any work from the end-user. Similarly these models can be compiled at run time into XLA without. 

TBC...


---

## Package Overview

∂Lux has been built to be as simple and easy as possible for end-users, without abstracting them away from the underlying computations. 

There are two main types of classes that form the foundation of ∂Lux, the `OpticalSystem()` and the layers. In order to construct a model of an optical system one simply defines the series of operations/transforms that is performed on the input wavefront in a list, which is passed as an argument to the `OpticalSystem()` class. Each transformation or operation is a single 'layer' in that list. For a very simple optical a typical list of layers would look something like this:

```
layers = [
  CreateWavefront(wf_npix, wf_size),
  TiltWavefront(),
  CircualrAperture(wf_npix),
  NormaliseWavefront(),
  MFT(det_npix, fl, det_pixsize)
]
```

This list of layers can then be turned into an optical system -> `OpticalSystem(layers)`. We now have a fully differentiable optical model!

The `OpticalSystem()` is the main class that we will interact with and does most of the heavy lifting, so lets a take a detailed look at what this class does.

---

# The `OpticalSystem()` object!

The OpticalSystem object is the primary object of dLux, so here is a quick overview.

> dLux curently does not check that inputs are correctly shaped/formatted in order to making things work appropriately (under development)

## Inputs:


### layers: list, required
 - A list of layers that defines the tranformaitons and operations of the system (typically optical)
 
### wavels: ndarray, optional
 - An array of wavelengths in meters to simulate
 - The shape must be 1d - stellar spectrums are controlled through the weights parameter
 - No default value is set if not provided and this will throw an error if you try to call functions that depend on this parameter
 - It is left as optional so that functions that allow wavelength input can be called on objects without having to pre-input wavelengths
 
### positions: ndarray, optional
 - An array of (x,y) stellar positions in units of radians, measured as deivation of the optical axis. 
 - Its input shape should be (Nstars, 2), defining an x, y position for each star. 
 - If not provided, the value defaults to (0, 0) - on axis

### fluxes: ndarray, optional
 - An array of stellar fluxes, its length must match the positions inputs size to work properly
 - Theoretically this has arbitrary units, but we think of it as photons
 - Defaults to 1 (ie, returning a unitary flux psf if not specified)

### weights: ndarray, optional
 - An array of stellar spectral weights (arb units)
 - This can take multiple shapes
     - Default is to weight all wavelengths equally (top-hat)
     - If a 1d array is provided this is applied to all stars, shape (Nwavels)
     - if a 2d array is provided each is applied to each individual star, shape (Nstars, Nwavels)
 - Note the inputs values are always normalised and will not directly change total output flux (inderectly it can change it by weighting more flux to wavelengths with more aperture losses, for example)

### dithers: ndarray, optional
 - An arary of (x, y) positional dithers in units of radians
 - Its input shape should be (Nims, 2), defining the (x,y) dither for each image
 - if not provided, defualts to no-dither

### detector_layers: list, optional
 - A second list of layer objects designed to allow processing of psfs, rather than wavefronts
 - It is applied to each image after psfs have been approraitely weighted and summed
     
     
## Functions:

### __call__()
> Primary call function applying all parameters of the scene object through the systems
 - Takes no inputs, returning a image, or array of images
 - The primary function designed to apply all of the inputs of the class in order to generate the appropriate output images
 - Automatically maps the psf calcualtion over both wavelength and input position for highly efficient calculations
 - It takes no inputs as to allow for easier coherent optimsation of the whole system 
 
### propagate_mono(wavel):
> Propagates a single monochromatic wavelength through only the layers list
 - Inputs:
     - wavel (float): The wavelength in meters to be modelled through the system
     - offset (ndarray, optional): the (x,y) offest from the optical axis in radians
 - Returns: A sigle monochromatic PSF
 
### propagate_single(wavels)
> Propagataes a single broadband stellar source through the layers list
 - Inputs:
     - wavels (ndarray): The wavelengths in meters to be modelled through the system
     - offset (ndarray, optional): the (x,y) offest from the optical axis in radians
     - weights (ndarray, optional): the realative weights of each wavelength, 
         - No normalisation is applied to the weights to allow user flexibility
         - Unitary weights will output a total sum of 1
 - Returns: A single broadband PSF
 
 
### debug_prop(wavels)
> Propagataes a single wavelength through while storing the intermediary value of the wavefront and pixelscale between each operation. This is designed to help build and debug unexpected behaviour. It is functionally a mirror of propagate_mono() that stored intermediary values/arrays
 - Inputs:
     - wavels (ndarray): The wavelengths in meters to be modelled through the system
     - offset (ndarray, optional): the (x,y) offest from the optical axis in radians
 - Returns: [A single monochromatic PSF, intermediate wavefront, intermediate pixelscales]
     
     
 -->

