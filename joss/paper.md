---
title: 'dLux: Differentiable Physical Optics in Jax'
tags:
  - Python
  - Physical Optics
  - Telescopes
  - Exoplanet Science

authors:
  - name: Louis Desdoigts
    orcid: 0000-0002-1015-9029
    affiliation: 1 
    corresponding: true
  - name: Benjamin J. S. Pope
    affiliation: "2,3"
    orcid: 0000-0003-2595-9114
  - name: Jordan Dennis 
    affiliation: 2
    orcid: 0000-0001-8125-6494
  - name: Adam K. Taras
    orcid: 0000-0002-4558-2222
    affiliation: 1 
  - name: Max Charles
    orcid: 0009-0003-5950-4828
    affiliation: 1
affiliations:
 - name: School of Physics, University of Sydney, Camperdown, NSW 2006, Australia
   index: 1
 - name: School of Mathematics and Physics, University of Queensland, St Lucia, QLD 4072, Australia
   index: 2
 - name: Centre for Astrophysics, University of Southern Queensland, West Street, Toowoomba, QLD 4350, Australia
   index: 3
date: 22 Jul 2024
bibliography: paper.bib
---

# Summary

<!-- what is dLux -->
In this paper we introduce `dLux`[^dlux], an open-source Python package for differentiable physical optics simulation. Leveraging `jax` [@jax] for automatic differentiation and vectorization, it deploys natively on CPU, GPU, and parallelized HPC environments. `dLux` can perform Fourier and Fresnel optical simulations using matrix and FFT based propagation [@Soummer2007], as well as simulate linear and nonlinear detector effects. In published work so far, `dLux` has been used to demonstrate inference of pixel sensitivities jointly with optical aberrations in imaging data [@Desdoigts2023] and to demonstrate principled optimal experimental design of a telescope by direct optimization of the Fisher Information Matrix [@Desdoigts2024]. 

<!-- something about zodiax? -->

# Statement of need

<!-- describe problem and relevant citations -->
One of the foundational problems in optical astronomy is that of imaging scenes at resolutions close to the diffraction limit of a telescope. One of the most stringent cases for high-dynamic-range, high-resolution imaging is exoplanet direct imaging [@Follette2023], whether with adaptive optics systems on large telescopes on Earth [@Guyon2018], or with space-based imagers such as the James Webb Space Telescope coronagraphs [@Boccaletti2022 ; @Girard2022] and interferometer [@Sivaramakrishnan2023]. In each case, a central issue is in accurately modelling the point spread function (PSF) of the telescope: the diffraction pattern by which light from a point source is spread out over the detector, which is affected by wavelength-scale irregularities at each optical surface the light encounters, and which can drown out the signals of faint planets and circumstellar material. 

While there are many data-driven approaches to nonparametrically inferring and subtracting this PSF [@Cantalloube2021], the motivation for our work here is to use principled deterministic physics to model optical systems; to perform high-dimensional inferences from data, jointly about telescopes and the scenes they observe; to train neural networks to model electronics together with optics; and to produce principled, high-dimensional designs for telescope hardware. These problems necessitate a physical optics model which is fast and *differentiable*, so as to permit high-dimensional optimization by gradient descent [eg in `optax`; @optax] or sampling with Hamiltonian Monte Carlo or similar gradient-based algorithms [@Betancourt2017]. Physical optics is the study of the wave physics of light, and is taken to be separate from geometric optics, which is the approximation treating light as rays. A physical optics model is usually taken to simulate the propagation of light between one or more pupil planes and focal planes by the Fraunhofer or Fresnel approximation, based around Fourier transform calculations. 

<!-- describe what has to happen in physical optics etc -->
Non-differentiable open-source physical optics packages used in astronomy include `poppy` [@poppy], `prysm` [@prysm]; differentiable alternatives to `dLux` used in astronomy so far include `WaveDiff` [@Liaudat2023] and recent versions of `hcipy` [@hcipy].

<!-- dLux is open source: briefly explain its use -->
We introduce a new open-source physical optics package, `dLux` (named for taking *partial derivatives of light*), written in Python and using `jax`. It inherits an object oriented framework from `equinox` [@kidger2021equinox], around which we build a thin wrapper `zodiax`[^zodiax] to allow for a more efficient syntax building probabilistic models in `numpyro` [@numpyro].

<!-- alternative packages outside of astronomy -->
Similar approaches using differentiable optical models have been applied in the `DeepOptics` project [@Sitzmann2018]; `WaveBlocks` [@Page2020] in microscopy; `dO` [@Wang2022] for general cameras; and in `WaveOpticsPropagation.jl` [@Wechsler24]. `dLux` similarly leverages the strengths of differentiable simulation, however a focus on generic physical optics modules enables applications spanning domains and encompasses projects from the design to data processing stages.



# Documentation & Case Studies
<!-- briefly summarize tutorials  -->

In the accompanying [documentation](https://louisdesdoigts.github.io/dLux), we have produced several notebooks illustrating use cases...

- [example 1](https://louisdesdoigts.github.io/dLux/notebooks/whatever/);

Figures produced for... are shown in \autoref{fig1}. 
<!-- also ref joss_figure.py to make figure, or do similar with notebook, so that it is reproducible -->

![Figure Caption. \label{fig1}](joss_figure.png)

# Acknowledgements

<!-- whoever we acknowledge  -->

# References
<!-- you are only supposed to put refs in the .bib if they are actually used -->

[^dlux]: [https://github.com/louisdesdoigts/dLux](https://github.com/louisdesdoigts/dLux)
[^zodiax]: [https://github.com/louisdesdoigts/zodiax](https://github.com/louisdesdoigts/zodiax)

