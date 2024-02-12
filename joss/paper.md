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
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Benjamin J. S. Pope
    affiliation: "2,3"
    orcid: 0000-0003-2595-9114
  - name: Jordan Dennis 
    affiliation: 2
    orcid: 0000-0001-8125-6494
  - name: Adam K. Taras
    orcid: 0000-0002-4558-2222
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: School of Physics, University of Sydney, Camperdown, NSW 2006, Australia
   index: 1
 - name: School of Mathematics and Physics, University of Queensland, St Lucia, QLD 4072, Australia
   index: 2
 - name: Centre for Astrophysics, University of Southern Queensland, West Street, Toowoomba, QLD 4350, Australia
   index: 3
date: 9 Feb 2024
bibliography: paper.bib
---

# Summary

<!-- why physical optics in astronomy  -->

<!-- what is dLux -->
`dLux`[^dlux] is an first open-source Python package for physical optics simulation. Using `jax` [@jax] it is differentiable and deploys natively on CPU, GPU, and parallelized HPC environments. `dLux` can perform Fourier optical simulations using matrix and FFT based propagation, as well as simulate linear and nonlinear detector effects. 

<!-- more here -->

<!-- something about zodiax? -->

# Statement of need

<!-- describe problem and relevant citations -->

<!-- describe what has to happen in physical optics etc -->

<!-- alternative packages for astronomy: poppy, prysm, xaosim, hcipy, whatever liaudat has -->
Non-differentiable open-source physical optics packages used in astronomy include `poppy` [@poppy], `prysm` [@prysm], and in `WaveOpticsPropagation.jl`. By 

Differentiable alternatives to `dLux` used in astronomy so far include `WaveDiff` [@Liaudat2023] and recent versions of `hcipy` [@hcipy]...

<!-- alternative packages outside of astronomy -->
Similar approaches using differentiable optical models have been applied in the `DeepOptics` project [@Sitzmann2018] or `dO` [@Wang2022] for general cameras and `WaveBlocks` [@Page2020] in microscopy... etc. `dLux` similarly leverages the strengths of differentiable simulation, however a focus on generic physical optics modules enables applications spanning domains and encompasses projects from the design to data processing stages.

<!-- dLux is open source: briefly explain its use -->
We introduce a new open-source physical optics package, `dLux` (named for taking *partial derivatives of light*), written in Python and using `jax`. It inherits an object oriented framework from `equinox`... etc

- class or feature 1
- class or feature 2, etc

<!-- cite Desdoigts papers it has been used in  -->

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
