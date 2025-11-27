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
    affiliation: 2
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
 - name: School of Mathematical and Physical Sciences, 12 Wally's Walk, Macquarie University, Macquarie Park, NSW 2113, Australia
   index: 2
 - name: School of Mathematics and Physics, University of Queensland, St Lucia, QLD 4072, Australia
   index: 3
date: 27 Nov 2025
bibliography: paper.bib
---

# Summary

<!-- what is dLux -->
In this paper we introduce `dLux`[^dlux], an open-source Python package for differentiable physical optics simulation. Leveraging `jax` [@jax] for automatic differentiation and vectorization, it deploys natively on CPU, GPU, and parallelized HPC environments. `dLux` can perform Fourier and Fresnel optical simulations using matrix and FFT based propagation [@Soummer2007], as well as simulate linear and nonlinear detector effects. In published work so far, `dLux` has been used to demonstrate inference of pixel sensitivities jointly with optical aberrations in imaging data [@Desdoigts2023] and to demonstrate principled optimal experimental design of a telescope by direct optimization of the Fisher Information Matrix [@Desdoigts2024]; and to perform and end-to-end calibration of the James Webb Aperture Masking Interferometer [@Desdoigts2025amigo ; @Charles2025]. 

<!-- something about zodiax? -->

# Statement of need

<!-- describe problem and relevant citations -->
One of the foundational problems in optical astronomy is that of imaging scenes at resolutions close to the diffraction limit of a telescope. One of the most stringent cases for high-dynamic-range, high-resolution imaging is exoplanet direct imaging [@Follette2023], whether with adaptive optics systems on large telescopes on Earth [@Guyon2018], or with space-based imagers such as the James Webb Space Telescope coronagraphs [@Boccaletti2022 ; @Girard2022] and interferometer [@Sivaramakrishnan2023]. In each case, a central issue is in accurately modelling the point spread function (PSF) of the telescope: the diffraction pattern by which light from a point source is spread out over the detector, which is affected by wavelength-scale irregularities at each optical surface the light encounters, and which can drown out the signals of faint planets and circumstellar material. 

While there are many data-driven approaches to nonparametrically inferring and subtracting this PSF [@Cantalloube2021], the motivation for our work here is to use principled deterministic physics to model optical systems; to perform high-dimensional inferences from data, jointly about telescopes and the scenes they observe; to train neural networks to model electronics together with optics; and to produce principled, high-dimensional designs for telescope hardware. These problems necessitate a physical optics model which is fast and *differentiable*, so as to permit high-dimensional optimization by gradient descent [eg in `optax`; @optax] or sampling with Hamiltonian Monte Carlo or similar gradient-based algorithms [eg in `numpyro`; @numpyro]. Physical optics is the study of the wave physics of light, and is taken to be separate from geometric optics, which is the approximation treating light as rays. A physical optics model is usually taken to simulate the propagation of light between one or more pupil planes and focal planes by the Fraunhofer or Fresnel approximation, based around Fourier transform calculations. 

<!-- describe what has to happen in physical optics etc -->
Non-differentiable open-source physical optics packages used in astronomy include `poppy` [@poppy], `prysm` [@prysm]; differentiable alternatives to `dLux` used in astronomy so far include `WaveDiff` [@Liaudat2023] and recent versions of `hcipy` [@hcipy].

<!-- dLux is open source: briefly explain its use -->
We introduce a new open-source physical optics package, `dLux` (named for taking *partial derivatives of light*), written in Python and using `jax`. It inherits an object oriented framework from `equinox` [@kidger2021equinox], around which we build a thin wrapper `zodiax`[^zodiax] to allow for a more efficient syntax building probabilistic models in `numpyro` [@numpyro].

<!-- alternative packages outside of astronomy -->
Similar approaches using differentiable optical models have been applied in the `DeepOptics` project [@Sitzmann2018]; `WaveBlocks` [@Page2020] in microscopy; `dO` [@Wang2022] for general cameras; and in `WaveOpticsPropagation.jl` [@Wechsler24]. `dLux` similarly leverages the strengths of differentiable simulation, however a focus on generic physical optics modules enables applications spanning domains and encompasses projects from the design to data processing stages.


# Documentation & Case Studies
<!-- briefly summarize tutorials  -->

In the accompanying [documentation](https://louisdesdoigts.github.io/dLux), we have produced several pages illustrating use cases for `dLux`, incuding

- [phase retrieval in simulated data](https://louisdesdoigts.github.io/dLux/notebooks/whatever/);
- [binarization for design of a diffractive pupil](https://louisdesdoigts.github.io/dLux/tutorials/examples/designing_a_mask/)
- [Fisher-optimal design of a binary-valued diffractive pupil](https://louisdesdoigts.github.io/dLux/tutorials/examples/designing_a_mask/)
- [calibration of pixel sensitivities](https://louisdesdoigts.github.io/dLux/tutorials/examples/flatfield_calibration/)
- [inference of a binary star system with HMC in `numpyro`](https://louisdesdoigts.github.io/dLux/tutorials/examples/HMC/)

The core objects of `dLux` are implemented in `zodiax`, a lightweight wrapper for `equinox`. We have also [documented example uses](https://louisdesdoigts.github.io/zodiax/docs/usage/) of `zodiax` so that users can understand the deeper functionality available here.

Both the `dLux` and `zodiax` docs incorporate full API references and installation instructions.

# Acknowledgements

We acknowledge and pay respect to the traditional owners of the land on which the University of Sydney, Macquarie University, and University of Queensland are situated, upon whose unceded, sovereign, ancestral lands we work. We pay respects to their Ancestors and descendants, who continue cultural and spiritual connections to Country.

BP and PT have been supported by the Australian Research Council grant DP230101439 and BP by DE210101639; and LD and MC have been supported by the Australian Government Research Training Program (RTP) award. We are grateful to the Australian public for enabling this science. BP would like to thank the Big~Questions~Institute for their philanthropic support. Development of \dlux has been supported by the Breakthrough Foundation through their Toliman project as a part of the Breakthrough Watch initiative.

# References
<!-- you are only supposed to put refs in the .bib if they are actually used -->

[^dlux]: [https://github.com/louisdesdoigts/dLux](https://github.com/louisdesdoigts/dLux)
[^zodiax]: [https://github.com/louisdesdoigts/zodiax](https://github.com/louisdesdoigts/zodiax)

