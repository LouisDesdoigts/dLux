# Overview

∂Lux has two modules, `dLux` and `dLux.utils`. The `dLux` repo contains the core functionality of ∂Lux, while `dLux.utils` contains various functions that are used throughout the package. 

In dLux there are a few different sets of classes:

## `Wavefronts` and classes that modify them

dLux at its core is a diffraction engine that models optical system via performing transformations on wavefronts. There are three main types of classes: `Wavefronts` (`wavefronts.py`) which represent the state of some monochromatic wavefront. `OpticalLayers` perform transformations on `Wavefronts` and `Optics` (`optics.py`) classes which hold a series of `OpticalLayers` in order to model some optical system.

The `OpticalLayers` classes are split up into four different scripts:

- `optical_layers.py` Which contain basic optics classes allowing for the modification of the amplitude, opd and phase of wavefronts, plus titls, rotations etc.
- `apertures.py` Which contain classes that model apertures dynamically. It is very extensive and allows for the modelling of most apertures.
- `aberrations.py` Which contain classes that model aberrations dynamically.
- `propagators.py` Which contain classes that perform the propagation of the wavefront.

## `Images` and classes that modify them

The `dLux` module also contains a series of classes that modify `Images` (`images.py`) which represent the state of some psf as it is transformed through a detector. The structure matches that of the `Wavefront` classes, with `DetectorLayers` (`detector_layers.py`) performing transformations on `Images` and `Detectors` (`detectors.py`) holding a series of `DetectorLayers` in order to model some detector.

## `Sources` and `Spectra`

The `dLux` module also contains a series of classes that represent sources and their spectra. The `Source` classes (`sources.py`) represent some parametric source, while the `Spectrum` classes (`spectrums.py`) represent the spectrum of the source.

## `Instrument`s and `Observation`s

The `Instrument` (`instruments.py`) is designed to coherently model the interaction between these different components and the `Observation` (`observations.py`) classes allow for fine-grained control over the modelling of the `Instrument` class. An example of this is to allow for dithering patterns modelled, or for observing the same source using different instrumental filters.