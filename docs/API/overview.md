# Overview

∂Lux has two modules, `dLux` and `dLux.utils`. The `dLux` repo contains the core functionality of ∂Lux, while `dLux.utils` contains various functions that are used throughout the package. 

∂Lux contains a few different sets of classes:

---

## `Wavefronts` and classes that modify them
[comment]: <> (Rephrase?)
∂Lux, at its core, is a diffraction engine that models optical systems by performing transformations on wavefronts. There are three main types of classes: `Wavefronts` (`wavefronts.py`) which represent the state of some monochromatic wavefront. `OpticalLayers` perform transformations on `Wavefronts` and `Optics` (`optics.py`) classes which hold a series of `OpticalLayers` in order to model some optical system.

The `OpticalLayers` classes are split into four different scripts:

- `optical_layers.py`, containing basic optics classes allowing for the modification of the amplitude, OPD and phase of wavefronts (plus tilts, rotations, etc.).
- `apertures.py`, containing classes that model apertures dynamically. It is extensive and allows for the modelling of most apertures.
- `aberrations.py`, containing classes that model aberrations dynamically.
- `propagators.py`, containing classes that perform the wavefront propagation.

---

## `Images` and classes that modify them
[comment]: <> (Rephrase?)
The `dLux` module also contains a series of classes that modify `Images` (`images.py`), which represent the state of some PSF as it is transformed through a detector. The structure matches that of the `Wavefront` classes, with `DetectorLayers` (`detector_layers.py`) performing transformations on `Images` and `Detectors` (`detectors.py`) holding a series of `DetectorLayers` in order to model some detector.

---

## `Sources` and `Spectra`
[comment]: <> (Is the object called `Spectra` or `spectrums`)
The `dLux` module also contains a series of classes that represent sources and their spectra. The `Source` classes (`sources.py`) represent some parametric source, while the `Spectrum` classes (`spectrums.py`) represent the spectrum of the source.

---

## `Instrument` and `Observation` objects
[comment]: <> (Rephrase last sentence?)
The `Instrument` (`instruments.py`) is designed to coherently model the interaction between these different components and the `Observation` (`observations.py`) classes allow for fine-grained control over the modelling of the `Instrument` class. An example of this is to allow for dithering patterns modelled, or for observing the same source using different instrumental filters.