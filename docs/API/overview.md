# Overview

∂Lux has two modules, `dLux` and `dLux.utils`. The `dLux` repo contains the core functionality of ∂Lux, while `dLux.utils` contains various functions that are used throughout the package. 

∂Lux contains a few different sets of classes:

---

## `Wavefront` objects and classes that modify them
∂Lux, at its core, is a diffraction engine that models optical systems by performing transformations on wavefronts. There are three main types of classes: Wavefronts, Optical Layers and Optics. `Wavefront` objects (`wavefronts.py`) represent the state of some monochromatic wavefront. `OpticalLayer` classes perform transformations on `Wavefront` objects. Finally, `Optics` classes (`optics.py`) hold a series of Optical Layers in order to model an optical system.

The `OpticalLayers` classes are split into four different scripts:

- `optical_layers.py`, containing basic optics classes allowing for the modification of the amplitude, OPD and phase of wavefronts (plus tilts, rotations, etc.).
- `apertures.py`, containing classes that model apertures dynamically. It is extensive and allows for the modelling of most apertures.
- `aberrations.py`, containing classes that model aberrations dynamically.
- `propagators.py`, containing classes that perform the wavefront propagation.

---

## `Image` objects and classes that modify them
The `dLux` module also contains a series of `Image` classes (`images.py`), which represent and modify the state of some PSF as it is transformed through a detector.
The structure of `Image` classes matches that of `Wavefront` classes; here, `DetectorLayer` classes (`detector_layers.py`) perform transformations on `Image` objects, and `Detector` classes (`detectors.py`) hold a series of Detector Layers in order to model some detector.

---

## `Source` and `Spectrum` objects
The `dLux` module also contains a series of classes that represent sources and their spectra.
The `Source` classes (`sources.py`) represent some parametric source, while the `Spectrum` classes (`spectra.py`) represent the source's spectrum.

---

## `Instrument` and `Observation` objects
The `Instrument` class (`instruments.py`) is designed to coherently model the interaction between these aforementioned components. 
The `Observation` classes (`observations.py`) allow for fine-grained control over the modelling of `Instrument` objects.
An example of this is the modelling of dithering patterns, or observing a source with different instrumental filters.