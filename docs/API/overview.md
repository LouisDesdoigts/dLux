# Overview

∂Lux has two modules, `dLux` and `dLux.utils`. The `dLux` repo contains the core functionality of ∂Lux, while `dLux.utils` contains various functions that are used throughout the package. 

∂Lux contains a few different sets of classes:

---

## `Wavefront` objects and classes that modify them
[comment]: <> (You say there are three main types of classes, but it is unclear what they are as you go on to list like 4 different ones and its not even clear whether ur still listing anymore. 2/10)
∂Lux, at its core, is a diffraction engine that models optical systems by performing transformations on wavefronts. There are three main types of classes: `Wavefronts` (`wavefronts.py`) which represent the state of some monochromatic wavefront. `OpticalLayers` perform transformations on `Wavefronts` and `Optics` (`optics.py`) classes which hold a series of `OpticalLayers` in order to model some optical system.

The `OpticalLayers` classes are split into four different scripts:

- `optical_layers.py`, containing basic optics classes allowing for the modification of the amplitude, OPD and phase of wavefronts (plus tilts, rotations, etc.).
- `apertures.py`, containing classes that model apertures dynamically. It is extensive and allows for the modelling of most apertures.
- `aberrations.py`, containing classes that model aberrations dynamically.
- `propagators.py`, containing classes that perform the wavefront propagation.

---

## `Image` objects and classes that modify them
[comment]: <> (Rephrase? kinda just hurt my brain, I think you said detector layers hold detector layers or idk small brain moment from me 6/10)
The `dLux` module also contains a series of classes that modify `Images` (`images.py`), which represent the state of some PSF as it is transformed through a detector.
The structure matches that of the `Wavefront` classes, with `DetectorLayers` (`detector_layers.py`) performing transformations on `Images` and `Detectors` (`detectors.py`) holding a series of `DetectorLayers` in order to model some detector.

---

## `Source` and `Spectrum` objects
[comment]: <> (chill dw 10/10 ez)
The `dLux` module also contains a series of classes that represent sources and their spectra.
The `Source` classes (`sources.py`) represent some parametric source, while the `Spectrum` classes (`spectra.py`) represent the source's spectrum.

---

## `Instrument` and `Observation` objects
[comment]: <> (To allow for the modelling of dither patterns? Not sure what you're getting at 5/10)
The `Instrument` (`instruments.py`) is designed to coherently model the interaction between these aforementioned components 
and the `Observation` (`observations.py`) classes allow for fine-grained control over the modelling of Instruments.
An example of this is to allow for dithering patterns modelled, or for observing the same source using different instrumental filters.