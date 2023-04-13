# Overview

∂Lux has two modules, `dLux` and `dLux.utils`. The `dLux` repo contains the core functionality of ∂Lux, while `dLux.utils` contains various functions that are used throughout the package.

The `dLux` module is composed of 10 distint scripts:

- `core.py`
- `optics.py`
- `apertures.py`
- `aberrations.py`
- `propagators.py`
- `wavefronts.py`
- `detectors.py`
- `sources.py`
- `spectrums.py`
- `observtions.py`

The `core.py` script contains the primarily classes that users will interact with and are generally populated with the rest of the classes in dLux. Most of these scripts contains one type of class, although a series of classes are related to each other. The core of dLux works by providing layers that modify the `Wavefront` class. Layers that modify the `Wavefront` class all inherit from the `OpticalLayer` class defined in the `optics.py` script, however all classes in `apertures.py`, `aberrations.py` and `propagators.py` inherit from the `OpticalLayer` class. All of the remaining classes are distinct from one another.

Lets get an overview of the different classes in dLux:

---

# Instrument:

Script: `core.py`

The `Instrument` class is the primary class that users will interact with. It is designed to hold all the other classes in dLux and automate their interactions. The `Instrument` class can hold the following classes:

- `Optics`
- `Detector`
- `Source` (Not a core class)
- `Observation` (Not a core class)

Lets have a look at these different classes and what they do.

# Optics:

Script: `core.py`

The main diffraction engine is the `Optics` class, which only has a single attribute `layers`. This attribute is stored as an `OrderedDict` and holds all the layers that modify the `Wavefront` class that all inherit from the `OpticalLayer` class which will be detailed later. While the `Optics` class is fully functional on its own, in general it is interacted with through the `Instrument` class.

# Detector:

Script: `core.py`

The `Detector` class mirrors the `Optics` class in that it only has a single attribute `layers` which is an `OrderedDict`. The layers arributes stores all classes that inherit from the `DetectorLayer` class which perform transformation on PSF's. 

---

# Observations:

Script: `oberservations.py`

The `Observation` class is designed to allow for fine-grained control over the modelling of the `Instrument` class. An example of this is to allow for dithering patterns modelled, or for observing a series of different sources independently.

---

# Wavefront classes:

Script: `wavefronts.py`

The `Wavefront` class represents a physical wavefront which is transformed and propagated through the `Optics` class. This class is generally not needed to be interacted with directly, unless you are creating a new `OpticalLayer` class. This class provides a wide range of methods that can perform calculations and transformations of the parameters of the wavefront.

---

# Optical Layers:

Scripts:

- `optics.py`
- `apertures.py`
- `aberrations.py`
- `propagators.py`

All layers that operate on the `Wavefront` class are known as `OpticalLayers`. Becuase there are many different operations that can be performed on the `Wavefront` class they are split up into four different scripts.

The `optics.py` class contains the main layers used to operate on the wavefront. They provide general classes used to create wavefronts, add phases and OPDs, titls, rotations etc. The `apertures.py` script is a module designed to generate most apertures that will be needed in a dynamic and differentiable way. The `aberrations.py` script contains classes that can be used to add aberrations (presently only Zernike aberrations) to the wavefront. The `propagators.py` script contains the classes that transform `Wavefronts` to and from pupil and focal planes.

---

# Sources:

Scripts:

- `sources.py`
- `spectrums.py`

The source classes are designed to provide parameterisations of light sources. Currently they are geared towards astronomical sources, however they can be used for other sources as well. The `Source` classes primarily control positions, fluxes and any resolved components. They require a `Spectrum` class which provides different way to parametrise the source spectrum.

---

All of these classes can be explored in more detail in the API documentation.