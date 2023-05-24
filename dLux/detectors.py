"""
Detector: detector.py
=====================
This module contains the classes that define the behaviour of Detectors in
dLux.

There is one public class:
    - LayeredDetector

This essentially operates in the same way as `LayeredOptics`, taking in a list
of `DetectorLayers` and applying them sequentially. DetectorLayers operate on
`Image` classes. It has one main method, `.model(image)` that applies the
detector layers to the input image.

---

# Examples:

Lets have a look at how we can construct a `LayeredDetector` class and apply it
to some psf.

First we construct some optics and a source:

```python
import jax.numpy as np
import dLux as dl

# Define the optical parameters
wf_npixels = 256
diameter = 1 # meters
psf_npixels = 128
psf_pixel_scale = 0.1 # arcseconds
psf_oversample = 4

# Use ApertureFactory class to make a simple circular aperture
aperture = dl.ApertureFactory(wf_npixels)

# Construct the optics class
optics = dl.AngularOptics(wf_npixels, diameter, aperture, 
    psf_npixels, psf_pixel_scale, psf_oversample)

# Construct Source
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
source = dl.PointSource(wavelengths)
raw_psf = source.model(optics)
```

Now we construct our detector:

```python
# Construct Detector
detector = dl.LayeredDetector([
    dl.ApplyJitter(20),
    dl.IntegerDownsample(4),
    dl.AddConstant(1),
])

# Combine into instrument and model
instrument = dl.Instrument(optics, source, detector)
psf = instrument.model()
```

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Raw PSF")
plt.imshow(raw_psf**0.5)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Detector Transformed PSF")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('assets/detectors.png')
```

Full API
"""
from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
import jax.numpy as np
from jax import Array
from zodiax import Base
import dLux.utils as dlu
import dLux


__all__ = ["LayeredDetector"]


DetectorLayer = lambda : dLux.detector_layers.DetectorLayer


class BaseDetector(Base):
    
    @abstractmethod
    def model(self, image): # pragma: no cover
        pass

class LayeredDetector(BaseDetector):
    """
    A high level class desgined to model the behaviour of some detectors
    response to some psf.

    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input psf as it interacts with the detector.
    """
    layers : OrderedDict


    def __init__(self : Detector, layers : list) -> Instrument:
        """
        Constructor for the Detector class.

        Parameters
        ----------
        layers : list
            An list of dLux detector layer classes that define the instrumental
            effects for some detector.

            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entried can either be dLux DetectorLayers, or tuples of the
            form (DetectorLayer, key), with the key being used as the dictionary
            key for the layer.
        """
        self.layers = dlu.list_to_dictionary(layers, True, DetectorLayer())
        super().__init__()


    def __getattr__(self : Detector, key : str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the layers dictionary of this class via the 'class.attribute' method.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                 .format(type(self), key))


    def model(self : Detector, image: Array) -> Array:
        """
        Applied the stored detector layers to the input image.

        Parameters
        ----------
        image : Array
            The input psf to be transformed.

        Returns
        -------
        image : Array
            The ouput 'image' after being transformed by the detector layers.
        """
        for key, layer in self.layers.items():
            image = layer(image)
        return image.image