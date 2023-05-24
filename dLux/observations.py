"""
Observations: observations.py
=============================
This module contains the classes that define the behaviour of observations in
dLux. Observations classes are designed to be constructed by users in order to
model arbitrary observation patterns. As an example we have implemented a
`Dither` class that applies a series of dithers to the source positions.

All `Observation` classes shoudl implement a `.model(instrument)` method that
performs the actual calculation of the observation.

Lets have a look how to construct a simple dither observation class.

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

# Construct Observation
dithers = 1e-6 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
observation = dl.Dither(dithers)

# Construct the instrument and observe
instrument = dl.Instrument(optics, source, observation=observation)
psfs = instrument.observe()
```

Plotting code box:
```python
plt.figure(figsize=(20, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.title("$\sqrt{PSF}$")
    plt.imshow(psfs[i]**0.5)
    plt.colorbar()
plt.savefig('assets/observation.png')
```

Full API
"""
from __future__ import annotations
from abc import abstractmethod
from zodiax import Base
from jax.tree_util import tree_map
import jax.numpy as np
from jax import vmap, Array
from equinox import tree_at
import dLux


__all__ = ["Dither"]


Instrument = lambda : dLux.core.BaseInstrument


class BaseObservation(Base):
    """
    Abstract base class for observations. All observations should inherit from
    this class and must implement an `.model()` method that only takes in a
    single instance of `dLux.Instrument`.
    """

    @abstractmethod
    def model(self       : BaseObservation, 
              instrument : Instrument()): # pragma: no cover
        """
        Abstract method for the observation function.
        """


class Dither(BaseObservation):
    """
    Observation class designed to apply a series of dithers to the insturment
    and return the corresponding psfs.

    Attributes
    ----------
    dithers : Array, (radians)
        The array of dithers to apply to the source positions. The shape of the
        array should be (ndithers, 2) where ndithers is the number of dithers
        and the second dimension is the (x, y) dither in radians.
    """
    dithers : Array


    def __init__(self : Dither, dithers : Array):
        """
        Constructor for the Dither class.

        Parameters
        ----------
        dithers : Array, (radians)
            The array of dithers to apply to the source positions. The shape of
            the array should be (ndithers, 2) where ndithers is the number of
            dithers and the second dimension is the (x, y) dither in radians.
        """
        super().__init__()
        self.dithers = np.asarray(dithers, float)
        if self.dithers.ndim != 2 or self.dithers.shape[1] != 2:
            raise ValueError("dithers must be an array of shape (ndithers, 2)")


    def dither_position(self       : Dither, 
                        instrument : Instrument, 
                        dither     : Array) -> Instrument:
        """
        Dithers the position of the source objects by dither.

        Parameters
        ----------
        dither : Array, radians
            The (x, y) dither to apply to the source positions.

        Returns
        -------
        instrument : Instrument
            The instrument with the sources dithered.
        """
        # Define the dither function
        dither_fn = lambda source: source.add('position', dither)

        # Map the dithers across the sources
        dithered_sources = tree_map(dither_fn, instrument.sources, \
            is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

        # Apply updates
        return tree_at(lambda instrument: instrument.sources, instrument, 
            dithered_sources)


    def model(self       : Dither,
              instrument : Instrument,
              *args, **kwargs) -> Array:
        """
        Applies a series of dithers to the instrument sources and calls the
        .model() method after applying each dither.

        Parameters
        ----------
        instrument : Instrument
            The array of dithers to apply to the source positions.

        Returns
        -------
        psfs : Array
            The psfs generated after applying the dithers to the source
            positions.
        """
        dith_fn = lambda dither: self.dither_position(instrument, 
            dither).model(*args, **kwargs)
        return vmap(dith_fn, 0)(self.dithers)