"""
Optical Systems: optics.py
==========================
This module contains the classes that define the behaviour of optical systems
in dLux. The classes are designed to be as flexible as possible, allowing
users to easily create their own optical systems.

There are four public classes:
    - AngularOptics
    - CartesianOptics
    - FlexibleOptics
    - LayeredOptics

Optics classes store `OpticalLayers` that operatre on `Wavefronts`. 

There are two types of optics classes, Layered and non-Layered. Layered optics
classes take in a list of `OpticalLayers` and apply them sequentially to the
wavefront, giving users full control and flexibilty in the modelling of their
optical system. 

Non-Layered optics classes are designed to be simple and easy to use, taking in
a few parameters that define the behaviour of common optical system. We will
explore these mode in the Examples section.

All public optics classes have 3 main methods:

1. `.model(sources)`
> Models dLux Source objects through the optics.

2. `.propagate(wavelengths, offset, weights)`
> Models a polychromatic point source through the optics.

3. `.propagate_mono(wavelength, offset)`
> Propagates a monochromatic point source through the optics.

The `.propagate_mono` method is where the actual propagation of the wavefront
through the optics occurs, but the `.propagate` vectorises the calcluations
across wavelengths for efficiency.

---

# Examples:

We will start here with the non-layered optics classes as they are simpler than
the LayeredOptics class. 

---

## AngularOptics

To construct a `AngularOptics` class we need to define 5 things:
1. The number of pixels of the initial wavefront
2. The diameter of the initial wavefront
3. The aperture of the system
4. The number of pixels of the final PSF
5. The pixel scale of the final PSF

*TODO Add tip box in docs*
TIP: Most code in dLux is written in SI units, but this class breaks from this
convention, with `psf_pixel_scale` taken in units of arcseconds typical of
astronomical optical sysetms.

Tip: The 'aperture' parameter can also be supplied as an array, where it will
treated as a array of transmission values!

The following code snippet shows how to construct a simple optical system and
propagate a point source through it.

```python
import jax.numpy as np
import dLux as dl

# Define the parameters
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

# Propagate the wavelengths
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
psf = optics.propagate(wavelengths)
```

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture")
plt.imshow(optics.aperture.transmission)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("$\sqrt{PSF}$")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('angular_psf.png')
```

Add API

---

## CartesianOptics

The `CartesianOptics` class is very similar to the `AngularOptics` class, but
it also takes in a focal length, and the units of psf_pixel_scale is taken in 
microns.

The following code snippet shows how to construct a simple optical system and
propagate a point source through it.

```python
import jax.numpy as np
import dLux as dl

# Define the parameters
wf_npixels = 256
diameter = 1 # meters
focal_length = 2 # meters
psf_npixels = 128
psf_pixel_scale = 1 # microns
psf_oversample = 4

# Use ApertureFactory class to make a simple circular aperture
aperture = dl.ApertureFactory(wf_npixels)

# Construct the optics class
optics = dl.CartesianOptics(wf_npixels, diameter, aperture, focal_length,
    psf_npixels, psf_pixel_scale, psf_oversample)

# Propagate the wavelengths
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
psf = optics.propagate(wavelengths)
```

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture")
plt.imshow(optics.aperture.transmission)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("$\sqrt{PSF}$")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('cartesian_psf.png')
```

Add API

---

The `FlexibleOptics` class allows for the use of any `Propagator` class in
dLux. Lets have a look at how we can use this propagator to model a psf with a
fresnel defocus.

```python
import jax.numpy as np
import dLux as dl

# Define the parameters
wf_npixels = 256
diameter = 1 # meters
focal_length = 2 # meters
psf_npixels = 128
psf_pixel_scale = 0.25e-6 # meters
focal_shift = 2e-5 # meters

# Use ApertureFactory class to make a simple circular aperture
aperture = dl.ApertureFactory(wf_npixels)

# Construct a Fresnel Propagator
propagator = dl.FarFieldFresnel(psf_npixels, psf_pixel_scale, focal_length,
    focal_shift)

# Construct the optics class
optics = dl.FlexibleOptics(wf_npixels, diameter, aperture, propagator)

# Propagate the wavelengths
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
psf = optics.propagate(wavelengths)
```

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture")
plt.imshow(optics.aperture.transmission)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("$\sqrt{PSF}$")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('fresnel_psf.png')
```

Add API

---

## LayeredOptics

The Layered Optics class allows us to define a list of `OpticalLayers` that
operate on a wavefront. This allows us to model more complex optical systems
than the previous classes, while also allowing users to define their own
OpticalLayers! Look at the OpticalLayers documentation for more information.

Lets have a look at how we can use this class to model a simple optical system.

```python
import jax.numpy as np
import dLux as dl

# Define the parameters
wf_npixels = 256
diameter = 1 # meters
focal_length = 2 # meters
psf_npixels = 128
psf_pixel_scale = 0.25e-6 # meters

# Construct the list of optical layers
layers = [
    (dl.ApertureFactory(wf_npixels), 'aperture'),
    dl.MFT(psf_npixels, psf_pixel_scale, focal_length),
]

# Construct the optics class
optics = dl.LayeredOptics(wf_npixels, diameter, layers)

# Propagate the wavelengths
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
psf = optics.propagate(wavelengths)
```

Tip box:
Note that we can pass in a tuple of the form (OpticalLayer, key) to the
LayeredOptics class. OpticalLayers and transformed into an OrderedDict and this
key is then used for that layer. This allows us to access the layers in the
class via the `class.attribute` method, ie `optics.aperture`. This can be very
helpful when using zodaix methods!

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture")
plt.imshow(optics.aperture.transmission)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("$\sqrt{PSF}$")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('layered_psf.png')
```

Add API

---

# Building your own Optical System

# TODO: Add link to dLux-JWST and dLux-Toliman repos.
It can often be helpful to create your own Optical System class to provide a
more complex optical models. For example the dLux-JWST and dLux-Toliman repos
built by the dLux devs! dLux is designed to facilitate this, requiring users to
only implement two methods in order to have a class that entirely integrates
with the wider dLux ecosystem.

Note: dLux is built in [Zodiax](https://github.com/LouisDesdoigts/zodiax). If
you are unfamiliar, you should read 
[this tutorial](https://louisdesdoigts.github.io/zodiax/docs/usage/) before
this example.

Lets have a look at how we can create a very simple optical system from the
ground up with a pixel scale in units of arcseconds, and a tranmissive mask. To
do this we need to implement both the `__init__` and `propagate_mono` methods.
It will have 6 attributes: `wf_npixels`, `diameter`, `aperture`, `mask`,
`psf_pixel_scale`, `psf_npixels`.

```python
from jax import Array
import dLux.utils as dlu
import dLux

# We must inherit from the base optics class, `BaseOptics`. This will integrate
# our class with the rest of the dLux code.
class MyOptics(dLux.optics.BaseOptics):
    wf_npixels      : int
    diameter        : float
    aperture        : Array
    mask            : Array
    psf_pixel_scale : float
    psf_npixels     : int

    def __init__(self, wf_npixes, diameter, aperture, mask, psf_npixels, 
        psf_pixel_scale):
        '''Constructs the class'''
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.aperture = aperture
        self.mask = mask
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = psf_pixel_scale
    
    # Our propagate_mono must have the expected behaviour of dLux optics,
    # meaning it must take the same inputs (wavelength, offset, return_wf) and
    # return the same outputs (psf).
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):
        '''Propagates a monochromatic source'''

        # Construct our wavefront
        wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)

        # Tilt the wavefront
        wf = wf.tilt(offset)

        # We can use the `self` keyword to access the class attributes
        wf *= self.aperture

        # Normalise the wavefront
        wf = wf.normalise()

        # Apply the mask as an array of OPDs
        wf += self.mask

        # Propagate the wavefront, casting the units of pixel scale to radians
        pixel_scale = dlu.arcmin_to_rad(self.psf_pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale)

        # Return the PSF
        if return_wf:
            return wf
        else:
            return wf.psf
```

Tip: We can apply OpticalLayer and arrays to wavefronts using the `*=` operator.
This is equivalent to `wf = layer(wf)`.

Now we can use it as per usual! This class will now be recognised by all dLux
functions that take in an optics class, such as `Sources` and `Instruments`

```python
import jax.numpy as np
import jax.random as jr
import dLux as dl

# Define the parameters
wf_npixels = 128
diameter = 1 # meters
psf_npixels = 128
psf_pixel_scale = 5e-4 # arcminutes

# Use ApertureFactory class to make a simple circular aperture
aperture = dl.ApertureFactory(wf_npixels)

# Make a mask of random OPDs
mask = 4.e-7 * jr.normal(jr.PRNGKey(0), (wf_npixels, wf_npixels))

# Construct the optics class
optics = MyOptics(wf_npixels, diameter, aperture, mask, psf_npixels, 
    psf_pixel_scale)

# Propagate the wavelengths
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
psf = optics.propagate(wavelengths)
```

Plotting code box:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture")
plt.imshow(optics.aperture.transmission)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("$\sqrt{PSF}$")
plt.imshow(psf**0.5)
plt.colorbar()
plt.savefig('my_optics_psf.png')
```

---

## Private classes

To further facilitate the creation of custom optics classes, dLux provides a
few base classes that can be used to build your own optics classes. 

The private classes are as follows:
    - `BaseOptics`
    - `SimpleOptics`
    - `NonPropagatorOptics`
    - `AperturedOptics`

These classes are not intended to be used directly, but rather to be inherited
from and extended. Lets look at them one by one.

The `BaseOptics` class is the base class for all optics classes in dLux, so if
you inherit from the any of the `SimpleOptics, `NonPropagatorOptics`, or
`AperturedOptics` classes, you will automatically inherit from `BaseOptics`. It
implements the `model` and `proapgate` methods, and defines the `propagate_mono`
method as an abstract method. This means that any class that inherits from
`BaseOptics` must implement a `propagate_mono` method with the same signature
as the abstract method. This is the only requirement for a class to be
recognised as an optics class by dLux.

The `SimpleOptics` class simply adds the `wf_npixels` and `diameter` attributes,
along with a single method `_construct_wavefront(wavelength, offset)` that
constructs a wavefront of the correct size and width, and performs the initial
tilt from the `offset` value.

The `NonPropagatorOptics` class adds the `psf_npixels`, `psf_pixel_scale`, and
`psf_oversample` attributes, along with a `true_pixel_scale.

The `AperturedOptics` class adds the `aperture` and `mask` attributes, along
with a `_apply_aperture(wavelength, offset)` method that applies the aperture
and mask to the wavefront, calling the `_construct_wavefront` method under the
hood. Note that by default, the `mask` attribute is applied as a transmissive
mask if it supplied as an array.

Full API
"""
from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map, tree_flatten
from equinox import tree_at
from zodiax import Base
from typing import Union
import dLux.utils as dlu
import dLux


__all__ = ["AngularOptics", "CartesianOptics", "FlexibleOptics", 
    "LayeredOptics"]


# Alias classes for simplified type-checking
OpticalLayer  = lambda : dLux.optical_layers.OpticalLayer
Propagator    = lambda : dLux.propagators.Propagator
Source        = lambda : dLux.sources.BaseSource


#######################
### Private Classes ###
#######################
class BaseOptics(Base):
    """
    The Base Optics class that all optics classes inherit from. Can be used to
    create your own optics classes that will integrate seamlessly with the rest
    of dLux.

    This class implements three concrete methods and on abstract one. The
    concrete methods are `model(sources)`, which models dLux sources through
    the optics, `propagate(wavelengths, offset, weights)`, which propagates a
    polychromatic point source through the optics, and `__getattr__`, which
    allows for easy access to the attributes of the class.
    
    The abstract method is `propagate_mono(wavelength, offset, return_wf)`,
    which propagates a monochromatic point source through the optics. This
    is where the the actual operations on the wavefront are performed. This
    method must be implemented by any class that inherits from `BaseOptics`.
    """


    def __getattr__(self : BaseOptics, key : str) -> Any:
        """
        Accessor for attributes of the class to simplify zodiax paths by
        searching for parameters in the attirbutes of the class.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the class.

        Returns
        -------
        item : object
            The item corresponding to the supplied key.
        """
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute "
            f"{key}.")


    @abstractmethod
    def propagate_mono(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2),
        return_wf  : bool = False) -> Array: # pragma: no cover
        """
        Propagates a monochromatic point source through the optics.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """


    def propagate(self        : BaseOptics, 
                  wavelengths : Array,
                  offset      : Array = np.zeros(2),
                  weights     : Array = None) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weights : Array, = None
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.

        Returns
        -------
        psf : Array
            The chromatic point spread function after being propagated
            though the optical layers.
        """
        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths)/len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError("wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively.")
        
        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) \
            else offset
        if offset.shape != (2,):
            raise ValueError("offset must be a 2-element array, got "
                f"shape {offset.shape}.")

        # Caluculate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))
        psfs = propagator(wavelengths, offset)
        if weights is not None:
            psfs *= weights[:, None, None]
        return psfs.sum(0)


    def model(self    : BaseOptics,
              sources : Union[list, Source]) -> Array:
        """
        Models the input sources through the optics. The sources input can be
        a single Source object, or a list of Source objects.

        Parameters
        ----------
        sources : Union[list, Source]
            The sources to model.

        Returns
        -------
        psf : Array
            The sum of the individual sources modelled through the optics.
        """
        if not isinstance(sources, list):
            sources = [sources]
        
        for source in sources:
            if not isinstance(source, Source()):
                raise TypeError("All input sources must be a Source "
                    f"object. Got type: {type(sources)})")
        
        return np.array([source.model(self) for source in sources]).sum(0)


class SimpleOptics(BaseOptics):
    """
    A Simple Optical system that initialises a wavefront based on the wavefront
    diameter and npixels. It adds two attributes, `wf_npixels` and `diameter`,
    as well as the `_construct_wavefront` method that constructs and tilts the
    initial wavefront.

    Attributes
    ----------
    wf_npixels : int
        The nuber of pixels of the initial wavefront to propagte.
    diameter : Array, meters
        The diameter of the initial wavefront to propagte.
    """
    wf_npixels  : int
    diameter    : Array


    def __init__(self       : BaseOptics, 
                 wf_npixels : int,
                 diameter   : float,
                 **kwargs):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, meters
            The diameter of the initial wavefront to propagte.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)

        super().__init__(**kwargs)


    def _construct_wavefront(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2)) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        
        Returns
        -------
        wavefront : Wavefront
            The wavefront object to propagate through the optics.
        """
        wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)
        return wf.tilt(offset)


class NonPropagatorOptics(BaseOptics):
    """
    Implements the basics required for an optical system with a parametric PSF
    output sampling. Adds the `psf_npixels`, `psf_pixel_scale`, and
    `psf_oversample` attributes.

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    psf_oversample : float
        The oversampling factor of the final PSF.
    """
    psf_npixels     : int
    psf_oversample  : float
    psf_pixel_scale : float


    def __init__(self            : BaseOptics,
                 psf_npixels     : int,
                 psf_pixel_scale : float,
                 psf_oversample  : float = 1.,
                 **kwargs):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        psf_oversample : float = 1.
            The oversampling factor of the final PSF.
        """
        self.psf_npixels = int(psf_npixels)
        self.psf_oversample = float(psf_oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)


    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        return dlu.arcsec_to_rad(self.psf_pixel_scale / self.psf_oversample)


class AperturedOptics(BaseOptics):
    """
    Constructs a simple optical system  with an aperture and an optional
    'mask'.
    
    Attributes
    ----------
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    """
    aperture : Union[Array, OpticalLayer()]
    mask     : Union[Array, OpticalLayer()]


    def __init__(
        self     : BaseOptics, 
        aperture : Union[Array, OpticalLayer()],
        mask     : Union[Array, OpticalLayer()] = None,
        **kwargs):
        """
        Parameters
        ----------
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        mask : Union[Array, OpticalLayer], = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
            Default is None.
        """
        if not isinstance(aperture, (Array, OpticalLayer())):
            raise TypeError("aperture must be an Array or "
                f"OpticalLayer, got {type(aperture)}.")
        self.aperture = aperture

        if mask is not None:
            if not isinstance(mask, (Array, OpticalLayer())):
                raise TypeError("mask must be an Array or "
                    f"OpticalLayer, got {type(aperture)}.")
        self.mask = mask

        super().__init__(**kwargs)


    def _apply_aperture(
        self       : BaseOptics, 
        wavelength : float, 
        offset     : Array) -> Wavefront():
        """
        Constructs the wavefront, applies the aperture and mask, and returns
        the wavefront.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians
            The (x, y) offset from the optical axis of the source.
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf *= self.mask
        return wf


######################
### Public Classes ###
######################
class AngularOptics(NonPropagatorOptics, AperturedOptics, SimpleOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    with `psf_pixel_scale` in units of arcseconds.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, meters
        The diameter of the initial wavefront to propagte.
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    psf_oversample : float
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """


    def __init__(self            : AngularOptics, 
                 wf_npixels      : int,
                 diameter        : float,
                 aperture        : Union[Array, OpticalLayer()],
                 psf_npixels     : int,
                 psf_pixel_scale : float,
                 psf_oversample  : float = 1,
                 mask            : Union[Array, OpticalLayer()] = None):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, meters
            The diameter of the initial wavefront to propagte.
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        psf_oversample : float
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        super().__init__(wf_npixels=wf_npixels, diameter=diameter, 
            aperture=aperture, psf_npixels=psf_npixels, mask=mask,
            psf_pixel_scale=psf_pixel_scale, psf_oversample=psf_oversample)


    def propagate_mono(self       : AngularOptics,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcsec_to_rad(pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale_radians)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class CartesianOptics(NonPropagatorOptics, AperturedOptics, SimpleOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    with `psf_pixel_scale` in units of microns.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, meters
        The diameter of the initial wavefront to propagte.
    focal_length : float, meters
        The focal length of the optical system.
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    psf_oversample : float
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """
    focal_length : None


    def __init__(self            : CartesianOptics,
                 wf_npixels      : int,
                 diameter        : float,
                 aperture        : Union[Array, OpticalLayer()],
                 focal_length    : float,
                 psf_npixels     : int,
                 psf_pixel_scale : float,
                 psf_oversample  : int = 1,
                 mask            : Union[Array, OpticalLayer()] = None):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, meters
            The diameter of the initial wavefront to propagte.
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        focal_length : float, meters
            The focal length of the optical system.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, microns
            The pixel scale of the final PSF in units of microns.
        psf_oversample : float
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        self.focal_length = float(focal_length)

        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale, psf_oversample=psf_oversample,
            mask=mask)


    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = 1e-6 * self.psf_pixel_scale / self.psf_oversample
        wf = wf.MFT(self.psf_npixels, pixel_scale, 
            focal_length=self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class FlexibleOptics(AperturedOptics, SimpleOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    using the user-supplied propagator. This allows for propagation of fresnel
    wavefronts.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, meters
        The diameter of the initial wavefront to propagte.
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    propagator : Propagator
        The propagator to use to propagate the wavefront through the optics.
    """
    propagator : None

    def __init__(self : BaseOptics, 
                 wf_npixels : int,
                 diameter : float,
                 aperture : Union[Array, OpticalLayer()],
                 propagator : Propagator(),
                 mask : Union[Array, OpticalLayer()] = None):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, meters
            The diameter of the initial wavefront to propagte.
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        propagator : Propagator
            The propagator to use to propagate the wavefront through the optics.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        if not isinstance(propagator, Propagator()):
            raise TypeError("propagator must be a Propagator object, "
                f"got {type(propagator)}.")
        self.propagator = propagator
        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, mask=mask)


    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        return self.propagator.pixel_scale
    
    def _construct_wavefront(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2)) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        
        Returns
        -------
        wavefront : Wavefront
            The wavefront object to propagate through the optics.
        """
        if isinstance(self.propagator, dLux.propagators.FarFieldFresnel):
            wf = dLux.FresnelWavefront(self.wf_npixels, self.diameter, 
                wavelength)
        else:
            wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)
        return wf.tilt(offset)


    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)
        wf = self.propagator(wf)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class LayeredOptics(SimpleOptics):
    """
    A fully flexible optical system that allows for the arbitrary chaining of
    dLux OpticalLayers.

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagte.
    diameter : Array
        The diameter of the wavefront to model through the system in meters.
    layers : OrderedDict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input wavefront through an optical system.
    """
    layers : OrderedDict


    def __init__(self       : Optics, 
                 wf_npixels : int, 
                 diameter   : float, 
                 layers     : list) -> Optics:
        """
        Constructor for the Optics class.

        Parameters
        ----------
        wf_npixels : int
            The number of pixels to use when propagating the wavefront through
            the optical system.
        diameter : float
            The diameter of the wavefront to model through the system in meters.
        layers : list
            A list of dLux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entried can either be dLux OtpicalLayers, or tuples of the
            form (OpticalLayer, key), with the key being used as the dictionary
            key for the layer.
        """
        super().__init__(wf_npixels=wf_npixels, diameter=diameter)
        self.layers = dlu.list_to_dictionary(layers, True, OpticalLayer())


    def __getattr__(self : Optics, key : str) -> object:
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
        super().__getattr__(key)


    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        # Note: This is a bit inefficient, but should work
        for layer in self.layers.values():
            if isinstance(layer, Propagator()):
                propagator = layer
        return propagator.pixel_scale


    def propagate_mono(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2),
        return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wavefront = self._construct_wavefront(wavelength, offset)
        for layer in list(self.layers.values()):
            wavefront *= layer
        
        if return_wf:
            return wavefront
        return wavefront.psf