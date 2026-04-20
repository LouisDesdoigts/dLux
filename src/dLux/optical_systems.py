"""Optical-system abstractions and concrete optical-system compositions."""

from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod
import jax.numpy as np
from jax import Array
import equinox as eqx
import zodiax as zdx
from typing import Any
import dLux.utils as dlu

__all__ = [
    "BaseOpticalSystem",
    "ParametricOpticalSystem",
    "AngularOpticalSystem",
    "CartesianOpticalSystem",
    "LayeredOpticalSystem",
]

from .layers.optical_layers import OpticalLayer
from .wavefronts import Wavefront
from .sources import BaseSource as Source
from .psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(zdx.Base):
    """
    Abstract base class for optical-system models.

    Defines the required propagation and source-modelling interfaces used by
    concrete optical-system implementations.

    ??? abstract "UML"
        ![UML](../../assets/uml/BaseOpticalSystem.png)
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically inherit method docstrings from parent class.
        """
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["propagate_mono", "propagate", "model"])

    @abstractmethod
    def propagate_mono(
        self: BaseOpticalSystem,
        wavelength: float,
        offset: Array | None = None,
        return_wf: bool = False,
    ) -> Array | Wavefront:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array | None, radians = None
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the PSF array?

        Returns
        -------
        result : Array | Wavefront
            If `return_wf` is False, returns the PSF array.
            If `return_wf` is True, returns the Wavefront object.
        """

    @abstractmethod
    def propagate(
        self: BaseOpticalSystem,
        wavelengths: Array,
        offset: Array | None = None,
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:  # pragma: no cover
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array | None, radians = None
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the PSF array?
        return_psf : bool = False
            Should the PSF object be returned instead of the PSF array?

        Returns
        -------
        result : Array | Wavefront | PSF
            If `return_wf` is False and `return_psf` is False, returns the PSF array.
            If `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            If `return_wf` is False and `return_psf` is True, returns the PSF object.

        """

    @abstractmethod
    def model(
        self: BaseOpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:  # pragma: no cover
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the PSF array?
        return_psf : bool = False
            Should the PSF object be returned instead of the PSF array?

        Returns
        -------
        result : Array | Wavefront | PSF
            If `return_wf` is False and `return_psf` is False, returns the PSF array.
            If `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            If `return_wf` is False and `return_psf` is True, returns the PSF object.
        """


class OpticalSystem(BaseOpticalSystem):
    """
    Base optics class implementing both the `propagate` and `model` methods that are
    universal to all optics classes.

    ??? abstract "UML"
        ![UML](../../assets/uml/OpticalSystem.png)
    """

    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array | None = None,
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array | None, radians = None
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the PSF array?
        return_psf : bool = False
            Should the PSF object be returned instead of the PSF array?

        Returns
        -------
        result : Array | Wavefront | PSF
            If `return_wf` is False and `return_psf` is False, returns the PSF array.
            If `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            If `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "Cannot return both Wavefront and PSF objects. Choose one: "
                "return_wf=True for Wavefront, or return_psf=True for PSF."
            )

        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError(
                f"Wavelength and weight shape mismatch: "
                f"wavelengths {wavelengths.shape} vs weights {weights.shape}. "
                f"Must have same length and dimensions."
            )

        # Check offset
        if offset is None:
            offset = np.zeros(2)
        else:
            offset = np.asarray(offset)
        if offset.shape != (2,):
            raise ValueError(
                f"offset must be [x, y] array of shape (2,), "
                f"got shape {offset.shape}. "
                "Pass offset as [on_axis_x, on_axis_y] angles in radians."
            )

        # Calculate - note we multiply by sqrt(weight) to account for the
        # fact that the PSF is the square of the amplitude
        prop_fn = lambda wavelength, weight: self.propagate_mono(
            wavelength, offset, return_wf=True
        ).multiply("phasor", weight**0.5)
        wf = eqx.filter_vmap(prop_fn)(wavelengths, weights)

        # Return PSF, Wavefront, or PSF array
        if return_wf:
            return wf
        if return_psf:
            return PSF(wf.psf.sum(0), wf.pixel_scale.mean())
        return wf.psf.sum(0)

    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        return source.model(self, return_wf, return_psf)


class ParametricOpticalSystem(OpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

    ??? abstract "UML"
        ![UML](../../assets/uml/ParametricOpticalSystem.png)

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    """

    psf_npixels: int
    oversample: int
    psf_pixel_scale: float

    def __init__(
        self: ParametricOpticalSystem,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        oversample : int = 1.
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.psf_npixels = int(psf_npixels)
        self.oversample = int(oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)

    @property
    def fov(self: ParametricOpticalSystem) -> float:
        """
        Returns the field of view of the optical system in the units of the pixel scale.

        Returns
        -------
        fov : float
            The field of view of the optical system in the units of the pixel scale.

        """
        return self.psf_npixels * self.psf_pixel_scale


##################
# Public Classes #
##################
class LayeredOpticalSystem(OpticalSystem):
    """
    A flexible optical system that allows for the arbitrary chaining of OpticalLayers.

    ??? abstract "UML"
        ![UML](../../assets/uml/LayeredOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagate.
    diameter : float, metres
        The diameter of the wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    """

    wf_npixels: int
    diameter: float
    layers: OrderedDict

    def __init__(
        self: LayeredOpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer | tuple[str, OpticalLayer]],
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The size of the initial wavefront to propagate.
        diameter : float
            The diameter of the wavefront to propagate.
        layers : list[OpticalLayer | tuple[str, OpticalLayer]]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)
        self.layers = dlu.list2dictionary(layers, True, OpticalLayer)

    def __getattr__(self: LayeredOpticalSystem, key: str) -> Any:
        """
        Raises both the individual layers and the attributes of the layers via
        their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : Any
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise dlu.missing_attribute_error(self, key, list(self.layers.keys()))

    def initialise_wavefront(
        self: LayeredOpticalSystem, wavelength: Array, offset: Array = None
    ) -> Wavefront:
        """
        Initialises the wavefront for the propagate_mono method. and applies the offset
        as a tilt to the wavefront.

        Parameters
        ----------
        wavelength : Array
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = None
            The (x, y) offset from the optical axis of the source. Passed as angles in
            radians.

        Returns
        -------
        wavefront : Wavefront
            The initialised wavefront with the offset applied as a tilt.
        """
        if offset is None:
            offset = np.zeros(2)

        # Initialise wavefront
        wavefront = Wavefront(wavelength, self.wf_npixels, self.diameter)
        wavefront = wavefront.tilt(offset)
        return wavefront

    def propagate_mono(
        self: LayeredOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
        return_wf: bool = False,
    ) -> Array | Wavefront:
        # Initialise wavefront
        wavefront = self.initialise_wavefront(wavelength, offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wavefront = layer(wavefront)

        # Return PSF or Wavefront
        if return_wf:
            return wavefront
        return wavefront.psf

    def debug_propagate_mono(
        self: LayeredOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
    ) -> Array | Wavefront:
        # Outputs dictionary
        outputs = {}

        # Initialise wavefront
        wavefront = self.initialise_wavefront(wavelength, offset)
        outputs["initial_wavefront"] = wavefront

        # Apply layers, storing the outputs
        for name, layer in self.layers.items():
            wavefront = layer(wavefront)
            outputs[name] = wavefront

        # Return the wavefront and the outputs dictionary for debugging
        return wavefront, outputs

    def insert_layer(
        self: LayeredOpticalSystem,
        layer: OpticalLayer | tuple[str, OpticalLayer],
        index: int,
    ) -> OpticalSystem:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : OpticalLayer | tuple[str, OpticalLayer]
            The layer to be inserted.
        index : int
            The index at which to insert the layer.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set(
            "layers", dlu.insert_layer(self.layers, layer, index, OpticalLayer)
        )

    def remove_layer(self: LayeredOpticalSystem, key: str) -> OpticalSystem:
        """
        Removes a layer from the layers dictionary, specified by its key.

        Parameters
        ----------
        key : str
            The key of the layer to be removed.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set("layers", dlu.remove_layer(self.layers, key))


class ParametricLayeredOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that also includes the attributes of
    the ParametricOpticalSystem class. Mainly used to enable the debug_propagate_mono
    method to be be common modelled across both the AngularOpticalSystem and
    CartesianOpticalSystem classes.

    ??? abstract "UML"
        ![UML](../../assets/uml/ParametricLayeredOpticalSystem.png)
    """

    @abstractmethod
    def to_focus(
        self: AngularOpticalSystem,
        wavefront: Wavefront,
    ) -> Array | Wavefront:
        """
        Propagate the wavefront to the focal plane using an FFT-based propagator with
        the specified pixel scale and number of pixels.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate to the focal plane.
        Returns
        -------
        result : Array | Wavefront
            The propagated wavefront at the focal plane
        """

    def debug_propagate_mono(
        self: AngularOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
    ) -> Array | Wavefront:
        # Propagate the upstream layers and store the outputs
        wf, outputs = super().debug_propagate_mono(wavelength, offset)

        # Propagate to the focal plane and store the output
        wf = self.to_focus(wf)
        outputs["final_wavefront"] = wf

        # Return the wavefront and the outputs dictionary for debugging
        return wf, outputs

    def propagate_mono(
        self: AngularOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
        return_wf: bool = False,
    ) -> Array | Wavefront:
        # Upstream layers propagation
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        wf = self.to_focus(wf)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class AngularOpticalSystem(ParametricLayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of arcseconds.

    ??? abstract "UML"
        ![UML](../../assets/uml/AngularOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, arcseconds
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    def __init__(
        self: AngularOpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer | tuple[str, OpticalLayer]],
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer | tuple[str, OpticalLayer]]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def to_focus(
        self: AngularOpticalSystem,
        wavefront: Wavefront,
    ) -> Array | Wavefront:
        """
        Propagate the wavefront to the focal plane using an FFT-based propagator with
        the specified pixel scale and number of pixels.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate to the focal plane.
        Returns
        -------
        result : Array | Wavefront
            The propagated wavefront at the focal plane
        """
        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        return wavefront.propagate(psf_npixels, pixel_scale)


class CartesianOpticalSystem(ParametricLayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of microns.

    ??? abstract "UML"
        ![UML](../../assets/uml/CartesianOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    focal_length : float, metres
        The focal length of the system.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, microns
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    focal_length: float

    def __init__(
        self: CartesianOpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer | tuple[str, OpticalLayer]],
        focal_length: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer | tuple[str, OpticalLayer]]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        focal_length : float, metres
            The focal length of the system.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, microns
            The pixel scale of the final PSF in units of microns.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.focal_length = float(focal_length)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def to_focus(
        self: AngularOpticalSystem,
        wavefront: Wavefront,
    ) -> Array | Wavefront:
        """
        Propagate the wavefront to the focal plane using an FFT-based propagator with
        the specified pixel scale and number of pixels.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate to the focal plane.
        Returns
        -------
        result : Array | Wavefront
            The propagated wavefront at the focal plane
        """
        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample
        return wavefront.propagate(psf_npixels, pixel_scale)
