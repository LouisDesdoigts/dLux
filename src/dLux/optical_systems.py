from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from zodiax import filter_vmap, Base
from typing import Union, Any
import dLux.utils as dlu


__all__ = [
    "BaseOpticalSystem",
    "AngularOpticalSystem",
    "CartesianOpticalSystem",
    "LayeredOpticalSystem",
    "ConvergingBeamOpticalSystem",
]

from .layers.optical_layers import OpticalLayer
from .wavefronts import Wavefront
from .sources import BaseSource as Source
from .psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(Base):
    @abstractmethod
    def propagate_mono(
        self: BaseOpticalSystem,
        wavelength: float,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """

    @abstractmethod
    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.

        """

    @abstractmethod
    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """


class OpticalSystem(BaseOpticalSystem):
    """
    Base optics class implementing both the `propagate` and `model` methods that are
    universal to all optics classes.
    """

    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. "
                "Please choose one."
            )

        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError(
                "wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively."
            )

        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) else offset
        if offset.shape != (2,):
            raise ValueError(
                "offset must be a 2-element array, got "
                f"shape {offset.shape}."
            )

        # Calculate - note we multiply by sqrt(weight) to account for the
        # fact that the PSF is the square of the amplitude
        prop_fn = lambda wavelength, weight: self.propagate_mono(
            wavelength, offset, return_wf=True
        ).multiply("amplitude", weight**0.5)
        wf = filter_vmap(prop_fn)(wavelengths, weights)

        # Return PSF, Wavefront, or array psf
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
    ) -> Array:
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        return source.model(self, return_wf, return_psf)


class ParametricOpticalSystem(OpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

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
        self: OpticalSystem,
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
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The size of the initial wavefront to propagate.
        diameter : float
            The diameter of the wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)
        self.layers = dlu.list2dictionary(layers, True, OpticalLayer)

    def __getattr__(self: OpticalSystem, key: str) -> Any:
        """
        Raises both the individual layers and the attributes of the layers via
        their keys.

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
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Initialise wavefront
        wavefront = Wavefront(self.wf_npixels, self.diameter, wavelength)
        wavefront = wavefront.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wavefront *= layer

        # Return PSF or Wavefront
        if return_wf:
            return wavefront
        return wavefront.psf

    def insert_layer(
        self: OpticalSystem, layer: Union[OpticalLayer, tuple], index: int
    ) -> OpticalSystem:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : Any
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

    def remove_layer(self: OpticalLayer, key: str) -> OpticalSystem:
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


class AngularOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
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
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
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
        layers : list[OpticalLayer, tuple]
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

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class CartesianOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
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

    focal_length: None

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
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
        layers : list[OpticalLayer, tuple]
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

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale, self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class ConvergingBeamOpticalSystem(
    ParametricOpticalSystem, LayeredOpticalSystem
):
    """
    An extension to the `LayeredOpticalSystem` class that propagates a wavefront
    through a converging beam to an intermediate pupil plane before propagating
    to the image plane with `psf_pixel_scale` in units of arcseconds.

    The primary (plane 1) and secondary (plane 2) pupils are related by a
    geometric pupil magnification factor, `magnification`, which describes how
    much the converging beam shrinks between the primary mirror and the
    secondary plane. For a collimated input beam and a primary mirror of focal
    length `f1`, with the secondary plane located a distance `d` downstream
    along the converging beam (`0 < d < f1`), the pupil magnification is

        M = 1 / (1 - d / f1),

    so that the beam footprint at the secondary has diameter

        D_secondary = D_primary / M.

    This magnification is used to map diameters, pixel scales and padding
    between the primary and secondary pupil planes.

    Note
    ----
    The `pad_factor` attribute is derived from `magnification` and the primary /
    secondary diameters at construction time. Because array sizes must remain
    static for JAX, changing `magnification` or the plane diameters after
    construction may invalidate the padding assumptions and lead to aliasing or
    wrap-around. In this implementation `magnification` is intended to be
    treated as fixed for a given system instance.

    ??? abstract "UML"
        ![UML](../../assets/uml/ConvergingBeamOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    plane_names : tuple[str, ...]
        Ordered names of the pupil planes in the system. For this class,
        the planes are ("primary", "secondary"), corresponding to the
        primary and secondary pupils.
    diameter : dict[str, float]
        Mapping from plane name to pupil diameter in metres. For this class,
        the keys are "primary" and "secondary".
    layers : dict[str, OrderedDict[str, OpticalLayer]]
        Mapping from plane name to an ordered dictionary of `OpticalLayer`
        objects applied at that plane. For example, `layers["primary"]`
        contains the stack at the primary pupil.
    plane_separation : float, metres
        The physical distance between plane 1 and plane 2 along the
        converging beam.
    magnification : float
        Geometric pupil magnification between the primary pupil and the
        secondary plane, defined as `M = D_primary / D_secondary`. For a
        primary of focal length `f1` and a secondary plane at distance
        `d` along the converging beam (`0 < d < f1`), this can be
        computed as `M = 1 / (1 - d / f1)`. This value is treated as fixed
        for a given system instance.
    pad_factor : int
        Integer padding factor used during Fresnel propagation to ensure the
        magnified secondary pupil footprint fits within the padded grid. It is
        computed from the primary/secondary diameters and `magnification` when
        the system is constructed and is assumed constant thereafter.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, arcseconds
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the `psf_pixel_scale`
        parameter while increasing `psf_npixels`.

    """

    plane_names: tuple[str, ...]
    plane_separation: float
    magnification: float
    pad_factor: int

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        p1_diameter: float,
        p2_diameter: float,
        p1_layers: list[OpticalLayer, tuple],
        p2_layers: list[OpticalLayer, tuple],
        plane_separation: float,
        magnification: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        p1_diameter : Array, metres
            The diameter of the first plane.
        p2_diameter : Array, metres
            The diameter of the second plane.
        p1_layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply at the pupil. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        p2_layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply at plane 2. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        plane_separation : float, metres
            The physical distance between plane 1 and plane 2.
        magnification : float
            Geometric pupil magnification between the primary pupil and the
            secondary plane, defined as `M = D_primary / D_secondary`. For a
            primary of focal length `f1` and a secondary plane at distance `d`
            along the converging beam (`0 < d < f1`), this can be computed as
            `M = 1 / (1 - d / f1)`. This value is treated as fixed for a given
            system instance.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """

        # Instantiate the parent class with empty layers
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=p1_diameter,
            layers=[],  # we take over layer handling below
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

        # Define the ordered plane names (primary, secondary)
        self.plane_names = ("primary", "secondary")

        # Reuse the diameter attribute to store both plane diameters by key
        self.diameter = {
            "primary": float(p1_diameter),
            "secondary": float(p2_diameter),
        }

        # Reuse layers and map to multiple planes
        self.layers = {
            "primary": dlu.list2dictionary(p1_layers, True, OpticalLayer),
            "secondary": dlu.list2dictionary(p2_layers, True, OpticalLayer),
        }

        # Converging-beam specific parameters
        self.plane_separation = float(plane_separation)
        self.magnification = float(magnification)

        # Compute pad_factor for propagation to P2
        self.pad_factor = int(
            np.ceil((p2_diameter * magnification) / p1_diameter)
        )

    def __getattr__(self, key: str):
        """
        Search for layers and layer attributes across all planes.

        This extends the LayeredOpticalSystem behaviour to a multi-plane
        system by iterating over the per-plane layer dictionaries stored
        in `self.layers`.
        """
        # Iterate over each plane's layers
        for plane_layers in self.layers.values():
            # Direct layer lookup by key
            if key in plane_layers:
                return plane_layers[key]

            # Attribute lifted from any layer
            for layer in plane_layers.values():
                if hasattr(layer, key):
                    return getattr(layer, key)

        raise AttributeError(
            f"{self.__class__.__name__} has no attribute '{key}'."
        )

    def insert_layer(
        self: OpticalSystem,
        layer: Union[OpticalLayer, tuple],
        index: int,
        plane_index: int,
    ) -> OpticalSystem:
        """
        Inserts a layer into the specified plane's layers at a given index.

        Parameters
        ----------
        layer : OpticalLayer or tuple
            The layer to insert. Can be a tuple (key, layer) to specify a key.
        index : int
            The index at which to insert the layer.
        plane_index : int
            The index of the plane where the layer should be inserted.
            For this class, 0 corresponds to the primary plane and
            1 corresponds to the secondary plane.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index < 0 or plane_index >= len(self.plane_names):
            raise ValueError("Invalid plane_index. Must be 0 or 1.")

        plane_name = self.plane_names[plane_index]

        # Insert into the appropriate per-plane layer dictionary
        updated_plane_layers = dlu.insert_layer(
            self.layers[plane_name], layer, index, OpticalLayer
        )

        # Create a new multi-plane layers mapping with this plane updated
        new_layers = {**self.layers, plane_name: updated_plane_layers}

        # Use zodiax-style .set to return an updated system
        return self.set("layers", new_layers)

    def remove_layer(
        self: OpticalSystem, key: str, plane_index: int
    ) -> OpticalSystem:
        """
        Removes a layer from the specified plane's layers.

        Parameters
        ----------
        key : str
            The key of the layer to remove.
        plane_index : int
            The plane where the layer should be removed. Must be 0 or 1.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index < 0 or plane_index >= len(self.plane_names):
            raise ValueError("Invalid plane_index. Must be 0 or 1.")

        plane_name = self.plane_names[plane_index]

        # Remove the layer from the appropriate per-plane dictionary
        updated_plane_layers = dlu.remove_layer(self.layers[plane_name], key)

        # Create a new multi-plane layers mapping with this plane updated
        new_layers = {**self.layers, plane_name: updated_plane_layers}

        return self.set("layers", new_layers)

    def _propagate_mono_to_secondary_core(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array,
    ) -> tuple[Wavefront, float, float, float]:
        """
        Internal helper: propagate a monochromatic wavefront from the primary
        pupil to the secondary pupil, applying all primary and secondary
        layers, and return the Wavefront at the secondary plane on the padded
        grid.

        Returns
        -------
        wf_p2 : Wavefront
            Wavefront after propagation to and through the secondary plane,
            still on the padded grid.
        p1_diameter : float
            Diameter of the primary pupil in metres.
        p2_diameter : float
            Diameter of the secondary pupil in metres.
        ps_in : float
            Effective pixel scale (metres per pixel) at the secondary pupil
            plane before rescaling to the canonical pupil grid.
        """
        primary, secondary = self.plane_names
        p1_diameter = self.diameter[primary]
        p2_diameter = self.diameter[secondary]
        p1_layers = self.layers[primary]
        p2_layers = self.layers[secondary]

        # === Initialize primary wavefront ===
        wf = Wavefront(self.wf_npixels, p1_diameter, wavelength)
        wf = wf.tilt(offset)

        # === Apply layers at primary plane ===
        for layer in list(p1_layers.values()):
            wf *= layer

        # === Propagate to secondary using Fresnel_AS ===
        prop_dist = self.plane_separation * self.magnification
        wf = wf.propagate_fresnel_AS(prop_dist, pad=self.pad_factor)
        # wf now has pad * wf_npixels

        # === Apply layers at secondary plane ===
        ps_in = p2_diameter * self.magnification / self.wf_npixels
        ps_out = p1_diameter / self.wf_npixels
        npix_out = self.pad_factor * self.wf_npixels

        for layer in list(p2_layers.values()):
            # scaled layer on padded grid
            scaled_layer = dlu.scale_layer(layer, ps_in, ps_out, npix_out)
            wf *= scaled_layer

        return wf, p1_diameter, p2_diameter, ps_in

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Custom propagation using hybrid Fresnel/MFT logic:
        - Apply primary plane layers.
        - Fresnel propagate to the secondary plane.
        - Dynamically apply scaled aperture at the secondary plane.
        - Back-propagate to an image of the primary pupil.
        - Final MFT propagation to the focal plane.
        """
        wf, p1_diameter, _, _ = self._propagate_mono_to_secondary_core(
            wavelength, offset
        )

        # === Back-propagate to primary pupil plane ===
        prop_dist = self.plane_separation * self.magnification
        wf = wf.propagate_fresnel_AS(-prop_dist, pad=1)
        # wf still has pad * wf_npixels

        # === Resize to original pupil size ===
        wf = wf.resize(self.wf_npixels)  # Crops to original wf_npixels

        # === Final MFT propagation to PSF ===
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale_rad = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale_rad)

        if return_wf:
            return wf
        return wf.psf

    def prop_mono_to_p2(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the first optical plane and
        stops at the second optical plane.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the PSF Array at Plane 2.
            if `return_wf` is True, returns the Wavefront object at Plane 2.
        """
        (
            wf,
            p1_diameter,
            p2_diameter,
            ps_in,
        ) = self._propagate_mono_to_secondary_core(wavelength, offset)

        # === Scale to the canonical secondary-pupil grid ===
        wf = wf.scale_to(self.wf_npixels, ps_in)
        wf = wf.set("pixel_scale", np.asarray(p2_diameter / self.wf_npixels))

        if return_wf:
            return wf
        return wf.psf
