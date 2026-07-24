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
from .coordinates import CoordSpec
from .layers.optical_layers import BaseOpticalLayer

__all__ = [
    "BaseOpticalSystem",
    "LayeredOpticalSystem",
]

from .wavefronts import Wavefront
from .sources import BaseSource as Source
from .psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(zdx.Base):
    """Base class for optical-system models."""

    def __init_subclass__(cls, **kwargs):
        """Inherit the optical-system interface documentation."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(
            cls,
            ["__call__", "propagate_mono", "propagate", "model"],
        )

    @abstractmethod
    def __call__(self, wavefront: Wavefront) -> Wavefront:  # pragma: no cover
        """Apply the optical system to a wavefront."""

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Backwards-compatible alias for calling the optical system."""
        return self(wavefront)

    @abstractmethod
    def propagate_mono(
        self,
        wavelength: float,
        offset: Array | None = None,
        return_wf: bool = False,
        stokes: Array | None = None,
    ) -> Any:  # pragma: no cover
        """Propagate a monochromatic point source."""

    @abstractmethod
    def propagate(
        self,
        wavelengths: Array,
        offset: Array | None = None,
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
        stokes: Array | None = None,
    ) -> Any:  # pragma: no cover
        """Propagate a polychromatic point source."""

    @abstractmethod
    def model(
        self,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Any:  # pragma: no cover
        """Model a source through the optical system."""


class OpticalSystem(BaseOpticalSystem):
    """
    Base optics class implementing both the `propagate` and `model` methods that are
    universal to all optics classes.

    ??? abstract "UML"
        ![UML](../assets/uml/OpticalSystem.png)
    """

    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array | None = None,
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
        stokes: Array | None = None,
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
        stokes : Array | None = None
            The input Stokes vector [I, Q, U, V] of the source. If provided, the
            wavefront is initialised as a `PolarisedWavefront` carrying these Stokes
            parameters. If None, defaults to an unpolarised [1, 0, 0, 0] state when a
            polarising layer is present.

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
            wavelength, offset, return_wf=True, stokes=stokes
        ).multiply("phasor", weight**0.5)
        wf = eqx.filter_vmap(prop_fn)(wavelengths, weights)

        # Return the Wavefront
        if return_wf:
            return wf

        # Return psf object or array
        psf = wf.psf_from_stokes(stokes).sum(0)
        if return_psf:
            return PSF(psf, wf.spec)
        return psf

    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        return source.model(self, return_wf, return_psf)


class LayeredOpticalSystem(OpticalSystem):
    """
    A flexible optical system that allows for the arbitrary chaining of OpticalLayers.

    ??? abstract "UML"
        ![UML](../assets/uml/LayeredOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagate.
    diameter : float, metres
        The diameter of the wavefront to propagate.
    layers : OrderedDict
        A series of `BaseOpticalLayer` transformations to apply to wavefronts.
    """

    wf_npixels: int
    diameter: float
    layers: OrderedDict

    def __init__(
        self: LayeredOpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[BaseOpticalLayer | tuple[str, BaseOpticalLayer]],
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The size of the initial wavefront to propagate.
        diameter : float
            The diameter of the wavefront to propagate.
        layers : list[BaseOpticalLayer | tuple[str, BaseOpticalLayer]]
            A list of `BaseOpticalLayer` transformations to apply to wavefronts. The
            list entries can be either `BaseOpticalLayer` objects or tuples of
            specify a key for the layer in the layers dictionary.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = np.asarray(diameter, float)
        self.layers = dlu.list2dictionary(layers, True, BaseOpticalLayer)

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
        self: LayeredOpticalSystem,
        wavelength: Array,
        offset: Array = None,
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
        spec = CoordSpec(
            n=(self.wf_npixels, self.wf_npixels),
            d=self.diameter / self.wf_npixels,
            c=0.0,
            unit="m",
        )
        wavefront = Wavefront(wavelength, spec)
        return wavefront.tilt(offset)

    def _apply_mono(self: LayeredOpticalSystem, wavefront: Wavefront) -> Wavefront:
        """
        Applies all system layers to a monochromatic wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The monochromatic wavefront to propagate through the layers.

        Returns
        -------
        wavefront : Wavefront
            The wavefront after applying every layer in the system.
        """
        # Apply the layers sequentially
        for layer in self.layers.values():
            wavefront = layer(wavefront)
        return wavefront

    def __call__(self: LayeredOpticalSystem, wavefront: Wavefront) -> Wavefront:
        if not isinstance(wavefront, Wavefront):
            raise TypeError(
                f"wavefront must be a Wavefront instance, got "
                f"{type(wavefront).__name__}."
            )

        # Monochromatic wavefronts can just be applied directly
        if not wavefront.is_chromatic:
            return self._apply_mono(wavefront)

        # Chromatic wavefronts are vectorised over their wavelength dimensions so that
        # each system layer receives a monochromatic wavefront.
        apply_fn = eqx.filter_vmap(self._apply_mono, in_axes=(wavefront._mapped_axis,))
        return apply_fn(wavefront)

    def propagate_mono(
        self: LayeredOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
        return_wf: bool = False,
        stokes: Array | None = None,
    ) -> Array | Wavefront:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array | None, radians = None
            The (x, y) offset from the optical axis of the source.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the PSF array?
        stokes : Array | None = None
            The input Stokes vector [I, Q, U, V] of the source. If provided, the
            wavefront is initialised as a `PolarisedWavefront` carrying these Stokes
            parameters.

        Returns
        -------
        result : Array | Wavefront
            If `return_wf` is False, returns the PSF array.
            If `return_wf` is True, returns the Wavefront object.
        """
        # Initialise wavefront
        wavefront = self.initialise_wavefront(wavelength, offset)

        # Apply the complete system
        wavefront = self(wavefront)

        # Return PSF or Wavefront
        if return_wf:
            return wavefront
        return wavefront.psf_from_stokes(stokes)

    def debug_propagate_mono(
        self: LayeredOpticalSystem,
        wavelength: Array,
        offset: Array | None = None,
    ) -> Array | Wavefront:
        """
        Propagates a monochromatic wavefront through the layers, returning
        intermediate wavefront states for debugging.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate.
        offset : Array | None, radians = None
            The (x, y) offset from the optical axis of the source.

        Returns
        -------
        wavefront : Wavefront
            The final propagated wavefront.
        outputs : dict
            Dictionary mapping layer names to their output wavefronts.
        """
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
        layer: BaseOpticalLayer | tuple[str, BaseOpticalLayer],
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
        layer : BaseOpticalLayer | tuple[str, BaseOpticalLayer]
            The layer to be inserted.
        index : int
            The index at which to insert the layer.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set(
            "layers", dlu.insert_layer(self.layers, layer, index, BaseOpticalLayer)
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
