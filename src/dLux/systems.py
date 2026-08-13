"""Ordered layer systems for optical and detector modelling."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax.numpy as np

import dLux.utils as dlu

from .base import Base
from .grids import GridSpec
from .layers.detector import BaseDetectorLayer
from .layers.optical import BaseLayer, BaseOpticalLayer
from .fields import Intensity, Wavefront
from .sources import Spectrum

__all__ = ["LayeredSystem", "OpticalSystem", "DetectorSystem"]


class LayeredSystem(Base):
    """Base contract for an ordered immutable collection of compatible layers.

    Layer names provide raised parameter paths for inspection and optimisation. A
    system applies layers in insertion order and can return the final object or
    labelled intermediate states for debugging and analysis.
    """

    layers: OrderedDict

    def __init__(self, layers, layer_type=BaseLayer):
        """Initialise an ordered collection of compatible layers.

        Parameters
        ----------
        layers : mapping or sequence
            Named mapping, ``(name, layer)`` sequence, or layer sequence. Unnamed
            layers receive generated names while insertion order is preserved.
        layer_type : type
            Required base type for every layer.
        """
        self.layers = dlu.list2dictionary(layers, True, layer_type)

    def __getattr__(self, key: str) -> Any:
        """Resolve attributes from named or contained layers."""
        return dlu.resolve_attr(self, key, self.layers)

    def apply(self, target):
        """Apply every layer to a target in insertion order.

        Parameters
        ----------
        target : Base
            Object accepted by every layer in the system. Each layer receives the
            output of the preceding layer.

        Returns
        -------
        target : Base
            Final transformed object. The input object is not mutated.
        """
        for layer in self.layers.values():
            target = layer(target)
        return target

    def __call__(self, target):
        """Call :meth:`apply` using concise system syntax."""
        return self.apply(target)

    def debug(self, target):
        """Apply every layer and retain the intermediate states.

        Parameters
        ----------
        target : Base
            Object accepted by every layer in the system.

        Returns
        -------
        output : Base
            Final transformed object.
        states : dict[str, Base]
            Ordered mapping containing the original target under ``"input"`` and
            the output of each layer under its layer name. Values may have different
            concrete types when the system contains type-changing layers.
        """
        outputs = {"input": target}
        for name, layer in self.layers.items():
            target = layer(target)
            outputs[name] = target
        return target, outputs

    def insert_layer(self, layer, index: int, layer_type=BaseLayer):
        """Return a copy with a layer inserted at the requested index.

        Parameters
        ----------
        layer : BaseLayer or tuple[str, BaseLayer]
            Layer, optionally paired with its explicit name.
        index : int
            Insertion position in the ordered system.
        layer_type : type
            Required layer base class.

        Returns
        -------
        system : LayeredSystem
            Updated immutable system with the new layer inserted and all other layer
            order preserved.
        """
        layers = dlu.insert_layer(self.layers, layer, index, layer_type)
        return self.set("layers", layers)

    def remove_layer(self, key: str):
        """Return a copy with the named layer removed.

        Parameters
        ----------
        key : str
            Name of the layer to remove.

        Returns
        -------
        system : LayeredSystem
            Updated immutable system with the remaining layer order preserved.
        """
        return self.set("layers", dlu.remove_layer(self.layers, key))


class OpticalSystem(LayeredSystem, BaseOpticalLayer):
    """Model an optical train from a physical input coordinate specification.

    The input grid may use any supported length unit. Its coordinates are converted
    to canonical SI values when fields are evaluated; angular input grids are not
    accepted.

    Examples
    --------
    Build a pupil-to-focal-plane model and use its three propagation interfaces:

    ```python
    import jax.numpy as np

    import dLux as dl


    # Make the grids
    pupil_grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    focal_grid = dl.GridSpec(n=64, d=25, unit="mas")

    # Build the optical system
    optics = dl.OpticalSystem(
        layers=[
            ("pupil", dl.SimpleCircular(diameter=1.0)(pupil_grid)),
            ("focus", dl.Fraunhofer(focal_grid)),
        ],
        grid=pupil_grid,
    )

    # Model a point-spread function
    wavelengths = np.linspace(1.0e-6, 1.2e-6, 10)
    psf = optics.propagate(wavelengths)  # Returns an intensity array

    # Model a source
    source = dl.Source(wavelengths, weights=dl.Blackbody(10_000))
    intensity = optics.model(source)  # Returns an Intensity object

    # Propagate a wavefront
    wavefront = dl.Wavefront(wavelengths, pupil_grid)
    wavefront = optics.apply(wavefront)  # Returns a Wavefront object
    ```
    """

    layers: OrderedDict
    grid: GridSpec

    def __init__(self, layers, grid: GridSpec):
        """Initialise an optical system on its input pupil grid.

        Parameters
        ----------
        layers : mapping or sequence
            Ordered optical layers, optionally supplied as ``(name, layer)`` pairs.
        grid : GridSpec
            Two-dimensional input sampling with defined ``n`` and ``d`` in a
            supported physical length unit.
        """
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        grid = grid.broadcast(2)
        if grid.n is None or grid.d is None:
            raise ValueError("grid must define n and d.")
        dlu.canonical_unit(grid.unit, dimension="length", name="input grid unit")
        self.grid = grid
        super().__init__(layers, BaseOpticalLayer)

    @staticmethod
    def _to_intensity(wavefront: Wavefront, stokes=None) -> Intensity:
        """Convert a propagated wavefront into sampled intensity."""
        intensity = wavefront.to_intensity(stokes)
        mapped_sampling = wavefront.d.ndim > 1 or (
            wavefront.c is not None and wavefront.c.ndim > 1
        )
        if wavefront.is_chromatic and not mapped_sampling:
            intensity = intensity.set(data=intensity.data.sum(0))
        return intensity

    def apply_mono(self, wavefront: Wavefront):
        """Propagate one monochromatic wavefront through every optical layer.

        This method satisfies the `BaseOpticalLayer` primitive contract and delegates
        to `apply`. It returns the final immutable wavefront with layer-defined grid
        changes retained.
        """
        return self.apply(wavefront)

    def apply(self, wavefront: Wavefront):
        """Propagate a complete wavefront through the optical train.

        Parameters
        ----------
        wavefront : Wavefront
            Scalar or polarised wavefront. Leading wavelength and batch axes are
            handled by the individual optical layers.

        Returns
        -------
        wavefront : Wavefront
            Final propagated wavefront. The input wavefront is not mutated.
        """
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront instance.")
        return super().apply(wavefront)

    def initialise_wavefront(self, wavelength, offset=None) -> Wavefront:
        """Construct the system input wavefront for a point source.

        Parameters
        ----------
        wavelength : float or Array, metres
            Scalar wavelength or wavelength array. Array dimensions become leading
            wavefront axes.
        offset : Array or None, radians
            Angular ``(x, y)`` source offset with shape ``(2,)``. Defaults to an
            on-axis source.

        Returns
        -------
        wavefront : Wavefront
            Unit-power wavefront on the system input grid with the source tilt
            applied. The offset is not converted from a named unit.
        """
        offset = np.zeros(2) if offset is None else np.asarray(offset)
        if offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")
        return Wavefront(wavelength, self.grid).tilt(offset)

    def propagate_mono(
        self, wavelength, offset=None, return_wf=False, return_all=False, stokes=None
    ):
        """Propagate a monochromatic point source through the system.

        Returns the sampled intensity array by default, or the final wavefront when
        ``return_wf`` is true. ``return_all`` returns all output containers.

        Parameters
        ----------
        wavelength : float, metres
            Monochromatic source wavelength.
        offset : Array or None, radians
            Optional two-component angular source offset.
        return_wf, return_all : bool
            Select the final wavefront or complete result mapping; mutually exclusive.
        stokes : Array or None
            Optional input Stokes vector for intensity evaluation.

        Returns
        -------
        output : Array, Wavefront, or dict
            Intensity data by default, the final wavefront with ``return_wf=True``,
            or a mapping containing both representations and retained PSF aliases
            with ``return_all=True``.
        """
        # Validate the requested output
        if return_wf and return_all:
            raise ValueError("return_wf and return_all are mutually exclusive.")

        # Initialize and propagate the monochromatic wavefront
        wavefront = self(self.initialise_wavefront(wavelength, offset))
        intensity = self._to_intensity(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {
                "Wavefront": wavefront,
                "Intensity": intensity,
                "intensity": intensity.data,
                "PSF": intensity,
                "psf": intensity.data,
            }
        if return_wf:
            return wavefront
        return intensity.data

    def propagate(
        self,
        wavelengths,
        offset=None,
        weights=None,
        return_wf=False,
        return_all=False,
        stokes=None,
    ):
        """Propagate a weighted polychromatic point source through the system.

        Returns the sampled intensity array by default, or the final wavefront when
        ``return_wf`` is true. ``return_all`` returns all output containers.

        Parameters
        ----------
        wavelengths : Array, metres
            One-dimensional wavelength samples.
        offset : Array or None, radians
            Optional two-component angular source offset.
        weights : Array or None
            Spectral weights matching ``wavelengths``; defaults to equal weights.
        return_wf, return_all : bool
            Select the final wavefront or complete result mapping; mutually exclusive.
        stokes : Array or None
            Optional input Stokes vector for intensity evaluation.

        Returns
        -------
        output : Array, Wavefront, or dict
            Spectrally summed intensity data by default, the weighted final wavefront
            with ``return_wf=True``, or a mapping containing both representations
            with ``return_all=True``.
        """
        # Validate the requested output
        if return_wf and return_all:
            raise ValueError("return_wf and return_all are mutually exclusive.")

        # Standardize and validate the spectrum
        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / wavelengths.size
        else:
            weights = np.atleast_1d(weights)

        if weights.shape != wavelengths.shape:
            raise ValueError("wavelengths and weights must have matching shapes.")

        # Initialize and spectrally weight the wavefront
        wavefront = self(self.initialise_wavefront(wavelengths, offset))
        ndim = wavefront.phasor.ndim - weights.ndim
        shape = weights.shape + (1,) * ndim
        scale = np.sqrt(weights).reshape(shape)
        wavefront = wavefront.set(phasor=wavefront.phasor * scale)

        # Convert the propagated wavefront into sampled intensity
        intensity = self._to_intensity(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {
                "Wavefront": wavefront,
                "Intensity": intensity,
                "PSF": intensity,
            }
        if return_wf:
            return wavefront
        return intensity.data

    def model(self, source, return_all=False):
        """Model a source through the optical system.

        Parameters
        ----------
        source : Spectrum
            Source or source collection whose spectral and spatial parameters are
            resolved before propagation.
        return_all : bool
            Return the source's complete result mapping instead of only its modelled
            intensity array.

        Returns
        -------
        model : Intensity or dict
            Modelled deterministic intensity, or the source-defined result mapping
            when ``return_all=True``. This is the system-oriented alias of
            ``source.model(system, return_all)``.
        """
        if not isinstance(source, Spectrum):
            raise TypeError("source must be a Spectrum.")
        return source.model(self, return_all)

    def debug_propagate_mono(self, wavelength, offset=None):
        """Propagate one point-source wavefront and retain every layer state.

        Parameters
        ----------
        wavelength : float, metres
            Monochromatic source wavelength.
        offset : Array or None, radians
            Optional angular ``(x, y)`` source offset with shape ``(2,)``.

        Returns
        -------
        output : Wavefront
            Final propagated wavefront.
        states : dict[str, Wavefront]
            Ordered wavefront states, beginning with ``"initial_wavefront"`` and
            followed by one entry per named optical layer.
        """
        wavefront = self.initialise_wavefront(wavelength, offset)
        output, states = self.debug(wavefront)
        states = {
            "initial_wavefront": states["input"],
            **{name: value for name, value in states.items() if name != "input"},
        }
        return output, states


class DetectorSystem(LayeredSystem):
    """Apply deterministic detector transformations to an intensity.

    Examples
    --------
    Compose elementary detector effects while retaining deterministic intensity:

    ```python
    import jax.numpy as np

    import dLux as dl


    # Build the detector system
    detector = dl.DetectorSystem(
        layers=[
            ("jitter", dl.Jitter(sigma=0.5)),
            ("sensitivity", dl.Sensitivity(0.8)),
            ("bias", dl.Bias(5.0)),
        ]
    )

    # Make an input intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    intensity = dl.Intensity(np.ones((64, 64)), grid)

    # Apply the detector model
    intensity = detector(intensity)  # Returns an Intensity object
    ```
    """

    layers: OrderedDict

    def __init__(self, layers):
        """Initialise an ordered deterministic detector model.

        Parameters
        ----------
        layers : mapping or sequence
            Ordered detector layers, optionally supplied as ``(name, layer)`` pairs.
            Every layer must transform an `Intensity` into an `Intensity`.
        """
        super().__init__(layers, BaseDetectorLayer)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply every deterministic detector layer to an intensity.

        Parameters
        ----------
        intensity : Intensity
            Sampled deterministic intensity. Its grid and leading axes are preserved
            unless an individual detector layer explicitly changes them.

        Returns
        -------
        intensity : Intensity
            Final detector-plane intensity. No noise realisation or uncertainty is
            introduced, and the input object is not mutated.
        """
        if not isinstance(intensity, Intensity):
            raise TypeError("intensity must be an Intensity instance.")
        return super().apply(intensity)

    def model(self, intensity: Intensity, return_all=False):
        """Model deterministic detector effects on an intensity.

        Parameters
        ----------
        intensity : Intensity
            Sampled deterministic intensity accepted by the detector layers.
        return_all : bool
            Return a result mapping rather than the final intensity directly.

        Returns
        -------
        output : Intensity or dict[str, Intensity]
            Final intensity, or ``{"Intensity": output}`` when ``return_all=True``.
            Use `Image` explicitly when a realised exposure and uncertainty are
            required.
        """
        output = self(intensity)
        return {"Intensity": output} if return_all else output
