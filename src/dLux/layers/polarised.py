"""Polarised optical layers and parameterised polarisation fields."""

from __future__ import annotations

import dLux.utils as dlu
import zodiax as zdx
from jax import Array


from ..fields import Wavefront
from .optical import OpticalLayer

__all__ = [
    "BasePolarisingOptic",
    "PolarisationLayer",
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
]


class BasePolarisingOptic(OpticalLayer):
    """Base class for layers that apply a Jones matrix to a wavefront.

    Subclasses expose Jones matrices with shape `(2, 2, ...)`, matching the
    polarisation utility convention. The trailing axes may be empty for global
    optics or spatial for spatially varying optics. Context-dependent layers resolve
    their Jones matrices when applied to a wavefront.

    Scalar inputs are promoted to `PolarisedWavefront`. Leading wavelength or
    parameter axes follow ordinary JAX broadcasting around the Jones axes.
    """

    def apply_mono(self: PolarisingOptic, wavefront: Wavefront, **context) -> Wavefront:
        """Applies the layer Jones matrix to the input wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront. Non-polarised wavefronts are promoted by
            `Wavefront.apply_jones`.

        Returns
        -------
        wavefront : Wavefront
            Wavefront after applying the Jones matrix.
        """
        self = self.resolve(**self.context(wavefront, **context))
        return wavefront.apply_jones(self.jones)


class PolarisationLayer(OpticalLayer):
    """Apply an ordered collection of polarising optics as one layer.

    The first optic promotes a scalar input to `PolarisedWavefront` when required.
    Subsequent Jones operations are applied in insertion order while preserving
    wavelength and spatial vectorisation.
    """

    polarisation: dict | None = None

    def __init__(self, polarisation=None, *, alias=None):
        """Initialise an ordered collection of polarising optics.

        Parameters
        ----------
        polarisation : BasePolarisingOptic, sequence, or None
            One optic or a sequence of optics, optionally named with
            ``(name, optic)`` pairs. ``None`` leaves the wavefront unchanged.
        """
        if polarisation is None:
            self.polarisation = None
            self.alias = alias
            return
        items = (
            list(polarisation)
            if isinstance(polarisation, (list, tuple))
            else [polarisation]
        )
        self.polarisation = dlu.list2dictionary(items, True, BasePolarisingOptic)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply every configured polarising optic in insertion order.

        Scalar wavefronts are promoted when the first Jones optic is applied. The
        returned polarised wavefront retains wavelength and grid metadata.
        """
        if self.polarisation is not None:
            for optic in self.polarisation.values():
                wavefront = optic(wavefront, **context)
        return wavefront


class PolarisingOptic(BasePolarisingOptic):
    """A polarising optic defined directly by a Jones matrix.

    Attributes
    ----------
    jones : Array or zodiax.Expression
        Jones matrix with shape `(2, 2, ...)`.

    Examples
    --------
    Apply uniform and spatially varying Jones matrices:

    ```python
    import jax.numpy as np
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a polarising optic from a uniform Jones matrix
    jones = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0j],
        ]
    )
    optic = dl.PolarisingOptic(jones=jones)

    # Apply it to a scalar wavefront
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)

    # The output is promoted to a polarised wavefront
    stokes = wavefront.stokes()
    intensity = wavefront.intensity

    # Construct a spatially varying linear polariser
    angle = jr.uniform(jr.key(0), (128, 128), minval=0, maxval=np.pi)
    jones = dlu.linear_polariser(angle)
    optic = dl.PolarisingOptic(jones=jones)

    # Apply the spatial Jones field to a wavefront
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)
    ```
    """

    jones: Array | zdx.Expression

    def __init__(self: PolarisingOptic, jones: Array | zdx.Expression, *, alias=None):
        """Parameters
        ----------
        jones : Array or zodiax.Expression
            Jones matrix with shape `(2, 2, ...)`.
        """
        self.jones = zdx.as_array(jones, dtype=None)

        self.alias = alias


class UniformPolarisingOptic(PolarisingOptic):
    """A spatially uniform Jones matrix optic.

    The input Jones matrix has shape `(2, 2)` and broadcasts across every spatial
    sample. Matrix and orientation expressions resolve at the incident grid and
    wavelength. If `orientation` is provided, the matrix is rotated when applied.

    Attributes
    ----------
    jones : Array or zodiax.Expression
        Spatially uniform Jones matrix with shape `(2, 2)`.
    orientation : Array, zodiax.Expression, or None
        Rotation angle in radians. If None, the Jones matrix is applied unchanged.
    """

    jones: Array | zdx.Expression
    orientation: Array | zdx.Expression | None = None

    def __init__(
        self: UniformPolarisingOptic,
        jones: Array | zdx.Expression,
        orientation: Array | zdx.Expression | None = None,
        *,
        alias=None,
    ):
        """Parameters
        ----------
        jones : Array or zodiax.Expression
            Spatially uniform Jones matrix with shape `(2, 2)`.
        orientation : Array, zodiax.Expression, or None = None
            Rotation angle in radians.
        """
        orientation = zdx.as_array(orientation, dtype=float)
        jones = zdx.as_array(jones, dtype=None)

        if not isinstance(jones, zdx.Expression) and jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        self.orientation = orientation
        super().__init__(jones)

        self.alias = alias

    def apply_mono(
        self: UniformPolarisingOptic, wavefront: Wavefront, **context
    ) -> Wavefront:
        """Applies the rotated Jones matrix to the input wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.

        Returns
        -------
        wavefront : Wavefront
            Wavefront after applying the rotated Jones matrix.
        """
        self = self.resolve(**self.context(wavefront, **context))
        if self.jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        return wavefront.apply_jones(dlu.rotate_jones(self.jones, self.orientation))


class LinearPolariser(BasePolarisingOptic):
    """An ideal linear polariser.

    `angle` is the transmission-axis angle measured counter-clockwise from the
    horizontal x-axis. It may be a scalar, array, or `Parametric` object. Parametric
    angles are evaluated against the input wavefront when the layer is applied.

    Attributes
    ----------
    angle : Array or Parametric
        Transmission-axis angle in radians.
    """

    angle: Array | zdx.Expression

    def __init__(
        self: LinearPolariser, angle: Array | zdx.Expression = 0.0, *, alias=None
    ):
        """Parameters
        ----------
        angle : Array or Parametric = 0.0
            Transmission-axis angle in radians.
        """
        self.angle = zdx.as_array(angle, dtype=float)

        self.alias = alias

    @property
    def jones(self: LinearPolariser) -> Array:
        """Return the spatially uniform ``(2, 2)`` Jones matrix.

        The configured transmission-axis angle is interpreted in radians.
        """
        return dlu.linear_polariser(self.angle)

    def apply_mono(self: LinearPolariser, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the linear-polariser Jones matrix to one wavefront.

        Parametric angle is resolved from the wavefront context. Scalar wavefronts
        are promoted and all sampling metadata are preserved.
        """
        self = self.resolve(**self.context(wavefront, **context))
        return wavefront.apply_jones(dlu.linear_polariser(self.angle))


class Retarder(BasePolarisingOptic):
    """A retarder with uniform or spatially varying parameters.

    `retardance` is the phase delay of the vertical component relative to horizontal.
    `angle` rotates the fast axis counter-clockwise from horizontal. Both parameters
    may be scalars, arrays, or `Parametric` objects and are evaluated against the
    input wavefront when the layer is applied.

    Attributes
    ----------
    retardance : Array or Parametric
        Retardance in radians.
    angle : Array or Parametric
        Fast-axis angle in radians.
    """

    retardance: Array | zdx.Expression
    angle: Array | zdx.Expression

    def __init__(
        self: Retarder,
        retardance: Array | zdx.Expression,
        angle: Array | zdx.Expression = 0.0,
        *,
        alias=None,
    ):
        """Parameters
        ----------
        retardance : Array or Parametric
            Retardance in radians.
        angle : Array or Parametric = 0.0
            Fast-axis angle in radians.
        """
        self.retardance = zdx.as_array(retardance, dtype=float)
        self.angle = zdx.as_array(angle, dtype=float)

        self.alias = alias

    @property
    def jones(self: Retarder) -> Array:
        """Return the spatially uniform ``(2, 2)`` retarder Jones matrix.

        Retardance and fast-axis angle are interpreted in radians.
        """
        return dlu.retarder(self.retardance, self.angle)

    def apply_mono(self: Retarder, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the retarder Jones matrix to one wavefront.

        Parametric retardance and angle are resolved from the wavefront context.
        Scalar wavefronts are promoted and all sampling metadata are preserved.
        """
        self = self.resolve(**self.context(wavefront, **context))
        return wavefront.apply_jones(dlu.retarder(self.retardance, self.angle))
