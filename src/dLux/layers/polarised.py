"""Polarised optical layers and parameterised polarisation fields."""

from __future__ import annotations
import dLux.utils as dlu
from jax import Array


from ..parametric import Parametric
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

    def apply_mono(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
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
        return wavefront.apply_jones(self.jones)


class PolarisationLayer(OpticalLayer):
    """Apply an ordered collection of polarising optics as one layer.

    The first optic promotes a scalar input to `PolarisedWavefront` when required.
    Subsequent Jones operations are applied in insertion order while preserving
    wavelength and spatial vectorisation.
    """

    polarisation: dict | None

    def __init__(self, polarisation=None):
        """Initialise a fixed or parametric Jones transformation.

        Parameters
        ----------
        polarisation : Array, Parametric, or None
            Jones matrix with leading or spatial dimensions supported by the input
            polarised wavefront. ``None`` uses the identity transformation.
        """
        if polarisation is None:
            self.polarisation = None
            return
        items = (
            list(polarisation)
            if isinstance(polarisation, (list, tuple))
            else [polarisation]
        )
        self.polarisation = dlu.list2dictionary(items, True, BasePolarisingOptic)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply every configured polarising optic in insertion order.

        Scalar wavefronts are promoted when the first Jones optic is applied. The
        returned polarised wavefront retains wavelength and grid metadata.
        """
        if self.polarisation is not None:
            for optic in self.polarisation.values():
                wavefront = optic(wavefront)
        return wavefront


class PolarisingOptic(BasePolarisingOptic):
    """A polarising optic defined directly by a Jones matrix.

    Attributes
    ----------
    jones : Array
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

    jones: Array

    def __init__(self: PolarisingOptic, jones: Array):
        """Parameters
        ----------
        jones : Array
            Jones matrix with shape `(2, 2, ...)`.
        """
        self.jones = dlu.to_value(jones, dtype=None, name="jones")


class UniformPolarisingOptic(PolarisingOptic):
    """A spatially uniform Jones matrix optic.

    The input Jones matrix must have trailing shape `(2, 2)` and broadcasts across
    every spatial sample. Leading axes can represent wavelength-dependent or batched
    matrices. If `orientation` is provided, the matrix is rotated when applied.

    Attributes
    ----------
    jones : Array
        Spatially uniform Jones matrix with shape `(2, 2)`.
    orientation : Array or None
        Rotation angle in radians. If None, the Jones matrix is applied unchanged.
    """

    jones: Array
    orientation: Array | None

    def __init__(
        self: UniformPolarisingOptic, jones: Array, orientation: Array | None = None
    ):
        """Parameters
        ----------
        jones : Array
            Spatially uniform Jones matrix with shape `(2, 2)`.
        orientation : Array or None = None
            Rotation angle in radians.
        """
        orientation = dlu.to_value(orientation, optional=True, name="orientation")
        jones = dlu.to_value(jones, dtype=None, name="jones")

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        self.orientation = orientation
        super().__init__(jones)

    def apply_mono(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
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

    angle: Array | Parametric

    def __init__(self: LinearPolariser, angle: Array | Parametric = 0.0):
        """Parameters
        ----------
        angle : Array or Parametric = 0.0
            Transmission-axis angle in radians.
        """
        self.angle = dlu.to_value(angle, types=Parametric)

    @property
    def jones(self: LinearPolariser) -> Array:
        """Return the spatially uniform ``(2, 2)`` Jones matrix.

        The configured transmission-axis angle is interpreted in radians.
        """
        return dlu.linear_polariser(self.angle)

    def apply_mono(self: LinearPolariser, wavefront: Wavefront) -> Wavefront:
        """Apply the linear-polariser Jones matrix to one wavefront.

        Parametric angle is resolved from the wavefront context. Scalar wavefronts
        are promoted and all sampling metadata are preserved.
        """
        self = self.resolve(wavefront=wavefront)
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

    retardance: Array | Parametric
    angle: Array | Parametric

    def __init__(
        self: Retarder, retardance: Array | Parametric, angle: Array | Parametric = 0.0
    ):
        """Parameters
        ----------
        retardance : Array or Parametric
            Retardance in radians.
        angle : Array or Parametric = 0.0
            Fast-axis angle in radians.
        """
        self.retardance = dlu.to_value(retardance, types=Parametric)
        self.angle = dlu.to_value(angle, types=Parametric)

    @property
    def jones(self: Retarder) -> Array:
        """Return the spatially uniform ``(2, 2)`` retarder Jones matrix.

        Retardance and fast-axis angle are interpreted in radians.
        """
        return dlu.retarder(self.retardance, self.angle)

    def apply_mono(self: Retarder, wavefront: Wavefront) -> Wavefront:
        """Apply the retarder Jones matrix to one wavefront.

        Parametric retardance and angle are resolved from the wavefront context.
        Scalar wavefronts are promoted and all sampling metadata are preserved.
        """
        self = self.resolve(wavefront=wavefront)
        return wavefront.apply_jones(dlu.retarder(self.retardance, self.angle))
