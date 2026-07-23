"""Polarised optical layers and parameterised polarisation fields."""

from __future__ import annotations
import jax.numpy as np
import dLux.utils as dlu
from jax import Array


from ..parametric import BaseParametric
from ..wavefronts import Wavefront
from .unified_layers import BaseOpticalLayer

__all__ = [
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
]


class BasePolarisingOptic(BaseOpticalLayer):
    """
    Base class for layers that apply a Jones matrix to a wavefront.

    Subclasses expose Jones matrices with shape `(2, 2, ...)`, matching the
    polarisation utility convention. The trailing axes may be empty for global
    optics or spatial for spatially varying optics. Context-dependent layers resolve
    their Jones matrices when applied to a wavefront.
    """

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer Jones matrix to the input wavefront.

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


class PolarisingOptic(BasePolarisingOptic):
    """
    A polarising optic defined directly by a Jones matrix.

    Attributes
    ----------
    jones : Array
        Jones matrix with shape `(2, 2, ...)`.
    """

    jones: Array  # Concrete this as an array

    def __init__(self: PolarisingOptic, jones: Array):
        """
        Parameters
        ----------
        jones : Array
            Jones matrix with shape `(2, 2, ...)`.
        """
        self.jones = jones


class UniformPolarisingOptic(PolarisingOptic):
    """
    A spatially uniform Jones matrix optic.

    The input Jones matrix must have shape `(2, 2)`. If `orientation` is provided, the
    Jones matrix is rotated when the layer is applied.

    Attributes
    ----------
    jones : Array
        Spatially uniform Jones matrix with shape `(2, 2)`.
    orientation : Array or None
        Rotation angle in radians. If None, the Jones matrix is applied unchanged.
    """

    orientation: Array | None

    def __init__(
        self: UniformPolarisingOptic,
        jones: Array,
        orientation: Array | None = None,
    ):
        """
        Parameters
        ----------
        jones : Array
            Spatially uniform Jones matrix with shape `(2, 2)`.
        orientation : Array or None = None
            Rotation angle in radians.
        """
        self.orientation = orientation
        jones = np.asarray(jones)

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        super().__init__(jones)

    def __call__(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
        """
        Applies the rotated Jones matrix to the input wavefront.

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
    """
    An ideal linear polariser.

    `angle` is the transmission-axis angle measured counter-clockwise from the
    horizontal x-axis. It may be a scalar, array, or `Parametric` object. Parametric
    angles are evaluated against the input wavefront when the layer is applied.

    Attributes
    ----------
    angle : Array or Parametric
        Transmission-axis angle in radians.
    """

    angle: Array | BaseParametric

    def __init__(
        self: LinearPolariser,
        angle: Array | BaseParametric = 0.0,
    ):
        """
        Parameters
        ----------
        angle : Array or Parametric = 0.0
            Transmission-axis angle in radians.
        """
        self.angle = self.as_parametric(angle)

    def evaluate_angle(self: LinearPolariser, wavefront: Wavefront = None) -> Array:
        """Returns the transmission-axis angle evaluated in context."""
        return self.resolve(self.angle, wavefront=wavefront)

    def evaluate_jones(self: LinearPolariser, wavefront: Wavefront = None) -> Array:
        """Returns the Jones matrix evaluated in context."""
        return dlu.linear_polariser(self.evaluate_angle(wavefront))

    @property
    def jones(self: LinearPolariser) -> Array:
        """Returns the Jones matrix for context-independent angles."""
        return self.evaluate_jones()

    def __call__(self: LinearPolariser, wavefront: Wavefront) -> Wavefront:
        """Applies the linear polariser to the input wavefront."""
        return wavefront.apply_jones(self.evaluate_jones(wavefront))


class Retarder(BasePolarisingOptic):
    """
    A retarder with uniform or spatially varying parameters.

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

    retardance: Array | BaseParametric
    angle: Array | BaseParametric

    def __init__(
        self: Retarder,
        retardance: Array | BaseParametric,
        angle: Array | BaseParametric = 0.0,
    ):
        """
        Parameters
        ----------
        retardance : Array or Parametric
            Retardance in radians.
        angle : Array or Parametric = 0.0
            Fast-axis angle in radians.
        """
        self.retardance = self.as_parametric(retardance)
        self.angle = self.as_parametric(angle)

    def evaluate_retardance(self: Retarder, wavefront: Wavefront = None) -> Array:
        """Returns the retardance evaluated in context."""
        return self.resolve(self.retardance, wavefront=wavefront)

    def evaluate_angle(self: Retarder, wavefront: Wavefront = None) -> Array:
        """Returns the fast-axis angle evaluated in context."""
        return self.resolve(self.angle, wavefront=wavefront)

    def evaluate_jones(self: Retarder, wavefront: Wavefront = None) -> Array:
        """Returns the Jones matrix evaluated in context."""
        retardance = self.evaluate_retardance(wavefront)
        angle = self.evaluate_angle(wavefront)
        return dlu.retarder(retardance, angle)

    @property
    def jones(self: Retarder) -> Array:
        """Returns the Jones matrix for context-independent parameters."""
        return self.evaluate_jones()

    def __call__(self: Retarder, wavefront: Wavefront) -> Wavefront:
        """Applies the retarder to the input wavefront."""
        return wavefront.apply_jones(self.evaluate_jones(wavefront))
