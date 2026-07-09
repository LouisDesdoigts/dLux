from __future__ import annotations
from jax import Array
import dLux.utils as dlu


from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront

__all__ = [
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
]


class PolarisingOptic(OpticalLayer):
    """
    A basic 'PolarisingOptic' class, which applies a polarisation transformation to the
    input wavefront.
    """

    jones: Array

    def __init__(self: PolarisingOptic, jones: Array):
        self.jones = jones

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        return wavefront.apply_jones(self.jones)


class UniformPolarisingOptic(PolarisingOptic):
    """
    A spatially uniform Jones matrix optic, which applies the same polarisation
    transformation across the entire wavefront. As such optics can be easily rotated,
    they also support an optional 'angle' parameter, which rotates the Jones matrix by
    the specified angle before applying it to the wavefront.
    """

    angle: Array | None

    def __init__(
        self: UniformPolarisingOptic,
        jones: Array,
        angle: Array | None = None,
    ):
        self.angle = angle

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        super().__init__(jones)

    def __call__(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
        return wavefront.apply_jones(dlu.rotate_jones(self.jones, self.angle))


class LinearPolariser(UniformPolarisingOptic):
    """
    A linear polariser, which can be oriented at any angle. The Jones matrix for a
    linear polariser is given by:

    [[cos^2(theta), cos(theta)sin(theta)],
     [cos(theta)sin(theta), sin^2(theta)]]

    where theta is the angle of the polariser's transmission axis relative to the
    horizontal.
    """

    def __init__(self: LinearPolariser, angle: Array | None = None):
        super().__init__(dlu.linear_polariser(0.0), angle)


class Retarder(UniformPolarisingOptic):
    """
    A retarder, which can be oriented at any angle. The Jones matrix for a retarder is
    given by:

    [[1, 0],
     [0, exp(i * delta)]]

    where delta is the retardance of the retarder. The fast axis of the retarder is
    assumed to be horizontal, and the Jones matrix can be rotated to any angle using
    the 'angle' parameter.
    """

    def __init__(
        self: Retarder,
        retardance: Array,
        angle: Array | None = None,
    ):
        super().__init__(dlu.retarder(retardance, 0.0), angle)
