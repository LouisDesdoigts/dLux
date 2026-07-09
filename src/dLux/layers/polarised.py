from __future__ import annotations
from jax import Array
import dLux.utils as dlu


from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront

__all__ = [
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "CircularPolariser",
    "QuarterWavePlate",
    "HalfWavePlate",
]


class PolarisingOptic(OpticalLayer):
    """
    A basic 'PolarisingOptic' class, which applies a polarisation transformation to the
    input wavefront.
    """

    jones: Array

    def __init__(
        self: PolarisingOptic,
        jones: Array,
    ):
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
        initial_angle: Array | None = None,
    ):
        if initial_angle is not None and angle is not None:
            raise ValueError(
                "Cannot specify both 'angle' and 'initial_angle' parameters as this "
                "leads to confusion."
            )

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")

        if initial_angle is not None:
            jones = dlu.rotate_jones(jones, initial_angle)

        super().__init__(jones)

        self.angle = angle

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

    def __init__(
        self: LinearPolariser,
        angle: Array | None = None,
        initial_angle: Array | None = None,
    ):
        jones = dlu.linear_polariser(0.0)
        super().__init__(jones, angle, initial_angle)


class CircularPolariser(UniformPolarisingOptic):
    """
    A circular polariser, which can be oriented at any angle. The Jones matrix for a
    circular polariser is given by:

    [[0.5, -0.5j],
     [0.5j, 0.5]]

    for a right-handed circular polariser, and the complex conjugate of this for a
    left-handed circular polariser.
    """

    def __init__(
        self: CircularPolariser,
        angle: Array | None = None,
        initial_angle: Array | None = None,
    ):
        super().__init__(dlu.rhc_polariser(), angle, initial_angle)


class QuarterWavePlate(UniformPolarisingOptic):
    """
    A quarter wave plate, which can be oriented at any angle. The Jones matrix for a
    quarter wave plate is given by:

    [[1, 0],
     [0, 1j]]

    when the fast axis is horizontal, and rotated versions of this for other
    orientations.
    """

    def __init__(
        self: QuarterWavePlate,
        angle: Array | None = None,
        initial_angle: Array | None = None,
    ):
        super().__init__(dlu.quarter_wave_plate(0.0), angle, initial_angle)


class HalfWavePlate(UniformPolarisingOptic):
    """
    A half wave plate, which can be oriented at any angle. The Jones matrix for a half
    wave plate is given by:

    [[1, 0],
     [0, -1]]

    when the fast axis is horizontal, and rotated versions of this for other
    orientations.
    """

    def __init__(
        self: HalfWavePlate,
        angle: Array | None = None,
        initial_angle: Array | None = None,
    ):
        super().__init__(dlu.half_wave_plate(0.0), angle, initial_angle)
