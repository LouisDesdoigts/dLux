from __future__ import annotations
from jax import Array
import jax.numpy as np


from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront, PolarisedWavefront

__all__ = [
    "PolarisingOptic",
]


def jones_matrix_rotated(init_jones: Array, angle: Array) -> Array:
    """
    Rotate a Jones matrix by a given angle.
    Supports batched shapes where matrix dimensions are leading.

    Parameters
    ----------
    init_jones: Array
        The initial Jones matrix to be rotated, of shape (2, 2, ...).
    angle: Array, radians
        The angle by which to rotate the Jones matrix, of shape (...).

    Returns
    -------
    Array
        The rotated Jones matrix, of shape (2, 2, ...).
    """
    c = np.cos(angle)
    s = np.sin(angle)

    # If angle has shape (...), rotation_matrix automatically
    # inherits the shape (2, 2, ...)
    rotation_matrix = np.array(
        [
            [c, s],
            [-s, c],
        ]
    )

    # 'mi...' transposes the first matrix by swapping row/col indices (m, i)
    # 'mn...' is the initial jones matrix
    # 'nj...' is the un-transposed rotation matrix
    # 'ij...' ensures the output retains the (2, 2, ...) structure
    return np.einsum(
        "mi..., mn..., nj... -> ij...", rotation_matrix, init_jones, rotation_matrix
    )


class PolarisingOptic(OpticalLayer):
    """
    A basic 'PolarisingOptic' class, which applies a polarisation transformation to the input wavefront.
    """

    jones_matrix: Array

    def __init__(
        self: PolarisingOptic,
        jones_matrix: Array,
    ):
        if jones_matrix.shape == (2, 2):
            jones_matrix = jones_matrix[:, :, np.newaxis, np.newaxis]

        self.jones_matrix = jones_matrix  # (2,2,n_pixels,n_pixels) or (2,2,1,1)

    def _promote_if_needed(
        self: PolarisingOptic, wavefront: Wavefront
    ) -> PolarisedWavefront:
        if isinstance(wavefront, PolarisedWavefront):
            return wavefront
        else:
            return PolarisedWavefront.from_wavefront(wavefront)

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        pol_wavefront = self._promote_if_needed(wavefront)

        # Apply the Jones matrix to the wavefront's phasor
        # jones_matrix: (2, 2, ...) -> 'ij...'
        # phasor:       (2, 2, n_pix, n_pix) -> 'jk...'
        # output:       (2, 2, n_pix, n_pix) -> 'ik...'
        new_phasor = np.einsum(
            "ij..., jk... -> ik...",
            self.jones_matrix,
            pol_wavefront.phasor,
        )

        pol_wavefront = pol_wavefront.set({"phasor": new_phasor})
        return pol_wavefront


class UniformPolarisingOptic(PolarisingOptic):
    """
    A spatially uniform Jones matrix optic, which applies the same polarisation transformation
    across the entire wavefront. As such optics can be easily rotated, they also support an optional
    'angle' parameter, which rotates the Jones matrix by the specified angle before applying it to the wavefront.
    """

    angle: Array | None

    def __init__(
        self: UniformPolarisingOptic,
        jones_matrix: Array,
        angle: Array | None = None,
        inital_angle: Array | None = None,
    ):
        if inital_angle is not None and angle is not None:
            raise ValueError(
                "Cannot specify both 'angle' and 'initial_angle' parameters as this leads to confusion."
            )

        if jones_matrix.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")

        if inital_angle is not None:
            jones_matrix = jones_matrix_rotated(jones_matrix, inital_angle)

        super().__init__(jones_matrix)

        self.angle = angle

    def __call__(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
        pol_wavefront = self._promote_if_needed(wavefront)

        if self.angle is not None:
            rotated_jones = jones_matrix_rotated(self.jones_matrix, self.angle)
        else:
            rotated_jones = self.jones_matrix

        new_phasor = np.einsum(
            "ij..., jk... -> ik...",
            rotated_jones,
            pol_wavefront.phasor,
        )

        pol_wavefront = pol_wavefront.set({"phasor": new_phasor})
        return pol_wavefront


class LinearPolariser(UniformPolarisingOptic):
    """
    A linear polariser, which can be oriented at any angle. The Jones matrix for a linear polariser is given by:

    [[cos^2(theta), cos(theta)sin(theta)],
     [cos(theta)sin(theta), sin^2(theta)]]

    where theta is the angle of the polariser's transmission axis relative to the horizontal.
    """

    def __init__(
        self: LinearPolariser,
        angle: Array | None = None,
        inital_angle: Array | None = None,
    ):
        jones_matrix = np.array([[1, 0], [0, 0]])  # Horizontal polariser
        super().__init__(jones_matrix, angle, inital_angle)


class CircularPolariser(UniformPolarisingOptic):
    """
    A circular polariser, which can be oriented at any angle. The Jones matrix for a circular polariser is given by:

    [[0.5, -0.5j],
     [0.5j, 0.5]]

    for a right-handed circular polariser, and the complex conjugate of this for a left-handed circular polariser.
    """

    def __init__(
        self: CircularPolariser,
        angle: Array | None = None,
        inital_angle: Array | None = None,
    ):
        jones_matrix = np.array(
            [[0.5, -0.5j], [0.5j, 0.5]]
        )  # Right-handed circular polariser
        super().__init__(jones_matrix, angle, inital_angle)


class QuarterWavePlate(UniformPolarisingOptic):
    """
    A quarter wave plate, which can be oriented at any angle. The Jones matrix for a quarter wave plate is given by:

    [[1, 0],
     [0, 1j]]

    when the fast axis is horizontal, and rotated versions of this for other orientations.
    """

    def __init__(
        self: QuarterWavePlate,
        angle: Array | None = None,
        inital_angle: Array | None = None,
    ):
        jones_matrix = np.array([[1, 0], [0, 1j]])  # Fast axis horizontal
        super().__init__(jones_matrix, angle, inital_angle)


class HalfWavePlate(UniformPolarisingOptic):
    """
    A half wave plate, which can be oriented at any angle. The Jones matrix for a half wave plate is given by:

    [[1, 0],
     [0, -1]]

    when the fast axis is horizontal, and rotated versions of this for other orientations.
    """

    def __init__(
        self: HalfWavePlate,
        angle: Array | None = None,
        inital_angle: Array | None = None,
    ):
        jones_matrix = np.array([[1, 0], [0, -1]])  # Fast axis horizontal
        super().__init__(jones_matrix, angle, inital_angle)
