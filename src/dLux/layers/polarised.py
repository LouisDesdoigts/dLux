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
    angle: Array

    def __init__(
        self: PolarisingOptic,
        jones_matrix: Array,
        angle: Array = 0.0,
    ):
        if jones_matrix.shape == (2, 2):
            jones_matrix = jones_matrix[:, :, np.newaxis, np.newaxis]

        self.jones_matrix = jones_matrix  # (2,2,n_pixels,n_pixels) or (2,2,1,1)
        self.angle = np.asarray(angle)

    def _promote_if_needed(
        self: PolarisingOptic, wavefront: Wavefront
    ) -> PolarisedWavefront:
        if isinstance(wavefront, PolarisedWavefront):
            return wavefront
        else:
            return PolarisedWavefront.from_wavefront(wavefront)

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        pol_wavefront = self._promote_if_needed(wavefront)

        # Rotate the Jones matrix if needed (now returns shape (2, 2, ...))
        if self.angle != 0.0:
            jones_matrix = jones_matrix_rotated(self.jones_matrix, self.angle)
        else:
            jones_matrix = self.jones_matrix

        # Apply the Jones matrix to the wavefront's phasor
        # jones_matrix: (2, 2, ...) -> 'ij...'
        # phasor:       (2, 2, n_pix, n_pix) -> 'jk...'
        # output:       (2, 2, n_pix, n_pix) -> 'ik...'
        new_phasor = np.einsum(
            "ij..., jk... -> ik...",
            jones_matrix,
            pol_wavefront.phasor,
        )

        pol_wavefront = pol_wavefront.set({"phasor": new_phasor})
        return pol_wavefront


class IdealPolariser(PolarisingOptic):
    """
    An ideal polariser, which fully transmits one linear polarisation and fully absorbs the other.

    """
