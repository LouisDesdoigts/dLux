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

    Parameters
    ----------
    init_jones: Array
        The initial Jones matrix to be rotated, of shape (2, 2).
    angle: Array, radians
        The angle by which to rotate the Jones matrix.

    Returns
    -------
    Array
        The rotated Jones matrix, of shape (2, 2).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array(
        [
            [c, s],
            [-s, c],
        ]
    )
    return rotation_matrix.T @ init_jones @ rotation_matrix


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

        # Rotate the Jones matrix if needed
        if self.angle != 0.0:
            jones_matrix = jones_matrix_rotated(self.jones_matrix, self.angle)
        else:
            jones_matrix = self.jones_matrix

        # Apply the Jones matrix to the wavefront's phasor
        new_phasor = np.einsum(
            "ij, jklm -> iklm",
            jones_matrix,
            pol_wavefront.phasor,
        )
        pol_wavefront = pol_wavefront.set({"phasor": new_phasor})
        return pol_wavefront


class IdealPolariser(PolarisingOptic):
    """
    An ideal polariser, which fully transmits one linear polarisation and fully absorbs the other.

    """
