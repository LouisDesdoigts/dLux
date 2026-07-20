from __future__ import annotations
from jax import Array
import jax.numpy as np

__all__ = [
    "apply_jones",
    "rotate_jones",
    "linear_polariser",
    "retarder",
    #
    "horizontal_polariser",
    "vertical_polariser",
    "rhc_polariser",
    "lhc_polariser",
    "quarter_wave_plate",
    "half_wave_plate",
    "jones_to_stokes",
]


###
def horizontal_polariser() -> Array:
    return np.array([[1, 0], [0, 0]])  # Horizontal polariser


def vertical_polariser() -> Array:
    return np.array([[0, 0], [0, 1]])  # Vertical polariser


def rhc_polariser() -> Array:
    return np.array([[0.5, -0.5j], [0.5j, 0.5]])  # Right-handed circular polariser


def lhc_polariser() -> Array:
    return np.array([[0.5, 0.5j], [-0.5j, 0.5]])  # Left-handed circular polariser


def quarter_wave_plate(angle: Array) -> Array:
    jones = np.array([[1, 0], [0, 1j]])  # Fast axis horizontal
    return rotate_jones(jones, angle)


def half_wave_plate(angle: Array) -> Array:
    jones = np.array([[1, 0], [0, -1]])  # Fast axis horizontal
    return rotate_jones(jones, angle)


###


def apply_jones(jones: Array, phasor: Array) -> Array:
    """Applies a Jones matrix to a phasor"""
    return np.einsum("ij..., jk... -> ik...", jones, phasor)


def linear_polariser(angle: Array) -> Array:
    """
    This already handles dimensionality inherently. An input shape of (N, M) will
    produce an output shape of (2, 2, N, M).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c**2, c * s], [c * s, s**2]])


def retarder(retardance: Array, angle: Array) -> Array:
    """
    Constructs a retarder Jones matrix given retardance and angle. Dimensional inputs
    are handled explicitly by vectorising the construction and rotation operation. An
    inputs shape of (N, M) will produce an output shape of (2, 2, N, M).

    Note that we dont apply the vectorise operation via partial because we want to
    shift the output axes to the back of the array which isn't possible with the
    vectorize function.
    """
    fn = lambda r, a: rotate_jones(np.array([[1, 0], [0, np.exp(1j * r)]]), a)
    J = np.vectorize(fn, signature="(),()->(i,j)")(retardance, angle)
    return np.moveaxis(J, (-2, -1), (0, 1))


def rotate_jones(jones: Array, angle: Array | None) -> Array:
    """
    Rotate a Jones matrix by a given angle.
    Supports batched shapes where matrix dimensions are leading.

    Parameters
    ----------
    jones: Array
        The initial Jones matrix to be rotated, of shape (2, 2, ...).
    angle: Array, radians
        The angle by which to rotate the Jones matrix, of shape (...).

    Returns
    -------
    Array
        The rotated Jones matrix, of shape (2, 2, ...).
    """
    if angle is None:
        return jones

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
        "mi..., mn..., nj... -> ij...", rotation_matrix, jones, rotation_matrix
    )


# Pre calculate the A and its inverse
A = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1j, 1j, 0],  # Swapped 1j and -1j to match IAU
    ]
)

A_inv = np.linalg.inv(A)


def jones_to_stokes(J, stokes=None):
    """
    Note the einsum convention used allows us to vectorise over the trailing dimensions
    freely, but prevents us from vectorising over the leading dimensions because which
    dimensions are the jones dimensions becomes undefined.
    """
    # Build the Stokes vector if not provided. This is equivalent to unpolarised light.
    stokes = np.array([1.0, 0.0, 0.0, 0.0]) if stokes is None else stokes

    # Kronecker product mapping: row = 2*i + j, col = 2*k + l
    # We order the indices as i, j, k, l followed by the batch dimensions (...)
    J_kron = np.einsum("ik...,jl...->ijkl...", J, np.conj(J))

    # Reshape the (2, 2, 2, 2, ...) array into (4, 4, ...)
    J_kron = J_kron.reshape((4, 4) + J.shape[2:])

    # Perform matrix multiplication: M = A @ J_kron @ A_inv for each batch element
    M = np.einsum("xy,yz...,zw->xw...", A, J_kron, A_inv)

    # Perform the transformation from Jones to Stokes space and return the result
    return np.einsum("ab...,b->a...", M.real, stokes)
