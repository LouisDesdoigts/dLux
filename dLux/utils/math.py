from jax import lax, Array
import jax.numpy as np

__all__ = ["factorial", "triangular_number", "eval_basis", "nandiv"]


# TODO: Use lax.cond to make n == 0 a safe case
def factorial(n: float) -> float:
    """
    Calculate n! in a jax friendly way. Note that n == 0 is not a
    safe case.

    Parameters
    ----------
    n : float
        The value to calculate the factorial of.

    Returns
    -------
    n! : float
        The factorial of the value.
    """
    return lax.exp(lax.lgamma(n + 1.0))


def triangular_number(n: int) -> int:
    """
    Calculate the nth triangular number.

    Parameters
    ----------
    n : int
        The nth triangular number to calculate.

    Returns
    -------
    n : int
        The nth triangular number.
    """
    return n * (n + 1) / 2


def eval_basis(basis: Array, coefficients: Array) -> Array:
    """
    Performs an n-dimensional dot-product between the basis and coefficients
    arrays.

    Parameters
    ----------
    basis: Array
        The basis to use.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    """
    ndim = coefficients.ndim
    return np.tensordot(basis, coefficients, axes=2 * (tuple(range(ndim)),))


def nandiv(a, b, fill=np.inf):
    return np.where(b == 0, fill, a / b)
