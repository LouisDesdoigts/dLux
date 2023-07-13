from jax import lax

__all__ = ["factorial", "triangular_number"]


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
