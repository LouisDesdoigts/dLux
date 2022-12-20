from jax import lax


__all__ = ["factorial"]

 
def factorial(n : int) -> int:
    """
    Calculate n! in a jax friendly way. Note that n == 0 is not a 
    safe case.  
 
    Parameters:
    -----------
    n : int
        The integer to calculate the factorial of.
 
    Returns:
    --------
    n! : int
        The factorial of the integer
    """
    return lax.exp(lax.lgamma(n + 1.))
