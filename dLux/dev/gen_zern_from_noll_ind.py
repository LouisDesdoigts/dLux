import jax 
import jax.numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dLux as dl

def factorial(n : int) -> int:
    """
    Calculate n! in a jax friendly way. Note that n == 0 is not a 
    safe case.  

    Parameters
    ----------
    n : int
        The integer to calculate the factorial of.

    Returns
    n! : int
        The factorial of the integer
    """
    return jax.lax.exp(jax.lax.lgamma(n + 1.))


def noll_index(j: int) -> tuple:
    """
    Decode the jth noll index of the zernike polynomials. This 
    arrises because the zernike polynomials are parametrised by 
    a pair numbers, e.g. n, m, but we want to impose an order.
    The noll indices are the standard way to do this see [this]
    (https://oeis.org/A176988) for more detail. The top of the 
    mapping between the noll index and the pair of numbers is 
    shown below:

    n, m Indices
    ------------
    (0, 0)
    (1, -1), (1, 1)
    (2, -2), (2, 0), (2, 2)
    (3, -3), (3, -1), (3, 1), (3, 3)

    Noll Indices
    ------------
    1
    3, 2
    5, 4, 6
    9, 7, 8, 10

    Parameters
    ----------
    j : int
        The noll index to decode.
    
    Returns
    -------
    n, m : tuple
        The n, m parameters of the zernike polynomial.
    """
    # To retrive the row that we are in we use the formula for 
    # the sum of the integers:
    #  
    #  n      n(n + 1)
    # sum i = -------- = x_{n}
    # i=0        2
    # 
    # However, `j` is a number between x_{n - 1} and x_{n} to 
    # retrieve the 0th based index we want the upper bound. 
    # Applying the quadratic formula:
    # 
    # n = -1/2 + sqrt(1 + 8x_{n})/2
    #
    # We know that n is an integer and hence of x_{n} -> j where 
    # j is not an exact solution the row can be found by taking 
    # the floor of the calculation. 
    #
    # n = (-1/2 + sqrt(1 + 8j)/2) // 1
    #
    # All the odd noll indices map to negative m integers and also 
    # 0. The sign can therefore be determined by -(j & 1). 
    # This works because (j & 1) returns the rightmost bit in 
    # binary representation of j. This is equivalent to -(j % 2).
    # 
    # The m indices range from -n to n in increments of 2. The last 
    # thing to do is work out how many times to add two to -n. 
    # This can be done by banding j away from the smallest j in 
    # the row. 
    #
    # The smallest j in the row can be calculated using the sum of
    # integers formula in the comments above with n = (n - 1) and
    # then adding one. Let this number be (x_{n - 1} + 1). We can 
    # then subtract j from it to get r = (j - x_{n - 1} + 1)
    #
    # The odd and even cases work differently. I have included the 
    # formula below:
    # odd : p = (j - x_{n - 1}) // 2 
   
    # even: p = (j - x_{n - 1} + 1) // 2
    # where p represents the number of times 2 needs to be added
    # to the base case. The 1 required for the even case can be 
    # generated in place using ~(j & 1) + 2, which is 1 for all 
    # even numbers and 0 for all odd numbers.
    #
    # For odd n the base case is 1 and for even n it is 0. This 
    # is the result of the bitwise operation j & 1 or alternatively
    # (j % 2). The final thing is adding the sign to m which is 
    # determined by whether j is even or odd hence -(j & 1).
    n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
    smallest_j_in_row = n * (n + 1) / 2 + 1 
    number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
    sign_of_shift = -(j & 1) + ~(j & 1) + 2
    base_case = (n & 1)
    m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
    return n, m


def jth_radial_zernike(n: int, m: int) -> list:
    """
    The radial zernike polynomial.

    Parameters
    ----------
    n : int
        The first index number of the zernike polynomial to forge
    m : int 
        The second index number of the zernike polynomial to forge.

    Returns
    -------
    radial : Tensor
        An npix by npix stack of radial zernike polynomials.
    """
    MAX_DIFF = 5
    m, n = np.abs(m), np.abs(n)
    upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1

    k = np.arange(MAX_DIFF)
    mask = (k < upper).reshape(MAX_DIFF, 1, 1)
    coefficients = (-1) ** k * factorial(n - k) / \
        (factorial(k) * \
            factorial(((n + m) / 2).astype(int) - k) * \
            factorial(((n - m) / 2).astype(int) - k))

    def _jth_radial_zernike(rho: list) -> list:
        rho = np.tile(rho, (MAX_DIFF, 1, 1))
        coeffs = coefficients.reshape(MAX_DIFF, 1, 1)
        rads = rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1)
        return (coeffs * mask * rads).sum(axis = 0)
            
    return _jth_radial_zernike


def jth_polar_zernike(n: int, m: int) -> list:
    is_m_zero = (m != 0).astype(int)
    norm_coeff = (1 + (np.sqrt(2) - 1) * is_m_zero) * np.sqrt(n + 1)

    # When m < 0 we have the odd zernike polynomials which are 
    # the radial zernike polynomials multiplied by a sine term.
    # When m > 0 we have the even sernike polynomials which are 
    # the radial polynomials multiplies by a cosine term. 
    # To produce this result without logic we can use the fact
    # that sine and cosine are separated by a phase of pi / 2
    # hence by casting int(m < 0) we can add the nessecary phase.

    phase_mod = (m < 0).astype(int) * np.pi / 2

    def _jth_polar_zernike(theta: list) -> list:
        return norm_coeff * np.cos(np.abs(m) * theta - phase_mod)

    return _jth_polar_zernike  


def jth_zernike(j: int) -> list:
    """
    Calculate the zernike basis on a square pixel grid. 

    Parameters
    ----------
    noll_index: int
        The noll index corresponding to the zernike to generate.
        The first ten zernikes have been computed analytically 
        and are available via the `PreCompZernikeBasis` class. 
        This is only for doing zernike terms that are of higher 
        order and not centered.

    Returns
    -------
    zernike : Tensor 
        The zernike polynomials evaluated until number. The shape
        of the output tensor is number by pixels by pixels. 
    """
    n, m = noll_index(j)

    def _jth_zernike(coords: list) -> list:
        polar_coords = dl.utils.cartesian_to_polar(coords)
        rho = polar_coords[0]
        theta = polar_coords[1]
        aperture = rho <= 1.
        _jth_rad_zern = jth_radial_zernike(n, m)
        _jth_pol_zern = jth_polar_zernike(n, m)
        return aperture * _jth_rad_zern(rho) * _jth_pol_zern(theta)
    
    return _jth_zernike 

coords = dl.utils.get_pixel_coordinates(128, 2. / 128)

fourth_zern = jax.jit(jth_zernike(3))

# %%timeit
fourth_zern(coords)

from hard_coded_zernike import zernikes

fourth_zern = jax.jit(zernikes[3])

# %%timeit
polar_coords = dl.utils.cartesian_to_polar(coords)
rho = polar_coords[0]
theta = polar_coords[1]
aperture = rho <= 1.
aperture * fourth_zern(rho, theta)

fig, axes = plt.subplots(5, 5, figsize=(5*4, 5*3))
for j in np.arange(25):
    jth_zern = jth_zernike(j)(coords)
    _map = axes[j // 5][j % 5].imshow(jth_zern)
    fig.colorbar(_map, ax=axes[j // 5][j % 5])

plt.show()

