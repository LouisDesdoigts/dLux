import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dLux as dl
import jax

jax.config.update("jax_debug_nans", True)

nolls: float = np.arange(3, 9, dtype = int)
coeffs: float = np.ones((6,), dtype = float)
aperture: object = dl.CircularAperture(1.)
basis: object = dl.AberratedAperture(nolls, coeffs, aperture)
npix: int = 128
width: float = 2.

def noll_index(j: int) -> tuple:
    n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
    smallest_j_in_row = n * (n + 1) / 2 + 1 
    number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
    sign_of_shift = -(j & 1) + ~(j & 1) + 2
    base_case = (n & 1)
    m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
    return n, m


def jth_radial_zernike(n: int, m: int) -> callable:
    MAX_DIFF = 5
    m, n = np.abs(m), np.abs(n)
    upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1

    k = np.arange(MAX_DIFF)
    mask = (k < upper)[:, None, None]

    print(n - k)

    coefficients = (-1) ** k * dl.utils.math.factorial(np.abs(n - k)) / \
        (dl.utils.math.factorial(k) * \
            dl.utils.math.factorial(((n + m) / 2).astype(int) - k) * \
            dl.utils.math.factorial(((n - m) / 2).astype(int) - k))

    def _jth_radial_zernike(rho: list) -> list:
        rho = np.tile(rho, (MAX_DIFF, 1, 1))
        coefficients_out = coefficients[:, None, None]
        rads = rho ** (n - 2 * k)[:, None, None]
        return (coefficients_out * mask * rads).sum(axis = 0)
           
    return _jth_radial_zernike


def jth_polar_zernike(n: int, m: int) -> callable:
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


def jth_zernike(j: int) -> callable:
    n, m = noll_index(j)

    def _jth_zernike(coordinates: list) -> list:
        polar_coordinates = dl.utils.cartesian_to_polar(coordinates)
        rho = polar_coordinates[0]
        theta = polar_coordinates[1]
        aperture = rho <= 1.
        _jth_rad_zern = jth_radial_zernike(n, m)
        _jth_pol_zern = jth_polar_zernike(n, m)
        return aperture * _jth_rad_zern(rho) * _jth_pol_zern(theta)

    return _jth_zernike 

coords: float = dl.utils.get_pixel_coordinates(npix, width / npix)
third_zern: callable = jth_zernike(3)
third_zern_arr: float = third_zern(coords)
