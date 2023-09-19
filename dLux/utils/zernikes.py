from __future__ import annotations
import jax.numpy as np
from jax import lax, Array
import dLux.utils as dlu

__all__ = [
    "zernike_name",
    "noll_indices",
    "zernike_factors",
    "eval_radial",
    "eval_azimuthal",
    "zernike",
    "zernike_fast",
    "zernike_basis",
    "polike",
    "polike_fast",
    "polike_basis",
]

zernike_names = {
    # 0th Radial
    1: "Piston",
    # 1st Radial
    2: "Tilt X",
    3: "Tilt Y",
    # Second Radial
    4: "Defocus",
    5: "Astig X",
    6: "Astig Y",
    # Third Radial
    7: "Coma X",
    8: "Coma Y",
    9: "Trefoil X",
    10: "Trefoil Y",
    # Fourth Radial
    11: "Spherical",
    12: "2nd Astig X",
    13: "2nd Astig Y",
    14: "Quadrafoil X",
    15: "Quadrafoil Y",
    # Fifth Radial
    16: "2nd Coma X",
    17: "2nd Coma Y",
    18: "2nd Trefoil X",
    19: "2nd Trefoil Y",
    20: "Pentafoil X",
    21: "Pentafoil Y",
    # Sixth Radial
    22: "2nd Spherical",
    23: "3rd Coma X",
    24: "3rd Coma Y",
    25: "3rd Astig X",
    26: "3rd Astig Y",
    27: "Hexafoil X",
    28: "Hexafoil Y",
    # Seventh Radial
    29: "4th Coma X",
    30: "4th Coma Y",
    31: "4th Astig X",
    32: "4th Astig Y",
    33: "3rd Trefoil X",
    34: "3rd Trefoil Y",
    35: "Heptafoil X",
    36: "Heptafoil Y",
}


def zernike_name(j: int) -> str:
    """
    Gets the name of the Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    name : str
        The name of the Zernike polynomial.
    """
    return zernike_names[int(j)] if j >= 1 and j <= 36 else f"Zernike {int(j)}"


def noll_indices(j: int) -> tuple[int]:
    """
    Calculate the radial and azimuthal orders of the Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    n, m : tuple[int]
        The radial and azimuthal orders of the Zernike polynomial.
    """
    n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
    smallest_j_in_row = n * (n + 1) / 2 + 1
    number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
    sign_of_shift = -(j & 1) + ~(j & 1) + 2
    base_case = n & 1
    m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
    return int(n), int(m)


def zernike_factors(j: int) -> tuple[Array]:
    """
    Calculates the normalisation coefficients and powers of the Zernike
    polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    c, k : tuple[Array]
        The normalisation coefficients and powers of the Zernike polynomial.
    """
    if j < 1:
        raise ValueError("The Zernike index must be greater than 0.")
    n, m = noll_indices(j)

    # Calculate k
    k = np.arange(((n - m) // 2) + 1, dtype=float)

    # Calculate c
    sign = lax.pow(-1.0, k)
    _fact_1 = dlu.factorial(np.abs(n - k))
    _fact_2 = dlu.factorial(k)
    _fact_3 = dlu.factorial(((n + m) // 2) - k)
    _fact_4 = dlu.factorial(((n - m) // 2) - k)
    c = sign * _fact_1 / _fact_2 / _fact_3 / _fact_4
    return c, k


def eval_radial(rho: Array, n: int, c: Array, k: Array) -> Array:
    """
    Calculates the radial component of the Zernike polynomial.

    Parameters
    ----------
    rho : Array
        The radial coordinate of the Zernike polynomial.
    n : int
        The radial order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.

    Returns
    -------
    radial : Array
        The radial component of the Zernike polynomial.
    """
    rads = lax.pow(rho[:, :, None], (np.abs(n) - 2 * k)[None, None, :])
    return (c * rads).sum(axis=2)


def eval_azimuthal(theta: Array, n: int, m: int) -> Array:
    """
    Calculates the azimuthal component of the Zernike polynomial.

    Parameters
    ----------
    theta : Array
        The azimuthal coordinate of the Zernike polynomial.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.

    Returns
    -------
    azimuthal : Array
        The azimuthal component of the Zernike polynomial.
    """
    # Get normalisation coefficient
    norm_coeff = np.sqrt(n + 1)
    if m != 0:
        norm_coeff *= 1 + (np.sqrt(2) - 1)

    # Get the right trig function and eval
    trig_fn = np.cos if m >= 0 else np.sin
    return norm_coeff * trig_fn(np.abs(m) * theta)


def scale_coords(coords, rmax):
    """Scales coordinates to the unit circle"""
    return coords / rmax


def zernike(j: int, coordinates: Array, diameter=2) -> Array:
    """
    Calculates the Zernike polynomial.

    Note that this function is not-jittable as is has dynamic array shapes. To
    use this function in a jittable way, use the zernike_fast function, with
    the pre-calculated c and k parameters.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike : Array
        The Zernike polynomial.
    """
    coordinates = scale_coords(coordinates, diameter / 2)
    polar_coordinates = dlu.cart2polar(coordinates)
    rho = polar_coordinates[0]
    theta = polar_coordinates[1]
    aperture = rho <= 1.0
    n, m = noll_indices(j)
    c, k = zernike_factors(j)
    return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


def zernike_fast(
    n: int, m: int, c: Array, k: Array, coordinates: Array
) -> Array:
    """
    Calculates the Zernike polynomial.

    Note this function is jittable as it has no dynamic array shapes.

    Parameters
    ----------
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike : Array
        The Zernike polynomial.
    """
    polar_coordinates = dlu.cart2polar(coordinates)
    rho = polar_coordinates[0]
    theta = polar_coordinates[1]
    aperture = rho <= 1.0
    return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


def zernike_basis(js, coordinates, diameter=2):
    return np.array([zernike(j, coordinates, diameter) for j in js])


def polike(nsides: int, j: int, coordinates: Array, diameter=2) -> Array:
    """
    Calculates the Zernike polynomial on an n-sided aperture.

    Note this function is not-jittable as it has dynamic array shapes. To
    use this function in a jittable way, use the polike_fast function, with
    the pre-calculated c and k parameters.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    j : int
        The Zernike (noll) index.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    polike : Array
        The Zernike polynomial on an n-sided aperture.
    """
    if nsides < 3:
        raise ValueError(f"nsides must be >= 3, not {nsides}.")
    coordinates = scale_coords(coordinates, diameter / 2)
    theta = dlu.cart2polar(coordinates)[1]
    alpha = np.pi / nsides
    phi = theta + alpha
    wedge = np.floor((phi + alpha) / (2.0 * alpha))
    u_alpha = phi - wedge * (2 * alpha)
    r_alpha = np.cos(alpha) / np.cos(u_alpha)
    return 1 / r_alpha * zernike(j, coordinates / r_alpha)


def polike_fast(
    nsides: int, n: int, m: int, c: Array, k: Array, coordinates: Array
) -> Array:
    """
    Calculates the Zernike polynomial on an n-sided aperture.

    Note this function is jittable as it has no dynamic array shapes.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    polike : Array
        The Zernike polynomial on an n-sided aperture.
    """
    if nsides < 3:
        raise ValueError(f"nsides must be >= 3, not {nsides}.")
    # theta =
    alpha = np.pi / nsides
    phi = dlu.cart2polar(coordinates)[1] + alpha
    wedge = np.floor((phi + alpha) / (2.0 * alpha))
    u_alpha = phi - wedge * (2 * alpha)
    r_alpha = np.cos(alpha) / np.cos(u_alpha)
    return 1 / r_alpha * zernike_fast(n, m, c, k, coordinates / r_alpha)


def polike_basis(nsides, js, coordinates, diameter=2):
    return np.array([polike(nsides, j, coordinates, diameter) for j in js])
