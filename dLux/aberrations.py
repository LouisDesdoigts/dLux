from __future__ import annotations
from zodiax import Base
import jax.numpy as np
from jax import lax, Array
import jax.tree_util as jtu
import dLux.utils as dlu

__all__ = ["Zernike", "ZernikeBasis"]

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


class Zernike(Base):
    """
    A class to generate Zernike polynomials dynamically.

    The 'jth' zernike polynomial is defined [here](https://oeis.org/A176988).
    The basic translation between the noll index and the pair of numbers is
    shown below:

    1 -> (0, 0)

    2, 3 -> (1, -1), (1, 1)

    4, 5, 6 -> (2, -2), (2, 0), (2, 2)

    7, 8, 9, 10 -> (3, -3), (3, -1), (3, 1), (3, 3)


    Attributes
    ----------
    j : int
        The Zernike (noll) index.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    name : str
        The name of the Zernike polynomial.
    _k : Array
        The array of powers using the radial calculation. This is a
        pre-calculated parameter and should not be changed.
    _c : Array
        The array of normalisation coefficients used in the radial calculation.
        This is a pre-calculated parameter and should not be changed.
    """

    j: int
    n: int
    m: int
    name: str
    _k: Array
    _c: Array

    def __init__(self: Zernike, j: int):
        """
        Construct for the Zernike class.

        Parameters
        ----------
        j : int
            The Zernike (noll) index.
        """
        if int(j) < 1:
            raise ValueError("The Zernike index must be greater than 0.")
        self.j = int(j)
        self.n, self.m = self._noll_indices(self.j)
        self.name = (
            zernike_names[int(self.j)]
            if self.j >= 1 and self.j <= 36
            else f"Zernike {int(self.j)}"
        )

        # Calculate values
        self._k = np.arange(((self.n - self.m) // 2) + 1, dtype=float)
        sign = lax.pow(-1.0, self._k)
        _fact_1 = dlu.factorial(np.abs(self.n - self._k))
        _fact_2 = dlu.factorial(self._k)
        _fact_3 = dlu.factorial(((self.n + self.m) // 2) - self._k)
        _fact_4 = dlu.factorial(((self.n - self.m) // 2) - self._k)
        self._c = sign * _fact_1 / _fact_2 / _fact_3 / _fact_4

    def _noll_indices(self: Zernike, j: int) -> tuple[int]:
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

    def _calculate_radial(self: Zernike, rho: Array) -> Array:
        """
        Calculates the radial component of the Zernike polynomial.

        Parameters
        ----------
        rho : Array
            The radial coordinate of the Zernike polynomial.

        Returns
        -------
        radial : Array
            The radial component of the Zernike polynomial.
        """
        rads = lax.pow(
            rho[:, :, None], (np.abs(self.n) - 2 * self._k)[None, None, :]
        )
        return (self._c * rads).sum(axis=2)

    def _calculate_azimuthal(self: Zernike, theta: Array) -> Array:
        """
        Calculates the azimuthal component of the Zernike polynomial.

        Parameters
        ----------
        theta : Array
            The azimuthal coordinate of the Zernike polynomial.

        Returns
        -------
        azimuthal : Array
            The azimuthal component of the Zernike polynomial.
        """
        norm_coeff = np.sqrt(self.n + 1)
        if self.m != 0:
            norm_coeff *= 1 + (np.sqrt(2) - 1)

        if self.m >= 0:
            return norm_coeff * np.cos(np.abs(self.m) * theta)
        else:
            return norm_coeff * np.sin(np.abs(self.m) * theta)

    def calculate_zernike(self: Zernike, coordinates: Array) -> Array:
        """
        Calculates the Zernike polynomial.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike polynomial upon.

        Returns
        -------
        zernike : Array
            The Zernike polynomial.
        """
        polar_coordinates = dlu.cart_to_polar(coordinates)
        rho = polar_coordinates[0]
        theta = polar_coordinates[1]
        aperture = rho <= 1.0
        return (
            aperture
            * self._calculate_radial(rho)
            * self._calculate_azimuthal(theta)
        )

    def calculate_polike(
        self: Zernike, coordinates: Array, nsides: int
    ) -> Array:
        """
        Calculates the Zernike polynomial on an n-sided aperture.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike polynomial upon.
        nsides : int
            The number of sides of the aperture.

        Returns
        -------
        polike : Array
            The Zernike polynomial on an n-sided aperture.
        """
        if nsides < 3:
            raise ValueError(f"nsides must be >= 3, not {nsides}.")
        theta = dlu.cart_to_polar(coordinates)[1]
        alpha = np.pi / nsides
        phi = theta + alpha
        wedge = np.floor((phi + alpha) / (2.0 * alpha))
        u_alpha = phi - wedge * (2 * alpha)
        r_alpha = np.cos(alpha) / np.cos(u_alpha)
        return 1 / r_alpha * self.calculate_zernike(coordinates / r_alpha)

    def calculate(self: Zernike, coordinates: Array, nsides: int = 0) -> Array:
        """
        Calculates the Zernike polynomial.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike polynomial upon.
        nsides : int
            The number of sides of the aperture. If 0, the Zernike polynomial
            is calculated on a circular aperture.

        Returns
        -------
        zernike : Array
            The Zernike polynomial.
        """
        if nsides == 0:
            return self.calculate_zernike(coordinates)
        else:
            return self.calculate_polike(coordinates, nsides)


class ZernikeBasis(Base):
    """
    A class to calculate a set of Zernike polynomials on a dynamic set of
    coordinates.

    The 'jth' zernike polynomial is defined [here](https://oeis.org/A176988).
    The basic translation between the noll index and the pair of numbers is
    shown below:

    1 -> (0, 0)

    2, 3 -> (1, -1), (1, 1)

    4, 5, 6 -> (2, -2), (2, 0), (2, 2)

    7, 8, 9, 10 -> (3, -3), (3, -1), (3, 1), (3, 3)

    Attributes
    ----------
    noll_indices : list[Zernike]
        The list of Zernike polynomial classes to calculate.
    """

    noll_indices: list[Zernike]

    def __init__(self: ZernikeBasis, js: list[int]):
        """
        Constructor for the DynamicZernike class.

        Parameters
        ----------
        js : list[int]
            The list of Zernike (noll) indices to calculate.
        """
        self.noll_indices = [Zernike(j) for j in js]

    def calculate_basis(
        self: ZernikeBasis, coordinates: Array, nsides: int = 0
    ) -> Array:
        """
        Calculates the full Zernike polynomial basis.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike basis upon.
        nsides : int
            The number of sides of the aperture. If 0, the Zernike basis is
            calculated on a circular aperture.

        Returns
        -------
        zernike_basis : Array
            The Zernike polynomial basis.
        """
        leaf_fn = lambda leaf: isinstance(leaf, Zernike)
        calculate_fn = lambda z: z.calculate(coordinates, nsides)
        return np.array(
            jtu.tree_map(calculate_fn, self.noll_indices, is_leaf=leaf_fn)
        )
