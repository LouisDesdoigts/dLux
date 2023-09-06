from __future__ import annotations
from zodiax import Base
import jax.numpy as np
from jax import Array
import jax.tree_util as jtu
import dLux.utils as dlu

__all__ = ["Zernike", "ZernikeBasis"]


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
    _c : Array
        The array of normalisation coefficients used in the radial calculation.
        This is a pre-calculated parameter and should not be changed.
    _k : Array
        The array of powers using the radial calculation. This is a
        pre-calculated parameter and should not be changed.
    """

    j: int
    n: int
    m: int
    name: str
    _c: Array
    _k: Array

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
        self.name = dlu.zernike_name(j)
        self.n, self.m = dlu.noll_indices(j)
        self._c, self._k = dlu.zernike_factors(j)

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
            return dlu.zernike_fast(
                self.n, self.m, self._c, self._k, coordinates
            )
        else:
            return dlu.polike_fast(
                nsides, self.n, self.m, self._c, self._k, coordinates
            )


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
