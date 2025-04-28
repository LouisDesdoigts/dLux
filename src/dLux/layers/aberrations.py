from __future__ import annotations
from zodiax import Base
import jax.numpy as np
from jax import Array
import jax.tree as jtu
import dLux.utils as dlu

__all__ = ["Zernike", "ZernikeBasis"]


# TODO: Should all the leaves of this class be static?
class Zernike(Base):
    """
    A Zernike polynomial that can be generated dynamically in a way that is both jit and
    grad safe. If you want a _static_ zernike (most use cases), use the zernike
    functions found in `utils.zernikes` and load the basis into a `BasisOptic` class.

    The 'jth' zernike polynomial is defined [here](https://oeis.org/A176988). The basic
    translation between the noll index and the pair of numbers is shown below:

    1 -> (0, 0) : Piston

    2, 3 -> (1, -1), (1, 1) : Tip, Tilt

    4, 5, 6 -> (2, -2), (2, 0), (2, 2) : Defocus, Astigmatism

    7, 8, 9, 10 -> (3, -3), (3, -1), (3, 1), (3, 3) : Coma, Trefoil

    ??? abstract "UML"
        ![UML](../../assets/uml/Zernike.png)

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
        The array of powers using the radial calculation. This is a pre-calculated
        parameter and should not be changed.
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

        Note: standard zernike polynomials are only defined up to a radial value of 1,
        so generating one that spans the entire aperture needs a diameter of 2.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike upon.
        nsides : int
            The number of sides of the aperture. If 0, the Zernike is calculated
            on a circular aperture.

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


# TODO: Rename basis to basis_fns??
class ZernikeBasis(Base):
    """
    A Basis of Zernike polynomials that can be generated dynamically in a way that is
    both jit and grad safe. If you want a _static_ zernike (most use cases), use the
    zernike functions found in `utils.zernikes` and load the basis into a `BasisOptic`
    class.

    The 'jth' zernike polynomial is defined [here](https://oeis.org/A176988). The basic
    translation between the noll index and the pair of numbers is shown below:

    1 -> (0, 0) : Piston

    2, 3 -> (1, -1), (1, 1) : Tip, Tilt

    4, 5, 6 -> (2, -2), (2, 0), (2, 2) : Defocus, Astigmatism

    7, 8, 9, 10 -> (3, -3), (3, -1), (3, 1), (3, 3) : Coma, Trefoil

    ??? abstract "UML"
        ![UML](../../assets/uml/ZernikeBasis.png)

    Attributes
    ----------
    basis : list[Zernike]
        The list of `Zernike` polynomial classes to calculate.
    """

    basis: list[Zernike]

    def __init__(self: ZernikeBasis, js: list[int]):
        """
        Constructor for the DynamicZernike class.

        Parameters
        ----------
        js : list[int]
            The list of Zernike (noll) indices to calculate.
        """
        self.basis = [Zernike(j) for j in js]

    def calculate_basis(
        self: ZernikeBasis, coordinates: Array, nsides: int = 0
    ) -> Array:
        """
        Calculates the full Zernike polynomial basis.

        Note: standard zernike polynomials are only defined up to a radial value of 1,
        so generating a basis that spans the entire aperture needs a diameter of 2.

        Parameters
        ----------
        coordinates : Array
            The Cartesian coordinates to calculate the Zernike basis upon.
        nsides : int
            The number of sides of the aperture. If 0, the Zernike basis is calculated
            on a circular aperture.

        Returns
        -------
        basis : Array
            The Zernike polynomial basis.
        """
        leaf_fn = lambda leaf: isinstance(leaf, Zernike)
        calculate_fn = lambda z: z.calculate(coordinates, nsides)
        return np.array(jtu.map(calculate_fn, self.basis, is_leaf=leaf_fn))
