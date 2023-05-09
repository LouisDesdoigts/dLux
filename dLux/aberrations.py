from __future__ import annotations
from zodiax import Base
import jax.numpy as np
from jax import lax, Array
import jax.tree_util as jtu
import dLux
from dLux.utils.math import factorial, triangular_number
from dLux.utils.coordinates import cartesian_to_polar, get_pixel_positions


__all__ = ['Zernike', 'ZernikeBasis', 'AberrationFactory']


zernike_names = {
    # 0th Radial
    1: 'Piston',

    # 1st Radial
    2: 'Tilt X',
    3: 'Tilt Y',
    
    # Second Radial
    4: 'Defocus',
    5: 'Astig X',
    6: 'Astig Y',
    
    # Third Radial
    7: 'Coma X',
    8: 'Coma Y',
    9: 'Trefoil X',
    10: 'Trefoil Y',

    # Fourth Radial
    11: 'Spherical',
    12: '2nd Astig X',
    13: '2nd Astig Y',
    14: 'Quadrafoil X',
    15: 'Quadrafoil Y',
    
    # Fifth Radial
    16: '2nd Coma X',
    17: '2nd Coma Y',
    18: '2nd Trefoil X',
    19: '2nd Trefoil Y',
    20: 'Pentafoil X',
    21: 'Pentafoil Y',
    
    # Sixth Radial
    22: '2nd Spherical',
    23: '3rd Coma X',
    24: '3rd Coma Y',
    25: '3rd Astig X',
    26: '3rd Astig Y',
    27: 'Hexafoil X',
    28: 'Hexafoil Y',
    
    # Seventh Radial
    29: '4th Coma X',
    30: '4th Coma Y',
    31: '4th Astig X',
    32: '4th Astig Y',
    33: '3rd Trefoil X',
    34: '3rd Trefoil Y',
    35: 'Heptafoil X',
    36: 'Heptafoil Y',
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
        The array of normalisaton coefficients used in the radial calculation.
        This is a pre-calculated parameter and should not be changed.
    """
    j    : int
    n    : int
    m    : int
    name : str
    _k    : Array
    _c    : Array

    def __init__(self : Zernike, j : int):
        """
        Construct for the Zernike class.

        Parameters
        ----------
        j : int
            The Zernike (noll) index.
        """
        if int(j) < 1:
            raise ValueError('The Zernike index must be greater than 0.')
        self.j = int(j)
        self.n, self.m = self._noll_indexes(self.j)
        self.name = zernike_names[int(self.j)] if self.j >= 1 and self.j <= 36 \
                    else f'Zernike {int(self.j)}'

        # Calcualte values
        self._k = np.arange(((self.n - self.m) // 2) + 1, dtype=float)
        sign = lax.pow(-1., self._k)
        _fact_1 = factorial(np.abs(self.n - self._k))
        _fact_2 = factorial(self._k)
        _fact_3 = factorial(((self.n + self.m) // 2) - self._k)
        _fact_4 = factorial(((self.n - self.m) // 2) - self._k)
        self._c = sign * _fact_1 / _fact_2 / _fact_3 / _fact_4 


    def _noll_indexes(self : Zernike, j : int) -> tuple[int]:
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
        base_case = (n & 1)
        m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
        return int(n), int(m)
    

    def _calculate_radial(self : Zernike, rho : Array) -> Array:
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
        rads = lax.pow(rho[:, :, None], 
            (np.abs(self.n) - 2 * self._k)[None, None, :])
        return (self._c * rads).sum(axis = 2)


    def _calculate_azimuthal(self : Zernike, theta : Array) -> Array:
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
        

    def calculate_zernike(self : Zernike, coordinates : Array) -> Array:
        """
        Calculates the Zernike polynomial.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to calcualte the Zernike polynomial upon.
        
        Returns
        -------
        zernike : Array
            The Zernike polynomial.
        """
        polar_coordinates = cartesian_to_polar(coordinates)
        rho = polar_coordinates[0]
        theta = polar_coordinates[1]
        aperture = rho <= 1.
        return aperture * self._calculate_radial(rho) * \
            self._calculate_azimuthal(theta)
    

    def calculate_polike(self        : Zernike, 
                         coordinates : Array, 
                         nsides      : int) -> Array:
        """
        Calculates the Zernike polynomial on an nsided aperture.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to calcualte the Zernike polynomial upon.
        nsides : int
            The number of sides of the aperture.

        Returns
        -------
        polike : Array
            The Zernike polynomial on an nsided aperture.
        """
        if nsides < 3:
            raise ValueError(f'nsides must be >= 3, not {nsides}.')
        theta = cartesian_to_polar(coordinates)[1]
        alpha = np.pi / nsides
        phi = theta + alpha  
        wedge = np.floor((phi + alpha) / (2. * alpha))
        u_alpha = phi - wedge * (2 * alpha)
        r_alpha = np.cos(alpha) / np.cos(u_alpha)
        return 1 / r_alpha * self.calculate_zernike(coordinates / r_alpha)


    def calculate(self        : Zernike, 
                  coordinates : Array, 
                  nsides      : int = 0) -> Array:
        """
        Calculates the Zernike polynomial.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to calcualte the Zernike polynomial upon.
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
    coordiantes.

    The 'jth' zernike polynomial is defined [here](https://oeis.org/A176988).
    The basic translation between the noll index and the pair of numbers is
    shown below:
    
    1 -> (0, 0)
    
    2, 3 -> (1, -1), (1, 1)
    
    4, 5, 6 -> (2, -2), (2, 0), (2, 2)
    
    7, 8, 9, 10 -> (3, -3), (3, -1), (3, 1), (3, 3)

    Attributes
    ----------
    noll_indexes : list[Zernike]
        The list of Zernike polynomial classes to calculate.
    """
    noll_indexes : list[Zernike]


    def __init__(self : ZernikeBasis, js : list[int]):
        """
        Constructor for the DynamicZernike class.

        Parameters
        ----------
        js : list[int]
            The list of Zernike (noll) indices to calculate.
        """
        self.noll_indexes = [Zernike(j) for j in js]
    

    def calculate_basis(self        : ZernikeBasis, 
                        coordinates : Array, 
                        nsides      : int = 0) -> Array:
        """
        Calculates the full Zernike polynomial basis.

        Note: The zernike polynomial is defined on the coordinates up to a
        radial value of 1.

        Parameters
        ----------
        coordinates : Arraya
            The cartesian coordinates to calcualte the Zernike basis upon.
        
        Returns
        -------
        zernike_basis : Array
            The Zernike polynomial basis.
        """
        leaf_fn = lambda leaf: isinstance(leaf, Zernike)
        calculate_fn = lambda z: z.calculate(coordinates, nsides)
        return np.array(jtu.tree_map(calculate_fn, self.noll_indexes, 
                                     is_leaf=leaf_fn))


###############
### Factory ###
###############
class AberrationFactory():
    """
    This class is not actually ever instatiated, but is rather a class used to 
    give a simple constructor interface that is used to construct the most
    commonly used aberrations. It is able to construct hard-edged circular or 
    regular poygonal aberrations. 

    Lets look at an example of how to construct a simple circular aberration 
    class. Let calcualte this for a 512x512 array with the aperture spanning
    the full array.

    ```python
    from dLux import AberrationFactory
    import jax.numpy as np
    import jax.random as jr
    
    # Construct Zernikes
    noll_indexes = np.arange(4, 11)
    coefficients = jr.normal(jr.PRNGKey(0), (noll_indexes.shape[0],))

    # Construct aberrations
    aberrations = AberrationFactory(512, noll_indexes, coefficients)
    ```
    
    The resulting aperture class has two parameters, `.basis` and 
    `.coefficients`.

    We can also easily change this to a hexagonal aperture:

    ```python
    # Construct aberrations
    hexagonal_aberrations = AberrationFactory(512, nsides=6, noll_indexes, 
        coefficients)
    ```

    ```python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(aberrations.get_opd())
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(hexagonal_aberrations.get_opd())
    plt.colorbar()
    plt.show()
    ```
    """
    def __new__(cls              : AberrationFactory, 
                npixels          : int, 
                radial_order     : Array,
                coefficients     : Array = None, 
                aperutre_ratio   : float = 1.0,
                nsides           : int   = 0,
                rotation         : float = 0.,
                noll_indexes     : Array = None,
                name             : str   = 'Aberrations'):
        """
        Constructs a basic single static aberration class.

        TODO: Add link to the zenike noll indicies

        Parameters
        ----------
        npixels : int
            Number of pixels used to represent the aperture.
        radial_orders : Array
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indexes argument should be used instead.
        coefficients : Array = None
            The zernike cofficients to be applied to the aberrations. Defaults 
            to an array of zeros.
        aperutre_ratio : float = 1.
            The ratio of the aperture size to the array size. A value of 1. 
            results in an aperture that fully spans the array, a value of 0.5 
            retuls in an aperure that is half the size of the array, which is 
            equivilent to a padding factor of 2.
        nsides : int = 0
            Number of sides of the aperture. A zero input results in a circular
            aperture. All other other values of three and above are supported.
        rotation : float, radians = 0
            The global rotation of the aperture in radians.
        noll_indexes : Array = None
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus. 
        name : str = 'Aberrations'
            The name of the aperture used to index the layers dictionary. If 
            not supplied, the aperture will be named based on the number of
            sides. However this is only supported up to 8 sides, and a name
            must be supplied for apertures with more than 8 sides.
        
        Returns
        -------
        aberations : ApplyBasisOPD
            Returns an appropriately constructed ApplyBasisOPD.
        """
        # Check vaid inputs
        if nsides < 3 and nsides != 0:
            raise ValueError("nsides must be either 0 or >=3")

        # Construct coordinates
        coords = get_pixel_positions((npixels, npixels), (1/npixels, 1/npixels))

        # Circular Primary
        if nsides == 0:
            ap = dLux.apertures.CircularAperture
            dyn_aperture = ap(aperutre_ratio/2, softening=0)
            coordinates = dyn_aperture._normalised_coordinates(coords)

        # Polygonal Primary
        else: 
            ap = dLux.apertures.RegularPolygonalAperture
            dyn_aperture = ap(nsides, aperutre_ratio/2, softening=0, 
                          rotation=rotation)
            coordinates = dyn_aperture._normalised_coordinates(coords)


        # Noll_indexes overwrite radial_orders
        if noll_indexes is not None:
            radial_orders = np.array(radial_orders)

            if (radial_orders < 0).any():
                raise ValueError('Radial orders must be >= 0')

            noll_indexes = []
            for order in radial_orders:
                start = triangular_number(order-1)
                stop = triangular_number(order)
                noll_indexes.append(np.arange(start, stop))
            noll_indexes = np.concatenate(noll_indexes)        

        # Construct Aberrations
        basis = ZernikeBasis(noll_indexes).calculate_basis(coordinates)
        return dLux.optics.ApplyBasisOPD(basis, coefficients, name=name)


    def __init__(self           : AberrationFactory, 
                 npixels        : int, 
                 radial_orders  : int,
                 noll_indexes   : Array, 
                 coefficients   : Array = None, 
                 aperutre_ratio : float = 1.0,
                 nsides         : int   = 0,
                 rotation       : float = 0., 
                 name           : str   = 'Aberrations'):
        """
        Constructs a basic single static aberration class.

        TODO: Add link to the zenike noll indicies

        Parameters
        ----------
        npixels : int
            Number of pixels used to represent the aperture.
        radial_orders : Array
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indexes argument should be used instead.
        coefficients : Array = None
            The zernike cofficients to be applied to the aberrations. Defaults 
            to an array of zeros.
        aperutre_ratio : float = 1.
            The ratio of the aperture size to the array size. A value of 1. 
            results in an aperture that fully spans the array, a value of 0.5 
            retuls in an aperure that is half the size of the array, which is 
            equivilent to a padding factor of 2.
        nsides : int = 0
            Number of sides of the aperture. A zero input results in a circular
            aperture. All other other values of three and above are supported.
        rotation : float, radians = 0
            The global rotation of the aperture in radians.
        noll_indexes : Array = None
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus. 
        name : str = 'Aberrations'
            The name of the aperture used to index the layers dictionary. If 
            not supplied, the aperture will be named based on the number of
            sides. However this is only supported up to 8 sides, and a name
            must be supplied for apertures with more than 8 sides.
        
        Returns
        -------
        aperture : ApplyBasisOPD
            Returns an appropriately constructed ApplyBasisOPD.
        """