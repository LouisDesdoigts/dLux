import jax 
import abc 
import jax.numpy as np
import equinox as eqx
import dLux as dl
import typing 
from typing import List

__all__ = ["AberratedCircularAperture", "AberratedHexagonalAperture"]

Array = typing.TypeVar("Array")
Layer = typing.TypeVar("Layer")
Aperture = typing.TypeVar("Aperture")
CircularAperture = typing.TypeVar("CircularAperture")
HexagonalAperture = typing.TypeVar("HexagonalAperture")


zernikes: list = [
    lambda rho, theta: np.ones(rho.shape, dtype=float),
    lambda rho, theta: 2. * rho * np.sin(theta),
    lambda rho, theta: 2. * rho * np.cos(theta),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.sin(2. * theta),
    lambda rho, theta: np.sqrt(3.) * (2. * rho ** 2 - 1.),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.cos(2. * theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.sin(3. * theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.cos(3. * theta)
]


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
    """
    Generates a function representing the polar component 
    of the jth Zernike polynomial.

    Parameters:
    -----------
    n: int 
        The first index number of the Zernike polynomial.
    m: int 
        The second index number of the Zernike polynomials.

    Returns:
    --------
    polar: Array
        The polar component of the jth Zernike polynomials.
    """
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


# So the current problem is that I need to find some way of passing 
# rmax into the hexike dynamically (do I). Haha just worked it out,
# I normalise the corrdinates first hence I don't need rmax. 
def jth_hexike(j: int) -> callable:
    """
    The jth Hexike as a function. 

    Parameters:
    -----------
    j: int
        The noll index of the requested zernike. 

    Returns:
    --------
    hexike: callable
        A function representing the jth hexike that is evaluated 
        on a cartesian coordinate grid. 
    """
    _jth_zernike = jth_zernike(j)

    def _jth_hexike(rho: Array, phi: Array) -> Array:
        wedge = np.floor((phi + np.pi / 3.) / (2. * np.pi / 3.))
        u_alpha = phi - wedge * (2. * np.pi / 3.)
        r_alpha = np.cos(np.pi / 3.) / np.cos(u_alpha)
        return 1 / r_alpha * _jth_zernike(rho / r_alpha, phi)

    return _jth_hexike


class AberratedAperture(eqx.Module, abc.ABC):
    """
    An abstract base class representing an `Aperture` defined
    with a basis. The basis is a set of polynomials that are 
    orthonormal over the surface of the aperture (usually). 
    These can be used to represent any aberation on the surface
    of the aperture. In general, the basis should only be defined 
    on apertures that have a surface such as a mirror or phase 
    plate ect. It isn't really possible to have aberrations on 
    an opening. This rule may be broken to learn the atmosphere 
    above a telescope but whether or not this is a good idea 
    remains to be seen.

    Parameters:
    -----------
    basis_funcs: list[callable]
        A list of functions that represent the basis. The exact
        polynomials that are represented will depend on the shape
        of the aperture. 
    aperture: Aperture
        The aperture on which the basis is defined. Must be a 
        subclass of the `Aperture` class.
    coeffs: list[floats]
        The coefficients of the basis terms. By learning the 
        coefficients only the amount of time that is required 
        for the learning process is significantly reduced.
    """
    aperture: Aperture
    coeffs: Array


    @abc.abstractmethod
    def __init__(self   : Layer, 
            noll_inds   : List[int],
            aperture    : Aperture, 
            coeffs      : Array) -> Layer:
        """
        Parameters:
        -----------
        noll_inds: List[int]
            The noll indices are a scheme for indexing the Zernike
            polynomials. Normally these polynomials have two 
            indices but the noll indices prevent an order to 
            these pairs. All basis can be indexed using the noll
            indices based on `n` and `m`. 
        aperture: Aperture
            The aperture that the basis is defined on. The shape 
            of this aperture defines what the polynomials are. 
        coeffs: Array
            The coefficients of the basis vectors. 
        """


    @abc.abstractmethod
    def _basis(self: Layer, coords: Array) -> Array:
        """
        Generate the basis vectors over a set of coordinates.  

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. 
        """


    def _opd(self: Layer, coords: Array) -> Array:
        """
        Calculate the optical path difference that is caused 
        by the basis and the aberations that it represents. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        opd: Array
            The optical path difference associated with much of 
            the path. 
        """
        basis: Array = self._basis(coords)
        opd: Array = np.dot(basis.T, self.coeffs)
        return opd


    def __call__(self, params_dict: dict) -> dict:
        """
        Apply the aperture and the abberations to the wavefront.  

        Parameters:
        -----------
        params: dict
            A dictionary containing the key "Wavefront".

        Returns:
        --------
        params: dict 
            A dictionary containing the key "wavefront".
        """
        wavefront: object = params_dict["Wavefront"]
        coords: Array = wavefront.pixel_positions()
        opd: Array = self._opd(coords)
        aperture: Array = self.aperture._aperture(coords)
        params_dict["Wavefront"] = wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)
        return params_dict


class AberratedCircularAperture(AberratedAperture):
    """
    Parameters:
    -----------
    zernikes: Array
        An array of `jit` compiled zernike basis functions 
        that operate on a set of coordinates. In particular 
        these coordinates correspond to a normalised set 
        of coordinates that are centered at the the centre 
        of the circular aperture with 1. occuring along the 
        radius. 
    coeffs: Array
        The coefficients of the Zernike terms. 
    aperture: Layer
        Must be an instance of `CircularAperture`. This 
        is applied alongside the basis. 
    """
    zernikes: Array
    coeffs: Array
    aperture: CircularAperture


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : CircularAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: CircularAperture
            A `CircularAperture` within which the aberrations are 
            being studied. 
        """
        self.zernikes = [zernikes[ind] if ind < 10 else jth_zernike(ind)
            for ind in noll_inds]
        self.coeffs = np.asarray(coeffs).astype(float)
        self.aperture = aperture

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.CircularAperture)


    def _basis(self: Layer, coords: Array) -> Array:
        """
        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. Normally the basis is 
            cropped to be just on the aperture however, this 
            step is not necessary except for in visualisation. 
            It has been removed to save some time in the 
            calculations. 
        """
        pol_coords: list = dl.utils.cartesian_to_polar(coords)
        rho: list = pol_coords[0]
        theta: list = pol_coords[1]
        basis: list = np.stack([z(rho, theta) for z in self.zernikes])
        return basis


class AberratedHexagonalAperture(AberratedAperture):
    """
    Parameters:
    -----------
    Hexikes: Array
        An array of `jit` compiled hexike basis functions 
        that operate on a set of coordinates. In particular 
        these coordinates correspond to a normalised set 
        of coordinates that are centered at the the centre 
        of the circular aperture with 1. occuring along the 
        radius. 
    coeffs: Array
        The coefficients of the Hexike terms. 
    aperture: Layer
        Must be an instance of `HexagonalAperture`. This 
        is applied alongside the basis. 
    """
    hexikes: Array
    coeffs: Array
    aperture: HexagonalAperture


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : HexagonalAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: HexagonalAperture
            A `HexagonalAperture` within which the aberrations are 
            being studied. 
        """

        self.hexikes = [jth_hexike(j) for j in noll_inds]
        self.coeffs = np.asarray(coeffs).astype(float)
        self.aperture = aperture

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.HexagonalAperture)


    def _basis(self: Layer, coords: Array) -> Array:
        """
        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. Normally the basis is 
            cropped to be just on the aperture however, this 
            step is not necessary except for in visualisation. 
            It has been removed to save some time in the 
            calculations. 
        """
        pol_coords: list = dl.utils.cartesian_to_polar(coords)
        rho: list = pol_coords[0]
        theta: list = pol_coords[1]
        basis: list = np.stack([h(rho, theta) for h in self.hexikes])
        return basis
