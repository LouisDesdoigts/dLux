"""
src/dev/layers.py
-----------------
Development script for the new layers structure.
"""

from constants import *
from matplotlib import pyplot
from typing import TypeVar
from dLux.utils import (get_radial_positions, get_pixel_vector, 
    get_pixel_positions)
from abc import ABC, abstractmethod 
import equinox as eqx
import jax.numpy as np
import jax 
import functools

config.update("jax_enable_x64", True)

Array = TypeVar("Array")
Layer = TypeVar("Layer")
Tensor = TypeVar("Tensor")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")


MAX_DIFF = 4


def cartesian_to_polar(coordinates : Tensor) -> Tensor:
    """
    Change the coordinate system from rectilinear to curvilinear.
    
    Parameters
    ----------
    coordinates : Tensor
        The rectilinear coordinates.

    Returns
    -------
    coordinates : Tensor
        The curvilinear coordinates.
    """
    rho = np.hypot(coordinates[0], coordinates[1])
    theta = np.arctan2(coordinates[1], coordinates[0])
    return np.array([rho, theta])


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


class Aperture(eqx.Module, ABC):
    """
    An abstract class that defines the structure of all the concrete
    apertures. An aperture is represented by an array, usually in the
    range of 0. to 1.. Values in between can be used to represent 
    soft edged apertures and intermediate surfaces. 

    Attributes
    ----------
    _npix : int
        The number of pixels along one edge of the array which 
        represents the aperture.
    """
    _npix : int
    

    def __init__(self : Layer, number_of_pixels : int) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels : int
            The number of pixels along one side of the array that 
            represents this aperture.
        """
        self._npix = number_of_pixels

    
    @abstractmethod
    def _aperture(self : Layer, number_of_pixels : int) -> Array:
        """
        Generate the aperture array as an array. 

        Returns
        -------
        aperture : Array[Float]
            The aperture. If these values are confined between 0. and 1.
            then the physical interpretation is the transmission 
            coefficient of that pixel. 
        """


    def get_npix(self : Layer) -> int:
        """
        Returns
        -------
        pixels : int
            The number of pixels that parametrise this aperture.
        """
        return self._npix


    def __call__(self : Layer, parameters : dict) -> dict:
        """
        Apply the aperture to an incoming wavefront.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of the model. 
            The dictionary must satisfy `parameters.get("Wavefront")
            != None`. 

        Returns
        -------
        parameters : dict
            The parameter, parameters, with the "Wavefront"; key
            value updated. 
        """
        wavefront = parameters["Wavefront"]
        wavefront = wavefront.mulitply_amplitude(
           self._aperture(self.get_npix()))
        parameters["Wavefront"] = wavefront
        return parameters


class AnnularAperture(Aperture):
    """
    A circular aperture, parametrised by the number of pixels in
    the array. By default this is a hard edged aperture but may be 
    in future modifed to provide soft edges. 

    Attributes
    ----------
    rmax : float
        The proportion of the pixel vector that is contained within
        the outer ring of the aperture.
    rmin : float
        The proportion of the pixel vector that is contained within
        the inner ring of the aperture. 
    """
    rmin : float
    rmax : float


    def __init__(self : Layer, npix : int, rmax : float = 1., 
            rmin : float = 0.) -> Layer:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        rmax : float = 1. 
            The proportion of the pixel vector contained within the 
            outer ring.
        rmin : float = 0.
            The proportion of the pixel vector contained within the
            inner ring.
        """
        super().__init__(npix)
        self.rmax = rmax
        self.rmin = rmin


    def _aperture(self : Layer) -> Array:
        """
        Generates an array representing a hard edged circular aperture.
        All the values are 0. except for the outer edge. The t
 
        Returns
        -------
        aperture : Array[Float]
            The aperture. If these values are confined between 0. and 1.
            then the physical interpretation is the transmission 
            coefficient of that pixel. 
        """
        centre = (self.get_npix() - 1.) / 2.
        coords = 2 / self.get_npix() * get_radial_positions(self.get_npix())
        return np.logical_and(coords <= rmax, coords > rmin).astype(float)


class HexagonalAperture(Aperture):
    """
    Generate a hexagonal aperture, parametrised by rmax. 
    
    Attributes
    ----------
    rmax : float
        The proportion of the pixel vector that is taken up by a 
        circle containing the hexagon.
    x : int
        The pixel coordinate in the x direction of the centre of the 
        hexagonal aperture.
    y : int
        The pixel coordinate in the y direction of the centre of the 
        hexagonal aperture. 
    """
    _rmax : float
    _x : int
    _y : int


    def __init__(self : Layer, 
            number_of_pixels_on_side : int, 
            proportional_maximum_radius : float,
            x_position_of_centre : int = 0, 
            y_position_of_centre : int = 0) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels_on_side : int
            The number of pixels along one side of the square array
        proportional_maximum_radius : float 
            The outer radius of the smallest circle that contains the 
            hexagon.
        x_position_of_centre : int = 0
            The x position of the centre of the hexagonal aperture.
        y_poxition_of_centre : int = 0
            The y position of the centre of the hexagonal aperture.             
        """
        super().__init__(number_of_pixels_on_side)
        self._rmax = proportional_maximum_radius
        self._x = x_position_of_centre
        self._y = y_position_of_centre


    def _get_centre(self : Layer) -> tuple:
        """
        The x and y coordinates of the centre of the hexagonal aperture.
        
        Returns
        -------
        x, y : tuple
            The x and y coordinates of the centre in pixels
        """
        return self._x, self._y


    def _get_rmax(self : Layer) -> float:
        """
        Returns
        -------
        rmax : float
            The proportional maximum radius of the smallest circle that
            can completely enclose the aperture.
        """
        return self._rmax


    def _aperture(self : Layer) -> Array:
        """
        Generates an array representing the hard edged hexagonal 
        aperture. 

        Returns
        -------
        aperture : Array
            The aperture represented as a binary float array of 0. and
            1. representing no transmission and transmission 
            respectively.
        """
        x_centre, y_centre = self._get_centre()
        number_of_pixels = self._get_npix()
        maximum_radius = self._get_rmax()

        x, y = _get_pixel_positions(number_of_pixels, -x_centre,
            -y_centre)

        x *= 2 / number_of_pixels
        y *= 2 / number_of_pixels

        rectangle = (np.abs(x) <= maximum_radius / 2.) \
            & (np.abs(y) <= (maximum_radius * np.sqrt(3) / 2.))

        left_triangle = (x <= - maximum_radius / 2.) \
            & (x >= - maximum_radius) \
            & (np.abs(y) <= (x + maximum_radius) * np.sqrt(3))

        right_triangle = (x >= maximum_radius / 2.) \
            & (x <= maximum_radius) \
            & (np.abs(y) <= (maximum_radius - x) * np.sqrt(3))

        hexagon = rectangle | left_triangle | right_triangle
        return np.asarray(hexagon).astype(float)


class BasisPhase(eqx.Module):
    """
    _Abstract_ class representing a basis fixed over an aperture 
    that is used to optimise and learn aberations in the aperture. 
    
    Attributes
    ----------
    _x : float
        The x coordinate of the centre of the aperture.
    _y : float
        The y coordinate of the centre of the aperture.
    _r : float
        The radius of the aperture. The radius belongs to the smallest
        circle that completely contains the aperture. For the math
        nerds the infimum of the set of circles whose union with the
        aperture is the aperture. 
    _npix : int
        The number of pixels along the edge of the square array 
        representing each term in the basis.
    """
    x : float
    y : float
    rmax : float
    npix : int
    nterms : int


    def __init__(self : Layer, x : float, y : float, r : float,
            npix : int, nterms : int) -> Layer:
        """
        Parameters
        ----------
        x : float
            The x coordinate of the centre of the aperture containing 
            the basis.
        y : float
            The y coordinate of the centre of the aperture containing 
            the basis.
        r : float
            The radius of the aperture containing the basis. For 
            abitrary shapes the radius is belongs to the smallest 
            circle that can completely contain the aperture. 
        npix : int
            The number of pixels along one side of the basis arrays.
            That is each term in the basis will be evaluated on a 
            `npix` by `npix` grid.
        nterms : int
            The number of basis terms to generate. This determines the
            length of the leading dimension of the output Tensor. 
        """
        self.x = np.asarray(x).astype(float)
        self.y = np.asarray(y).astype(float)
        self.rmax = np.asarray(r).astype(float)
        self.npix = int(npix)
        self.nterms = int(nterms)


    @abstractmethod
    def _aperture(self : Layer) -> Matrix:
        """
        Generate a binary or soft edged aperture mask in which to 
        construct the basis polynomials. 

        Returns
        -------
        aperture : Matrix
            The aperture encoded in an `npix` by `npix` array. Note 
            that with hard edged apertures the gradient can become 
            unusual.
        """


    @abstractmethod
    def _basis(self : Layer, n : int) -> Tensor:
        """
        Generate the basis. Requires a single run after which,
        the basis is cached and can be used with no computational 
        cost.  

        Parameters
        ----------
        n : int 
            The number of terms in the basis to generate. 

        Returns
        -------
        basis : Tensor
            The basis polynomials evaluated on the square arrays
            containing the apertures until `maximum_radius`.
            The leading dimension is `n` long and 
            each stacked array is a basis term. The final shape is:
            ```py
            basis.shape == (n, npix, npix)
            ```
        """


    def save(self : Layer, file_name : str, n : int) -> None:
        """
        Save the basis to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the basis terms to.
        n : int
            The number of terms in the basis to generate in the save.
        """
        basis = self._basis
        with open(file_name, "w") as save:
            save.write(basis)


    def __call__(self : Layer, parameters : dict) -> dict:
        """
        Apply a phase shift to the wavefront based on the basis 
        terms that have been generated.

        Overrides
        ---------
        __call__ : Layer
            Provides a concrete implementation of the `__call__` method
            defined in the abstract base class `Layer`.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of the model. 
            The dictionary must satisfy `parameters.get("Wavefront")
            != None`. 

        Returns
        -------
        parameters : dict
            The parameter, parameters, with the "Wavefront"; key
            value updated. 
        """
        wavefront = parameters["Wavefront"]
        parameters["Wavefront"] = wavefront.add_phase(self._basis)
        return parameters
        

class PolygonalBasis(BasisPhase, ABC):
    """
    Orthonormalises the zernike basis over a polygon to provide
    a basis on a polygonal aperture.

    Attributes
    ----------
    theta : float, radians
        This represents the angle of rotation from the positive x 
        axis. 
    phi : float, radians
        This represents the angle of shear measured from the positive 
        y axis.
    """
    theta : float
    phi : float


    def __init__(self : Layer, nterms : int, npix : int, 
            rmax : float, theta : float, phi : float, x : float,
            y : float) -> Layer:
        """
        Parameters
        ---------- 
        nterm : int 
            The number of polynomials to generate. This is not a 
            gradable parameter.
        npix : int
            The number of pixels in the image that is to be generated.
        rmax : float
            The radius of the smallest circle that can fully enclose the 
            aperture upon which the basis will be orthonormalised. The 
            coordinates hadnled internally are normalised, but this 
            quantity can be any number. > 1 is a magnification and < 1
            makes it smaller. 
        theta : float, radians
            This represents the angle of rotation from the positive x 
            axis. 
        phi : float, radians
            This represents the angle of shear measured from the positive 
            y axis.
        x : float
            The _x_ coordinates of the centre of the aperture. This 
            occurs following the normalisation so that the type can 
            be given as a `float` for gradient stability.
        y : float
            The _y_ coordinates of the centre of the aperture. This 
            occurs following the normalisation so that the type can 
            be given as a `float` for gradient stability.
        """
        super().__init__(x, y, rmax, npix, nterms)
        self.theta = np.asarray(theta).astype(float)
        self.phi = np.asarray(phi).astype(float)


    @functools.partial(jax.vmap, in_axes=(None, 0))
    def _noll_index(self : Layer, j : int) -> tuple:
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


    def _radial_zernike(self : Layer, n : int, m : int,
            rho : Matrix) -> Tensor:
        """
        The radial zernike polynomial.

        Parameters
        ----------
        n : int
            The first index number of the zernike polynomial to forge
        m : int 
            The second index number of the zernike polynomial to forge.
        rho : Matrix
            The radial positions of the aperture. Passed as an argument 
            for speed.

        Returns
        -------
        radial : Tensor
            An npix by npix stack of radial zernike polynomials.
        """
        m, n = np.abs(m), np.abs(n)
        upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1
        rho = np.tile(rho, (MAX_DIFF, 1, 1))

        murder_weapon = (np.arange(MAX_DIFF) < upper)

        k = np.arange(MAX_DIFF) * murder_weapon
        coefficients = (-1) ** k * factorial(n - k) / \
            (factorial(k) * \
                factorial(((n + m) / 2).astype(int) - k) * \
                factorial(((n - m) / 2).astype(int) - k))
        radial = coefficients.reshape(MAX_DIFF, 1, 1) *\
            rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1) *\
            murder_weapon.reshape(MAX_DIFF, 1, 1)
         
        return radial.sum(axis=0)


    def _zernikes(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Calculate the zernike basis on a square pixel grid. 

        Parameters
        ----------
        number : int
            The number of zernike basis terms to calculate.
            This is a static argument to jit because the array
            size depends on it.
        pixels : int
            The number of pixels along one side of the zernike image
            for each of the n zernike polynomials.
        coordinates : Tensor
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        zernike : Tensor 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        j = np.arange(1, self.nterms + 1).astype(int)
        n, m = self._noll_index(j)
        coordinates = cartesian_to_polar(coordinates)

        # NOTE: The idea is to generate them here at the higher level 
        # where things will not change and we will be done. 
        rho = coordinates[0]
        theta = coordinates[1]

        aperture = (rho <= 1.).astype(int)

        # In the calculation of the noll coefficient we must define 
        # between the m == 0 and and the m != 0 case. I have done 
        # this in place by casting the logical operation to an int. 

        normalisation_coefficients = \
            (1 + (np.sqrt(2) - 1) * (m != 0).astype(int)) \
            * np.sqrt(n + 1)

        radial_zernikes = np.zeros((self.nterms,) + rho.shape)
        for i in np.arange(self.nterms):
            radial_zernikes = radial_zernikes\
                .at[i]\
                .set(self._radial_zernike(n[i], m[i], rho))

        # When m < 0 we have the odd zernike polynomials which are 
        # the radial zernike polynomials multiplied by a sine term.
        # When m > 0 we have the even sernike polynomials which are 
        # the radial polynomials multiplies by a cosine term. 
        # To produce this result without logic we can use the fact
        # that sine and cosine are separated by a phase of pi / 2
        # hence by casting int(m < 0) we can add the nessecary phase.
        out_shape = (self.nterms, 1, 1)

        theta = np.tile(theta, out_shape)
        m = m.reshape(out_shape)
        phase_mod = (m < 0).astype(int) * np.pi / 2
        phase = np.cos(np.abs(m) * theta - phase_mod)

        normalisation_coefficients = \
            normalisation_coefficients.reshape(out_shape)
        
        return normalisation_coefficients * radial_zernikes \
            * aperture * phase 


    def _orthonormalise(self : Layer, aperture : Matrix, 
            zernikes : Tensor) -> Tensor:
        """
        The hexike polynomials up until `number_of_hexikes` on a square
        array that `number_of_pixels` by `number_of_pixels`. The 
        polynomials can be restricted to a smaller subset of the 
        array by passing an explicit `maximum_radius`. The polynomial
        will then be defined on the largest hexagon that fits with a 
        circle of radius `maximum_radius`. 
        
        Parameters
        ----------
        aperture : Matrix
            An array representing the aperture. This should be an 
            `(npix, npix)` array. 
        number_of_hexikes : int = 15
            The number of basis terms to generate. 
        zernikes : Tensor
            The zernike polynomials to orthonormalise on the aperture.
            This tensor should be `(nterms, npix, npix)` in size, where 
            the first axis represents the noll indexes. 

        Returns
        -------
        hexikes : Tensor
            The hexike polynomials evaluated on the square arrays
            containing the hexagonal apertures until `maximum_radius`.
            The leading dimension is `number_of_hexikes` long and 
            each stacked array is a basis term. The final shape is:
            ```py
            hexikes.shape == (number_of_hexikes, number_of_pixels, number_of_pixels)
            ```
        """
        pixel_area = aperture.sum()
        basis = np.zeros(zernikes.shape).at[0].set(aperture)
        
        for j in np.arange(1, self.nterms):
            intermediate = zernikes[j] * aperture

            coefficient = -1 / pixel_area * \
                (zernikes[j] * basis[1 : j + 1] * aperture)\
                .sum(axis = (1, 2))\
                .reshape(j, 1, 1) 

            intermediate += (coefficient * basis[1 : j + 1])\
                .sum(axis = 0)
            
            basis = basis\
                .at[j]\
                .set(intermediate / \
                    np.sqrt((intermediate ** 2).sum() / pixel_area))
        
        return basis


    def _magnify(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Enlarge or shrink the coordinate system, by the inbuilt 
        amount specified by `self._rmax`.

        Parameters
        ----------
        coordinates : Tensor
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Tensor
            The enlarged or shrunken coordinate system.
        """
        return 1 / self.rmax * coordinates


    def _rotate(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Rotate the coordinate system by a pre-specified amount,
        `self._theta`

        Parameters
        ----------
        coordinates : Tensor
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Tensor
            The rotated coordinate system. 
        """
        rotation_matrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]])            
        return np.apply_along_axis(np.matmul, 0, coordinates, 
            rotation_matrix) 


    def _shear(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Shear the coordinate system by the inbuilt amount `self._phi`.

        Parameters
        ----------
        coordinates : Tensor
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Tensor
            The sheared coordinate system. 
        """
        return coordinates\
            .at[0]\
            .set(coordinates[0] - coordinates[1] * np.tan(self.phi)) 


    def _offset(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Offset the coordinate system by prespecified amounts in both
        the `x` and `y` directions. 

        Parameters
        ----------
        coordinates : Tensor
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Tensor
            The translated coordinate system. 
        """
        return coordinates\
            .at[0]\
            .set(coordinates[0] - self.x)\
            .at[1]\
            .set(coordinates[1] - self.y)


    def _basis(self : Layer):
        """
        Generate the basis. Requires a single run after which,
        the basis is cached and can be used with no computational 
        cost.  

        Returns
        -------
        basis : Tensor
            The basis polynomials evaluated on the square arrays
            containing the apertures until `maximum_radius`.
            The leading dimension is `n` long and 
            each stacked array is a basis term. The final shape is:
            `(n, npix, npix)`
        """
        coordinates = self._coordinates()
        zernikes = self._zernikes(coordinates)
        aperture = self._aperture(coordinates)

        return self._orthonormalise(aperture, zernikes)  


    def _coordinates(self : Layer) -> Tensor:
        """
        Generate the transformed coordinate system for the aperture.

        Returns
        -------
        coordinates : Tensor
            The coordinate system in the rectilinear view, with the
            x and y coordinates stacked above one another.
        """
        coordinates = self._shear(
            self._rotate(
                self._offset(
                    self._magnify(
                        2 / self.npix * get_pixel_positions(self.npix)))))
        return coordinates


    def set_theta(self : Layer, theta : float) -> Layer:
        """
        Parameters
        ----------
        theta : float
            The angle of rotation from the positive x-axis.  

        Returns
        -------
        basis : HexagonalBasis 
            The rotated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.theta, self, theta)


    def set_magnification(self : Layer, rmax : float) -> Layer:
        """
        Parameters
        ----------
        rmax : float
            The radius of the smallest circle that can completely 
            enclose the aperture.

        Returns
        -------
        basis : HexagonalBasis
            The magnified hexagonal basis.
        """
        return eqx.tree_at(lambda basis : basis.rmax, self, rmax)


    def set_shear(self : Layer, phi : float) -> Layer:
        """
        Parameters
        ----------
        phi : float
            The angle of shear from the positive y-axis.

        Returns
        -------
        basis : HexagonalBasis
            The sheared hexagonal basis.
        """
        return eqx.tree_at(lambda basis : basis.phi, self, phi)      


    def set_x_offset(self : Layer, x : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The x coordinate of the centre of the hexagonal
            aperture.

        Returns
        -------
        basis : HexagonalBasis
            The translated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.x, self, x)


    def set_y_offset(self : Layer, y : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The y coordinate of the centre of the hexagonal
            aperture.

        Returns
        -------
        basis : HexagonalBasis
            The translated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.y, self, y)


class HexagonalBasis(PolygonalBasis):
    """
    A basis on hexagonal surfaces. This is based on [this]
    (https://github.com/spacetelescope/poppy/poppy/zernike.py)
    code from the _POPPY_ library, but ported into [_JAX_]
    (https://github.com/google/jax) and allowed to perform 
    more complicated spatial transformations. 
    """
    def __init__(self : Layer, nterms : int, npix : int,
            rmax : float, theta : float,
            phi : float, x : float, y : float) -> Layer:
        """
        Parameters
        ----------
        nterms : int
            The number of terms to generate in the basis. 
        npix : int
            The number of pixels in the image that is to be generated.
        rmax : float
            The radius of the smallest circle that can fully enclose the 
            aperture upon which the basis will be orthonormalised. The 
            coordinates hadnled internally are normalised, but this 
            quantity can be any number. > 1 is a magnification and < 1
            makes it smaller. 
        theta : float, radians
            This represents the angle of rotation from the positive x 
            axis. 
        phi : float, radians
            This represents the angle of shear measured from the positive 
            y axis.
        x : float
            The _x_ coordinates of the centre of the aperture. This 
            occurs following the normalisation so that the type can 
            be given as a `float` for gradient stability.
        y : float
            The _y_ coordinates of the centre of the aperture. This 
            occurs following the normalisation so that the type can 
            be given as a `float` for gradient stability.
        """
        super().__init__(nterms, npix, rmax, theta, phi, x, y)


    def _aperture(self : Layer,
            coordinates : Tensor,
            maximum_radius : float = 1.) -> Array:
        """
        Generate a binary mask representing the pixels occupied by 
        the aperture. 

        Parameters
        ----------
        coordinates : Tensor
            The coordinate grid as a stacked array that is 
            `(2, npix, npix)`, where the leading axis denotes the x and 
            y coordinate sets. This Tensor must be normalised, if using 
            the inbuilt `get_pixel_positions` normalisation can be done
            via `2 / npix`.
        maximum_radius : float 
            The radius of the smallest circle that can completely contain
            the entire aperture. 

        Returns
        -------
        aperture : Array
            The bitmask that represents the circular aperture.        
        """
        x, y = coordinates[0], coordinates[1]

        rectangle = (np.abs(x) <= maximum_radius / 2.) \
            & (np.abs(y) <= (maximum_radius * np.sqrt(3) / 2.))

        left_triangle = (x <= - maximum_radius / 2.) \
            & (x >= - maximum_radius) \
            & (np.abs(y) <= (x + maximum_radius) * np.sqrt(3))

        right_triangle = (x >= maximum_radius / 2.) \
            & (x <= maximum_radius) \
            & (np.abs(y) <= (maximum_radius - x) * np.sqrt(3))

        hexagon = rectangle | left_triangle | right_triangle
        return np.asarray(hexagon).astype(float)


class JWSTPrimaryApertureSegment(PolygonalAperture):
    """
    A dLux implementation of the JWST primary aperture segment.
    The segments are sketched and drawn below:

                            +---+
                           *     *
                      +---+  B1   +---+
                     *     *     *     *
                +---+  C6   +---+  C1   +---+
               *     *     *     *     *     *
              +  B6   +---+  A1   +---+  B2   +
               *     *     *     *     *     *
                +---+  A6   +---+  A2   +---+
               *     *     *     *     *     *
              +  C5   +---+       +---+  C2   +
               *     *     *     *     *     * 
                +---+  A5   +---+  A3   +---+
               *     *     *     *     *     *   
              +  B5   +---+  A4   +---+  B3   +
               *     *     *     *     *     *
                +---+  C4   +---+  C3   +---+
                     *     *     *     *
                      +---+  B4   +---+
                           *     *         
                            +---+    

    The data for the vertices is retrieved from WebbPSF and the 
    syntax for using the class is as follows:

    >>> npix = 1008 # Total number of pixels for the entire primary
    >>> appix = 200 # Pixels for this specific aperture. 
    >>> C1 = JWSTPrimaryApertureSegment("C1-1", npix, appix)
    >>> aperture = C1()

    If you want to only model one mirror then appix and npix can be 
    set to the same. The assumption is that the entire aperture is 
    going to be modelled. 

    To use the aperture to generate an orthonormal basis on the not 
    quite a hexagon we use the following code. 

    >>> basis = Basis(C1(), nterms)()

    To learn the rotation, shear and other parameters of the mirror 
    we can provide this functionality to the constructor of the 
    aperture. For example:
    
    >>> C1 = JWSTPrimaryApertureSegment(
    ...     segement : str = "C1-1",
    ...     number_of_pixels : int = npix,
    ...     aperture_pixels : int = appix,
    ...     rotation : float = 0.1,
    ...     shear : float = 0.1,
    ...     magnification = 1.001)
    >>> basis = Basis(C1(), nterms)()    
    """


               
    def _wrap(array : Vector, order : Vector) -> tuple:
        """
        Re-order an array and duplicate the first element as an additional
        final element. Satisfies the postcondition `wrapped.shape[0] ==
        array.shape[0] + 1`. This is just a helper method to simplify 
        external object and is not physically important (Only invoke 
        this method if you know what you are doing)

        Parameters
        ----------
        array : Vector
            The 1-dimensional vector to sort and append to. Must be one 
            dimesnional else unexpected behaviour can occur.
        order : Vector
            The new order for the elements of `array`. Will be accessed 
            by invoking `array.at[order]` hence `order` must be `int`
            `dtype`.

        Returns
        -------
        wrapped : Vector
            `array` with `array[0]` appended to the end. The dimensions
            of `array` are also expanded twofold so that the final
            shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
            This is just for the vectorisation demanded later in the 
            code.
        """
        _array = np.zeros((array.shape[0] + 1,))\
            .at[:-1]\
            .set(array.at[order].get())\
            .reshape(-1, 1, 1)
        return _array.at[-1].set(_array[0])
        

    def _vertices(vertices : Matrix) -> tuple:
        """
        Generates the vertices that are compatible with the rest of 
        the transformations from the raw data vertices.

        Parameters
        ----------
        vertices : Matrix, meters
            The vertices loaded from the WebbPSF module. 

        Returns
        -------
        x, y, angles : tuple 
            The vertices in normalised positions and wrapped so that 
            they can be used in the generation of the compound aperture.
            The `x` is the x coordinates of the vertices, the `y` is the 
            the y coordinates and `angles` is the angle of the vertex. 
        """
        _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
        _y = (vertices[:, 1] - np.mean(vertices[:, 1]))

        _angles = np.arctan2(_y, _x)
        _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)

        # By default the `np.arctan2` function returns values within the 
        # range `(-np.pi, np.pi)` but comparisons are easiest over the 
        # range `(0, 2 * np.pi)`. This is where the logic implemented 
        # above comes from. 

        order = np.argsort(_angles)

        x = _wrap(_x, order)
        y = _wrap(_y, order)
        angles = _wrap(_angles, order).at[-1].add(2 * np.pi)

        # The final `2 * np.pi` is designed to make sure that the wrap
        # of the first angle is within the angular coordinate system 
        # associated with the aperture. By convention this is the
        # range `angle[0], angle[0] + 2 * np.pi` what this means in 
        # practice is that the first vertex appearing in the array 
        # is used to chose the coordinate system in angular units. 

        return x, y, angles


    def _offset(vertices : Matrix, pixel_scale : float) -> tuple:
        """
        Get the offsets of the coordinate system.

        Parameters
        ----------
        vertices : Matrix 
            The unprocessed vertices loaded from the JWST data file.
            The correct shape for this array is `vertices.shape == 
            (2, number_of_vertices)`. 
        pixel_scale : float, meters
            The physical size of each pixel along one of its edges.

        Returns 
        -------
        x_offset, y_offset : float, meters
            The x and y offsets in physical units. 
        """
        x_offset = np.mean(vertices[:, 0]) / pixel_scale
        y_offset = np.mean(vertices[:, 1]) / pixel_scale
        return x_offset, y_offset


    def _pixel_scale(vertices : Matrix, pixels : int) -> float:
        """
        The physical dimesnions of a pixel along one edge. 

        Parameters
        ----------
        vertices : Matrix
            The vertices of the aperture in a two dimensional array.
            The pixel scale is assumed to be the same in each dimension
            so only the first row of the vertices is used.
        pixels : int
            The number of pixels that this aperture is going to 
            occupy. 

        Returns
        -------
        pixel_scale : float, meters
            The physical length along one edge of a pixel. 
        """
        return vertices[:, 0].ptp() / pixels


    def _coordinates(number_of_pixels : int, vertices : Matrix,
            aperture_pixels : int, phi_naught : float) -> tuple:
        """
        Generates the vectorised coordinate system associated with the 
        aperture.

        Parameters
        ----------
        number_of_pixels : int
            The total number of pixels to generate. This is typically 
            more than `aperture_pixels` as this is used in the padding 
            of the array for the generation of compound apertures.
        vertices : Matrix, meters
            The vertices loaded from the file.
        aperture_pixels : int
            The number of pixels across that the individual aperture.
        phi_naught : float 
            The angle substending the first vertex. 

        Returns 
        -------
        rho, theta : tuple[Tensor]
            The stacked coordinate systems that are typically passed to 
            `_segments` to generate the segments.
        """
        pixel_scale = _pixel_scale(vertices, aperture_pixels)
        x_offset, y_offset = _offset(vertices, pixel_scale)

        positions = get_radial_positions(number_of_pixels,
            -x_offset, -y_offset)

        rho = positions[0] * pixel_scale

        theta = positions[1] 
        theta += 2 * np.pi * (positions[1] < 0.)
        theta += 2 * np.pi * (theta < phi_naught)

        rho = np.tile(rho, (vertices.shape[0], 1, 1))
        theta = np.tile(theta, (vertices.shape[0], 1, 1))
        return rho, theta


    def _edges(x : Vector, y : Vector, rho : Tensor, 
            theta : Tensor) -> Tensor:
        """
        Generate lines connecting adjacent vertices.

        Parameters
        ----------
        x : Vector
            The x positions of the vertices.
        y : Vector
            The y positions of the vertices.
        rho : Tensor, meters
            Represents the radial distance of every point from the 
            centre of __this__ aperture. 
        theta : Tensor, Radians
            The angle associated with every point in the final bitmap.

        Returns
        -------
        edges : Tensor
            The edges represented as a Bitmap with the points inside the 
            edge marked as 1. and outside 0. The leading axis contains 
            each unique edge and the corresponding matrix is the bitmap.
        """
        # This is derived from the two point form of the equation for 
        # a straight line (eq. 1)
        # 
        #           y_2 - y_1
        # y - y_1 = ---------(x - x_1)
        #           x_2 - x_1
        # 
        # This is rearranged to the form, ay - bx = c, where:
        # - a = (x_2 - x_1)
        # - b = (y_2 - y_1)
        # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
        # we can then drive the transformation to polar coordinates in 
        # the usual way through the substitutions; y = r sin(theta), and 
        # x = r cos(theta). The equation is now in the form 
        #
        #                  c
        # r = ---------------------------
        #     a sin(theta) - b cos(theta) 
        #
        a = (x[1:] - x[:-1])
        b = (y[1:] - y[:-1])
        c = (a * y[:-1] - b * x[:-1])

        linear = c / (a * np.sin(theta) - b * np.cos(theta))
        return rho < linear 
        

    def _wedges(phi : Vector, theta : Tensor) -> Tensor:
        """
        The angular bounds of each segment of an individual hexagon.

        Parameters
        ----------
        phi : Vector
            The angles corresponding to each vertex in order.
        theta : Tensor, Radians
            The angle away from the positive x-axis of the coordinate
            system associated with this aperture. Please note that `theta`
            May not start at zero. 

        Returns 
        -------
        wedges : Tensor 
            The angular bounds associated with each pair of vertices in 
            order. The leading axis of the Tensor steps through the 
            wedges in order arround the circle. 
        """
        # A wedge simply represents the angular bounds of the aperture
        # I have demonstrated below with a hexagon but understand that
        # these bounds are _purely_ angular (see fig 1.)
        #
        #               +-------------------------+
        #               |                ^^^^^^^^^|
        #               |     +--------+^^^^^^^^^^|
        #               |    /        /^*^^^^^^^^^|
        #               |   /        /^^^*^^^^^^^^|
        #               |  /        /^^^^^*^^^^^^^|
        #               | +        +^^^^^^^+^^^^^^|
        #               |  *              /       |
        #               |   *            /        |
        #               |    *          /         |
        #               |     +--------+          |
        #               +-------------------------+
        #               figure 1: The angular bounds 
        #                   between the zeroth and the 
        #                   first vertices. 
        #
        return (phi[:-1] < theta) & (theta < phi[1:])


    def _segments(x : Vector, y : Vector, phi : Vector, 
            theta : Tensor, rho : Tensor) -> Tensor:
        """
        Generate the segments as a stacked tensor. 

        Parameters
        ----------
        x : Vector
            The x coordinates of the vertices.
        y : Vector
            The y coordinates of the vertices.
        phi : Vector
            The angles associated with each of the vertices in the order. 
        theta : Tensor
            The angle of every pixel associated with the coordinate system 
            of this aperture. 
        rho : Tensor
            The radial positions associated with the coordinate system 
            of this aperture. 

        Returns 
        -------
        segments : Tensor 
            The bitmaps corresponding to each vertex pair in the vertices.
            The leading dimension contains the unique segments. 
        """
        edges = _edges(x, y, rho, theta)
        wedges = _wedges(phi, theta)
        return edges & wedges
        

    def _aperture(vertices : Matrix, number_of_pixels : int, 
            aperture_pixels : int) -> Matrix:
        """
        Generate the BitMap representing the aperture described by the 
        vertices. 

        Parameters
        ----------
        vertices : Matrix 
            The vertices describing the approximately hexagonal aperture. 
        number_of_pixels : int
            The number of pixels that represent the compound aperture.
        aperture_pixels : int
            The number of pixels that represent the single aperture. 

        Returns
        -------
        aperture : Matrix 
            The Bit-Map that represents the aperture. 
        """
        x, y, phi = _vertices(vertices)
        rho, theta = _coordinates(number_of_pixels, vertices, 
            aperture_pixels, phi[0])
        segments = _segments(x, y, phi, theta, rho)
        return segments.sum(axis=0)


    def _load(self : Layer):
        """
        
        """


vertices = np.stack(jax.tree_util.tree_map(
    lambda leaf : leaf[1], 
    JWST_PRIMARY_SEGMENTS,
    is_leaf = lambda leaf : isinstance(leaf[0], str)))

aperture = jax.vmap(_aperture, in_axes=(0, None, None))(vertices, 1008, 200)
pyplot.imshow(aperture.sum(axis=0))
pyplot.show()
