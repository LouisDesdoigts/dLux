"""
src/dev/layers.py
-----------------
Development script for the new layers structure.
"""

# NOTE: Experimental code by @Jordan-Dennis is below.
import equinox as eqx
import jax.numpy as np
import jax 
import functools


from typing import TypeVar, Dict
from dLux.utils import (get_radial_positions, get_pixel_vector, 
    get_pixel_positions)
from abc import ABC, abstractmethod 


Array = TypeVar("Array")
Tensor = TypeVar("Tensor")
Layer = TypeVar("Layer")
Matrix = TypeVar("Matrix")


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


    def __call__(self : Layer, parameters : Dict) -> Dict:
        """
        Apply the aperture to an incoming wavefront.

        Parameters
        ----------
        parameters : Dict
            A dictionary containing the parameters of the model. 
            The dictionary must satisfy `parameters.get("Wavefront")
            != None`. 

        Returns
        -------
        parameters : Dict
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
    _x : float
    _y : float
    _r : float
    _npix : int
    _nterms : int


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
        self._x = np.asarray(x).astype(float)
        self._y = np.asarray(y).astype(float)
        self._r = np.asarray(r).astype(float)
        self._npix = int(npix)
        self._nterms = int(nterms)


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
    @functools.cached_property 
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
        # TODO: This may cause `jit`ting problems in some cases. It
        # will force me to work on the zernike `jit`


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
        # TODO: Implement either a LoadedBasis abstract class 
        # or add conditional logic for a load method. 


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
        parameters : Dict
            A dictionary containing the parameters of the model. 
            The dictionary must satisfy `parameters.get("Wavefront")
            != None`. 

        Returns
        -------
        parameters : Dict
            The parameter, parameters, with the "Wavefront"; key
            value updated. 
        """
        wavefront = parameters["Wavefront"]
        parameters["Wavefront"] = wavefront.add_phase(self._basis)
        return parameters
        

class PolygonBasis(BasisPhase, ABC):
    """
    Orthonormalises the zernike basis over a polygon to provide
    a basis on a polygonal aperture.
    """
    @abstractmethod
    def _vertices(self : Layer) -> Matrix:
        """
        Returns
        -------
        vertices : Matrix
            A 2 by number of vertices matrix containing the x 
            x coordinates along the 0th row and the y coordinates
            along the 1st row. The nth column is the x and y coordinates
            of the nth vertice. 
        """


    def _aperture(self : Layer) -> Matrix:
        """
        Returns
        -------
        aperture : Matrix 
            The soft-edged aperture of the given polygonal shape.
        """
        positions = get_radial_positions(self._npix, -self._x, -self._y)
        rho = positions[0]
        theta = positions[1]

        # The non-PEP8 variable names come from the standard 
        # mathematical form y = mx + c. This can also be expressed
        # as ay + bx = c. If we use the two point form for the 
        # standard expression we get:
        #
        #           y_2 - y_1
        # y - y_1 = ---------(x - x_1)
        #           x_2 - x_1
        #
        # This can be rearranged into the form. 
        #
        # (x_2 - x_1) y - (y_2 - y_1) x = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
        # 
        # More simply the form ay + bx = c, where a, b and c are 
        # functions of the verticies points; x_1, x_2, y_1 and y_2.
        # In polar form this gives us:
        # 
        #                  c
        # r = ---------------------------
        #     a cos(theta) + b sin(theta)
        #
        # Polar coordinates are used because we can always use r >
        # to specify the aperture, avoiding the use of logic.

        vertices = self._vertices()

        i = np.arange(-1, vertices.shape[0])
        a = (vertices[i + 1, 0] - vertices[i, 0])
        b = (vertices[i + 1, 1] - vertices[i, 1])
        c = (vertices[i + 1, 0] - vertices[i, 0]) * vertices[i, 1] -\
            (vertices[i + 1, 1] - vertices[i, 1]) * vertices[i, 0]

        rho = np.expand_dims(rho, axis=2).tile(rho.shape[0])

        @jax.vmap
        def less_than(array : Matrix, comparator : Matrix) -> Matrix:
            """
            < comparator for Tensors.

            Parameters
            ----------
            array : Matrix
                The array on the left of the <.
            comparator : Matrix
                The array on the right of the <.

            Returns
            is_less_than : Matrix
                Elementwise comparison of the arrays. 
            """
            return array < comparator

        aperture = less_than(rho, 
            c / (a * np.sin(theta) + b * np.cos(theta)))

        return aperture
         

    def _gram_schmidt(self : Layer, zernikes : Tensor,
            aperture : Matrix) -> Tensor:
        """
        Orthormalises the zernike polynomials on the aperture. 

        Parameters
        ----------
        zernikes : Tensor
            The zernike polynomials that are to be used in the
            orthonormalisation. This Tensor should be number by 
            pixels by pixels. 
        aperture : Matrix
            The aperture. This could be calculated interally but to 
            save computation time this is passed as an argument.

        Returns
        -------
        basis : Tensor
            The basis polynomials on the aperture. 
        """
        pixel_area = aperture.sum()
        basis = np.zeros(zernikes.shape).at[0].set(aperture)
        
        j = np.arange(1, zernikes.shape[0])

        coefficients = -1 / pixel_area * \
           ((zernikes[j + 1] * basis[1 : j + 1]) * aperture)\
            .sum(axis = 1) 
        
        intermediates = zernikes[j + 1] * aperture
        intermediate += (coefficients * basis[1 : j + 1]).sum(axis = 1)

        return basis\
            .at[j + 1]\
            .set(intermediates / \
                np.sqrt((intermediates ** 2).sum(axis=(1, 2)) \
                    / pixel_area))


    @functools.cached_property
    def _basis(self : Layer) -> Tensor:
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
        aperture = self._aperture()
        zernikes = self._zernike_basis(self._nterms, self._npix)
        basis = self._gram_schmidt(zernikes, aperture)
        return basis


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


    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def _zernike(self : Layer, n : int, m : int) -> Matrix: 
        """
        Generate the zernike specified by the indexes n and m.

        Parameters
        ----------
        n : int
            The first integer that specifies the zernike polynomial.
        m : int 
            The second integer that specifies the zernike polynomial,
            must be within the range -n < m < n.

        Returns
        zernike : Matrix
            The zernike evaluated at the centre of a square matrix.  
        """
        positions = get_radial_positions(self._npix, -self._x, -self._y)
        rho = positions[0] * 2 / self._npix
        theta = positions[1]

        aperture = (rho <= 1.).astype(int)

        # In the calculation of the noll coefficient we must define 
        # between the m == 0 and and the m != 0 case. To determine 
        # if m == 0 we can use (m >> 1 == m) because 0 represented 
        # in binary is all 0s and can be translated without destroying 
        # that information. All other numbers will change under this
        # transformation. For the math nerds m is a fixed point of 
        # the left bit shift operation.

        norm_coeff = (1 + (np.sqrt(2) - 1) * (m != 0).astype(int)) *\
            np.sqrt(n + 1)
        radial_zernike = self._radial_zernike(n, m, rho)

        # When m < 0 we have the odd zernike polynomials which are 
        # the radial zernike polynomials multiplied by a sine term.
        # When m > 0 we have the even sernike polynomials which are 
        # the radial polynomials multiplies by a cosine term. 
        # To produce this result without logic we can use the fact
        # that sine and cosine are separated by a phase of pi / 2
        # hence by casting int(m < 0) we can add the nessecary phase.

        return norm_coeff * radial_zernike * aperture *\
            np.cos(np.abs(m) * theta - (m < 0).astype(int) * np.pi / 2)


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
        radial = np.zeros(rho.shape)
        m, n = np.abs(m), np.abs(n)
        upper = ((n - m) / 2).astype(int) + 1

        # This is for the `lax.while_loop` that is present in the 
        # `radial_zernike` calculation.
        def _body_fun(index_and_term : tuple) -> tuple:
            """
            The summation inside the radial polynomial calculation.

            Parameters
            ----------
            index_and_term : tuple
                The index of the summation and the radial polynomial
                summed to index - 1 terms.

            Returns
            -------
            radial : Matrix
                Radial but also added to the new index k.
            """
            k, radial = index_and_term
            coefficient = (-1) ** k * factorial(n - k) / \
                (factorial(k) * \
                    factorial(((n + m) / 2).astype(int) - k) * \
                    factorial(((n - m) / 2).astype(int) - k))
            return k + 1, radial + coefficient * rho ** (n - 2 * k)


        # This is for the `lax.while_loop` that is in the `radial_zernike`
        # calculation.
        def _cond_fun(index_and_term : tuple) -> bool:
            """
            Is the summation complete?        

            Parameters
            ----------
            index_and_term : tuple
                The index of the summation and the radial polynomial
                summed to index - 1 terms.

            Returns
            -------
            new_index_and_term : tuple
                Radial but also added to the new index k.
            """
            k, _ = index_and_term
            return (k < upper).astype(bool)

        return jax.lax.while_loop(
            _cond_fun, _body_fun, (0, radial))[1]


    def _zernike_basis(self : Layer) -> Tensor:
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

        Returns
        -------
        zernike : Tensor 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        j = np.arange(1, self._nterms + 1).astype(int)
        n, m = self._noll_index(j)
        return self._zernike(n, m)


class HexagonalBasis(PolygonBasis):
    """
    Generates an orthonormal hexagonal basis often called the Hexikes.
    """
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
        super().__init__(x, y, r, npix, nterms)


    def _vertices(self : Layer) -> Matrix:
        """
        Returns
        -------
        vertices : Matrix
            A 2 by number of vertices matrix containing the x 
            x coordinates along the 0th row and the y coordinates
            along the 1st row. The nth column is the x and y coordinates
            of the nth vertice. 
        """
        theta = np.arange(0, 2 * np.pi, np.pi / 3)
        return np.array([self._r * np.cos(theta) - self._x, 
            self._r * np.sin(theta) - self._y])


#    def _tilt():
#    def _rotate():
#    def _shear()
#    def _offset():

# NOTE: This is the short term version I may actually end up adopting
# this as the long term version. 
class HexagonalBasis(eqx.Module):
    """
    A basis on hexagonal surfaces. This is based on [this]
    (https://github.com/spacetelescope/poppy/poppy/zernike.py)
    code from the _POPPY_ library, but ported into [_JAX_]
    (https://github.com/google/jax) and allowed to perform 
    more complicated spatial transformations. 

    Attributes
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
    nterms : int 
    npix : int
    rmax : float
    theta : float
    phi : float
    x : float
    y : float


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
        self.nterms = int(nterms)
        self.npix = int(npix)
        self.rmax = np.asarray(rmax).astype(float)
        self.theta = np.asarray(theta).astype(float)
        self.phi = np.asarray(phi).astype(float)
        self.x = np.asarray(x).astype(float)
        self.y = np.asarray(y).astype(float)

    
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


    def _radial_zernike(self : Layer, n : int, m : int, upper : int,
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
        rho = np.tile(rho, (upper, 1, 1))

        k = np.arange(upper)
        coefficients = (-1) ** k * factorial(n - k) / \
            (factorial(k) * \
                factorial(((n + m) / 2).astype(int) - k) * \
                factorial(((n - m) / 2).astype(int) - k))
        radial = coefficients.reshape(upper, 1, 1) \
            * rho ** (n - 2 * k).reshape(upper, 1, 1)
        
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

        summation_limits = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1
        radial_zernikes = np.zeros((self.nterms,) + rho.shape)
        for i in np.arange(self.nterms):
            radial_zernikes = radial_zernikes\
                .at[i]\
                .set(self._radial_zernike(n[i], m[i], 
                    summation_limits[i], rho))

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


    def _othonormalise(self : Layer, aperture : Matrix, 
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
        
        j = np.arange(1, zernikes.shape[0])

        coefficients = -1 / pixel_area * \
           ((zernikes[j + 1] * basis[1 : j + 1]) * aperture)\
            .sum(axis = 1) 
        
        intermediates = zernikes[j + 1] * aperture
        intermediate += (coefficients * basis[1 : j + 1]).sum(axis = 1)

        return basis\
            .at[j + 1]\
            .set(intermediates / \
                np.sqrt((intermediates ** 2).sum(axis=(1, 2)) \
                    / pixel_area))


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
        return 1 / self._rmax * coordinates


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
            [np.cos(self._theta), -np.sin(self._theta)],
            [np.sin(self._theta), np.cos(self._theta)]])            
        return rotation_matrix @ coordinates


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
            .set(coordinates[0] - coordinates[1] * np.tan(self._phi)) 


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
            .set(coordinates[0] - self._x)\
            .at[1]\
            .set(coordinates[1] - self._y)


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
        coordinates = get_pixel_positions(self._npix)
        coordinates = self._offset(coordinates)
        coordinates = self._rotate(coordinates)
        coordinates = self._magnify(coordinates)
        coordinates = self._shear(coordinates)

        zernikes = self._zernikes(coordinates)
        aperture = self._aperture(coordinates)

        return self._orthonormalise(aperture, zernikes)        
