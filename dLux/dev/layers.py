"""
src/dev/layers.py
-----------------
Development script for the new layers structure.
"""

# NOTE: Experimental code by @Jordan-Dennis is below. 
from typing import TypeVar, Dict
from dLux.utils import (get_radial_coordinates, get_pixel_vector, 
    get_pixel_positions)
from abc import ABC, abstractmethod 


Array = TypeVar("Array")
Tensor = TypeVar("Tensor")
Layer = TypeVar("Layer")
Array = TypeVar("Array")


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
        coords = 2 / self.get_npix() * get_radial_coordinates(self.get_npix())
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


class BasisPhase(Layer, ABC):
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


    def __init__(self : Layer, x : float, y : float, r : float,
            npix : int) -> Layer:
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
        """
        self._x = np.asarray(x).astype(float)
        self._y = np.asarray(y).astype(float)
        self._r = np.asarray(r).astype(float)
        self._npix = int(npix)


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
    @cached_property 
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
        

class PolygonBasis(PhaseBasis, ABC):
    """
    Orthonormalises the zernike basis over a polygon to provide
    a basis on a polygonal aperture.
    """
    @abstractmethod
    def _get_vertices(self : Layer) -> Matrix:
        """
        Returns
        -------
        vertices : Matrix
            A 2 by number of vertices matrix containing the x 
            x coordinates along the 0th row and the y coordinates
            along the 1st row. The nth column is the x and y coordinates
            of the nth vertice. 
        """


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
        n = np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1
        smallest_j_in_row = n * (n + 1) / 2 + 1 
        number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
        sign_of_shift = -(j & 1) + ~(j & 1) + 2
        base_case = (n & 1)
        m = sign_of_shift * (base_case + number_of_shifts * 2)
        return int(n), int(m)


    def _zernike(self : Layer, n : int, m : int) -> Tensor: 


    @cached_property
    def _basis(self : Layer):


class HexagonalBasis(PolygonBasis)

    



def hexike_basis(
        number_of_hexikes : int = 15, 
        number_of_pixels : int = 512,
        x_pixel_offset : int = 0,
        y_pixel_offset : int = 0,
        maximum_radius : float = 1.) -> Tensor:
    """
    The hexike polynomials up until `number_of_hexikes` on a square
    array that `number_of_pixels` by `number_of_pixels`. The 
    polynomials can be restricted to a smaller subset of the 
    array by passing an explicit `maximum_radius`. The polynomial
    will then be defined on the largest hexagon that fits with a 
    circle of radius `maximum_radius`. 
    
    Parameters
    ----------
    number_of_hexikes : int = 15
        The number of basis terms to generate. 
    number_of_pixels : int = 512
        The size of the array to compute the hexikes on.
    x_pixel_offset : int
        The offset of the aperture in the square output array in the 
        x direction.
    y_pixel_offset : int
        The offset of the aperture in the square output array in the 
        y direction. 
    maximum_radius : float = 1.
        The radius of the the smallest circle that can contain the 
        hexagonal surface. 

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
    aperture = _hexagonal_aperture(number_of_pixels, x_pixel_offset,
        y_pixel_offset, maximum_radius)

    pixel_area = aperture.sum()
    shape = (number_of_hexikes, number_of_pixels, number_of_pixels)
    zernikes = dLux.utils.zernike_basis(number_of_hexikes, 
        number_of_pixels)

    # So the issue is currently that I cannot pass a negative offset
    # The aperture is fine with these it is just that I cannot set 
    # them properly in the following array.
    if y_pixel_offset >= 0:
        if x_pixel_offset >= 0:
            offset_zernikes = np.zeros(shape)\
                .at[:, : number_of_pixels - y_pixel_offset, 
                    : number_of_pixels - x_pixel_offset]\
                .set(zernikes[:, y_pixel_offset :, x_pixel_offset :])
        else:
            offset_zernikes = np.zeros(shape)\
                .at[:, : number_of_pixels - y_pixel_offset, -x_pixel_offset :]\
                .set(zernikes[:, y_pixel_offset :, : x_pixel_offset])
    else:
        if x_pixel_offset >= 0:
            offset_zernikes = np.zeros(shape)\
                .at[:, -y_pixel_offset :, : number_of_pixels - x_pixel_offset]\
                .set(zernikes[:, : y_pixel_offset, x_pixel_offset :])
        else:
            offset_zernikes = np.zeros(shape)\
                .at[:, -y_pixel_offset :, -x_pixel_offset :]\
                .set(zernikes[:, : y_pixel_offset, : x_pixel_offset])



    offset_hexikes = np.zeros(shape).at[0].set(aperture)
    
    for j in np.arange(1, number_of_hexikes): # Index of the zernike
        intermediate = offset_zernikes[j + 1] * aperture

        coefficients = -1 / pixel_area * \
           ((offset_zernikes[j + 1] * offset_hexikes[1 : j + 1]) * aperture)\
            .sum(axis = 0) 

        intermediate += (coefficients * offset_hexikes[1 : j + 1])\
            .sum(axis = 0)

        offset_hexikes = offset_hexikes\
            .at[j + 1]\
            .set(intermediate / \
                np.sqrt((intermediate ** 2).sum() / pixel_area))

    return offset_hexikes
