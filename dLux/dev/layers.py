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
    """
    _rmax : float


    def __init__(self : Layer, npix : int, rmax : float) -> Layer:
        """
        Parameters
        ----------
        npix : int
            The number of pixels along one side of the square array
        rmax : float 
            The outer radius of the smallest circle that contains the 
            hexagon.
        """
        super().__init__(npix)
        self.rmax = rmax


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
        x, y = _get_pixel_positions(number_of_pixels, x_pixel_offset,
            y_pixel_offset)

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
