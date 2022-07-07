"""
src/dev/layers.py
-----------------
Development script for the new layers structure.
"""

# NOTE: Experimental code by @Jordan-Dennis is below. 
from typing import TypeVar, Dict
from dLux.utilities import (get_radial_coordinates, get_pixel_vector, 
    get_pixel_positions)
from abc import ABC, abstractmethod 

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
    @classmethod # Just thought I would through this in for fun.
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
    rmax : float


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
        x, y = get_pixel_positions(self.get_npix())

        rectangle = (np.abs(x) <= rmax / 2.) \
            & (np.abs(y) <= (x + 1) * np.sqrt(3))

        left_triangle = (x <= -0.5) \
            & (x >= -1) \
            & (np.abs(y) <= (x + 1) * np.sqrt(3))

        right_triangle = (x >= 0.5) \
            & (x <= 1) \
            & (np.abs(y) <= (1 - x) * np.sqrt(3))

        hexagon = rectangle | left_triangle | right_triangle
        return np.asarray(hexagon).astype(float)

