"""
src/utilities/image.py
----------------------
A static class that implements centering conventions for dealing 
with para-axial arrays. 
"""
__author__ = "Jordan Dennis"
__date__ = "07/07/2022"


from typing import TypeVar


Vector = TypeVar("Vector")
Matrix = TypeVar("Matrix")
Tensor = TypeVar("Tensor")


class Image(object):
    """
    A static class that implements useful generative functions for 
    pixel arrays. 
    """
    @staticmethod
    def get_pixel_coordinates(number_of_pixels : int) -> Vector:
        """
        Generate the coordinates of the pixels along a para-axial 
        centered edge. 

        Parameters
        ----------
        number_of_pixels : int
            The number of pixels along the edge of the array.

        Returns
        -------
        coordinates : Vector
            The pixel coordinates along the edge.
        """
        return np.arange(number_of_pixels) - number_of_pixels / 2. + .5
        

    @staticmethod
    def get_pixel_positions(number_of_pixels : int) -> Tensor:
        """
        Generate the grid of coordinates for the pixels with para-axial
        centering.

        Parameters
        ----------
        number_of_pixels : int
            The number of pixels along one edge of the image.

        Returns
        -------
        positions : Matrix
            A grid of pixel coordinates. 
        """
        coordinates = self.get_pixel_coordinates
        return np.array(np.meshgrid(coordinates, coordinates))


    @staticmethod
    def get_radial_coordinates(number_of_pixels : int) -> Matrix:
        """
        Generate the radial coordinates of each pixel. 

        Parameters
        ----------
        number_of_pixels : int
            The number of pixels along one edge of the square array.

        Returns
        -------
        positions : Matrix
            A grid of the radial coordinates.
        """
        # NOTE: I think CircularAperture is Broken 
        return np.sum(self.get_pixel_coordinates(number_of_pixels) ** 2) ** 0.5
