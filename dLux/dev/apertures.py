import dLux as dl 
import jax.numpy as np
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import typing 

Layer = typing.TypeVar("Layer")
Array = typing.TypeVar("Array")

class HexagonalAperture(dl.RotatableAperture):
    """
    Generate a hexagonal aperture, parametrised by rmax. 
    
    Attributes
    ----------
    rmax : float, meters
        The infimum of the radii of the set of circles that fully 
        enclose the hexagonal aperture. In other words the distance 
        from the centre to one of the vertices. 
    """
    rmax : float


    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float, 
            theta       : float, 
            rmax        : float,
            softening   : bool,
            occulting   : bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        rmax : float, meters
            The distance from the center of the hexagon to one of
            the vertices. . 
        softening: bool
            True if the aperture is soft edged else False.
        occulting: bool
            True is the aperture is occulting else False. An occulting 
            Aperture is zero inside and one outside. 
        """
        super().__init__(x_offset, y_offset, theta, softening, occulting)
        self.rmax = np.asarray(rmax).astype(float)


    def get_rmax(self: Layer) -> float:
        """
        Returns
        -------
        max_radius : float, meters
            The distance from the centre of the hexagon to one of 
            the vertices.
        """
        return self.rmax


    def largest_extent(self: Layer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        largest_extent : float
            The maximum distance from centre to edge of aperture
        """
        return self.rmax


    def _metric(self: Layer, coords: Array) -> Array:
        """
        Generates an array representing the hard edged hexagonal 
        aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The coordinates over which to generate the aperture. 

        Returns
        -------
        aperture : Array
            The aperture represented as a binary float array of 0. and
            1. representing no transmission and transmission 
            respectively.
        """
        # So the challenge is how to make this soft edgeable. 
        # Well, I know the formula for a line. I could just do 
        # six lines that are perpendicular to the lines 
        # along multiples of pi on three.   
        coords: Array = self._rotate(self._translate(coords))
        theta: Array = np.linspace(0, 2 * np.pi, 6, endpoint=False).reshape((6, 1, 1)) + np.pi / 6.
        rmax: float = self.rmax

        m: Array = (-1. / np.tan(theta)).reshape((6, 1, 1))
        
        x1: Array = (rmax * np.cos(theta)).reshape((6, 1, 1))
        y1: Array = (rmax * np.sin(theta)).reshape((6, 1, 1))
        
        x: Array = np.tile(coords[0], (6, 1, 1))
        y: Array = np.tile(coords[1], (6, 1, 1))
        
        dist: Array = (y - y1 - m * (x - x1)) / np.sqrt(1 + m ** 2)
        dist: Array = (1. - 2. * (theta <= np.pi)) * dist
        lines: Array = self._soften(dist)

        return lines.prod(axis=0)

coords = dl.utils.get_pixel_coordinates(128, 2. / 128.)
hex_ap = HexagonalAperture(0., 0., 0., .5, False, False)

plt.imshow(hex_ap._aperture(coords))
plt.colorbar()
plt.show()
