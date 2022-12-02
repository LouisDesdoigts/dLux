import equinox as eqx
import jax.numpy as np
import jax 
import dLux
import abc
import functools
from typing import TypeVar


Array = TypeVar("Array")
Layer = TypeVar("Layer")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")


__all__ = ["Aperture", "CompoundAperture", "SquareAperture", 
    "RectangularAperture", "CircularAperture", "AnnularAperture",
    "MultiAperture", "RotatableAperture", "HexagonalAperture"]


class Aperture(eqx.Module, abc.ABC):
    """
    An abstract class that defines the structure of all the concrete
    apertures. An aperture is represented by an array, usually in the
    range of 0. to 1.. Values in between can be used to represent 
    soft edged apertures and intermediate surfaces. 

    Attributes
    ----------
    x_offset : float, meters
        The x coordinate of the centre of the aperture.
    y_offset : float, meters
        The y coordinate of the centre of the aperture.
    occulting: bool
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside.
        A non-occulting aperture is one inside and zero 
        outside. 
    softening: bool 
        True is the aperture is soft edged. This means that 
        there is a layer of pixels that is non-binary. The 
        way that this is implemented (due to the limitations)
        of `jax` is via a `np.tanh` function. This is good for 
        derivatives. Use this feature only if encountering 
        errors when using hard edged apertures. 
    """
    occulting: bool 
    softening: float
    x_offset: float
    y_offset: float
    

    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float, 
            occulting   : bool, 
            softening   : bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        """
        self.x_offset = np.asarray(x_offset).astype(float)
        self.y_offset = np.asarray(y_offset).astype(float)
        self.softening = jax.lax.cond(softening, lambda: 1., lambda: np.inf)
        self.occulting = bool(occulting)


    def get_centre(self: Layer) -> Array:
        """
        Returns 
        -------
        x, y : tuple(float, float) meters
            The coodinates of the centre of the aperture. The first 
            element of the tuple is the x coordinate and the second 
            is the y coordinate.
        """
        return np.array([self.x_offset, self.y_offset])


    def _translate(self, coordinates: Array) -> Array:
        """
        Move the center of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters 
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        coordinates: Array, meters
            The translated coordinate system. 
        """
        return coordinates - self.get_centre().reshape((2, 1, 1))


    def _soften(self: Layer, distances: Array) -> Array:
        """
        Softens an image so that the hard boundaries are not present. 

        Parameters
        ----------
        image: Array, meters
            The name I gave this is a misnomer. The image should be an 
            array representing distances from a particular point or line. 
            Typically it is easiest to apply this to each edge separately 
            and then multiply the result. This has the added benifit of 
            curving points slightly. 

        Returns
        -------
        smooth_image: Array
            The image represented as an approximately binary mask, but with 
            the prozed soft edges.
        """
        steepness = self.softening * distances.shape[-1]
        return (np.tanh(steepness * distances) + 1.) / 2.


    @abc.abstractmethod
    def _metric(self: Layer, distances: Array) -> Array:
        """
        A measure of how far a pixel is from the aperture.
        This is a very abstract description that was constructed 
        when dealing with the soft edging. For a normal binary 
        representation the metric is zero if it is inside the
        aperture and one if it is outside the aperture. Notice,
        we have not attempted to prove that this is a metric 
        via the axioms, this is just a handy name that brings 
        to mind the general idea. For a soft edged aperture the 
        metric is different.

        Parameters:
        -----------
        distances: Array
            The distances of each pixel from the edge of the aperture. 
            Again, the words distances is designed to aid in 
            conveying the idea and is not strictly true. We are
            permitting negative distances when inside the aperture
            because this was simplest to implement. 

        Returns:
        --------
        non_occ_ap: Array 
            This is essential the final step in processing to produce
            the aperture. What is returned is the non-occulting 
            version of the aperture. 
        """


    def _aperture(self: Layer, coordinates: Array) -> Array:
        """
        Compute the array representing the aperture. 


        """
        aperture = jax.lax.cond(self.occulting,
            lambda aperture: 1 - aperture,
            lambda aperture: aperture,
            self._metric(coordinates))

        return aperture


    def set_x_offset(self, x : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The x coordinate of the centre of the hexagonal
            aperture.
        """
        return eqx.tree_at(lambda basis : basis.x_offset, 
            self, np.asarray(x).astype(float))


    def get_x_offset(self: Layer) -> float:
        """
        Returns:
        --------
        x_offset: float, meters
            The x centre of the aperture relative to the optical
            axis. 
        """
        return self.x_offset


    def set_y_offset(self: Layer, y : float) -> Layer:
        """
        Parameters
        ----------
        y: float
            The y coordinate of the centre of the hexagonal
            aperture.
        """
        return eqx.tree_at(lambda basis : basis.y_offset, 
            self, np.asarray(y).astype(float))


    def get_y_offset(self: Layer) -> float:
        """
        Returns:
        --------
        y_offset: float, meters
            The y centre of the aperture relative to the optical
            axis. 
        """
        return self.y_offset


    def __call__(self: Layer, parameters : dict) -> dict:
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
        wavefront = wavefront.multiply_amplitude(
            self._aperture(
                wavefront.pixel_coordinates()))
        parameters["Wavefront"] = wavefront
        return parameters


    @abc.abstractmethod
    def largest_extent(self, coordinates : Array) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

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


    def compute_aperture_normalised_coordinates(self: Layer, 
            coordinates : Array) -> Array:
        """
        Shift a set of wavefront coodinates to be centered on the 
        aperture and scaled such that the radial distance is 1 to 
        the edge of the aperture, returned in polar form

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to generate the aperture on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  
        
        Returns
        -------
        coordinates : Array
            the radial coordinates centered on the centre of the aperture 
            and scaled such that they are 1
            at the maximum extent of the aperture
            The dimensions of the tensor are be `(2, npix, npix)`
        """
        # TODO: check where flips should go
        coordinates = coordinates.at[1].set(coordinates[1][::-1,:])

        x_offset = self.get_x_offset()
        y_offset = self.get_y_offset()

        # This is the translation and scaling of the normalised coordinate system. 
        # translate and then multiply by 1 / largest_extent.
        trans_coords = self._translate(coordinates)
        rad_trans_coords = dLux.utils.cartesian_to_polar(trans_coords)
        coordinates = rad_trans_coords.at[0].mul(1. / self.largest_extent(coordinates))

        return coordinates


class AnnularAperture(Aperture):
    """
    A circular aperture, parametrised by the number of pixels in
    the array. By default this is a hard edged aperture but may be 
    in future modifed to provide soft edges. 

    Attributes
    ----------
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    rmax : float
        The proportion of the pixel vector that is contained within
        the outer ring of the aperture.
    rmin : float
        The proportion of the pixel vector that is contained within
        the inner ring of the aperture. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    """
    rmin : float
    rmax : float


    def __init__(self   : Layer, 
            x_offset    : float,  
            y_offset    : float, 
            rmax        : float, 
            rmin        : float, 
            occulting   : bool, 
            softening   : bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        rmax : float, meters
            The outer radius of the annular aperture. 
        rmin : float, meters
            The inner radius of the annular aperture. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        """
        super().__init__(x_offset, y_offset, occulting, softening)
        self.rmax = np.asarray(rmax).astype(float)
        self.rmin = np.asarray(rmin).astype(float)


    def _metric(self: Layer, coordinates: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cartesian_to_polar(coordinates)[0]
        return self._soften(coordinates - self.rmin) * \
            self._soften(- coordinates + self.rmax)


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
      

class CircularAperture(Aperture):
    """
    A circular aperture represented as a binary array.

    Parameters
    ----------
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    radius: float, meters
        The radius of the opening. 
    """
    radius: float
   
 
    def __init__(self   : Layer, 
            x_offset    : float,
            y_offset    : float,
            radius      : float, 
            occulting   : bool, 
            softening   : bool) -> Array:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        radius: float, meters 
            The radius of the aperture.
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        """
        super().__init__(x_offset, y_offset, occulting, softening)
        self.radius = np.asarray(radius).astype(float)


    def _metric(self, coordinates: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cartesian_to_polar(coordinates)[0]
        return self._soften(- coordinates + self.radius)


    def largest_extent(self: Layer, coordinates : Array) -> float:
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
        return self.radius


class RotatableAperture(Aperture):
    """
    An abstract class that is used to represent an aperture 
    that does not have radial symmetry. This class can be 
    used to learn rotation of the apertures. 

    Parameters:
    -----------
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    theta: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    theta: float    


    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float, 
            theta       : float, 
            occulting   : bool, 
            softening   : bool) -> Layer:
        """
        Parameters:
        -----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        theta: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(x_offset, y_offset, occulting, softening)
        self.theta = np.asarray(theta).astype(float)


    def _rotate(self: Layer, coordinates: Array) -> Array:
        """
        Rotate the coordinate system by a pre-specified amount,
        `self._theta`

        Parameters
        ----------
        coordinates : Array
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Array
            The rotated coordinate system. 
        """
        x_coordinates, y_coordinates = coordinates[0], coordinates[1]
        new_x_coordinates = np.cos(self.theta) * x_coordinates + \
            np.sin(self.theta) * y_coordinates
        new_y_coordinates = -np.sin(self.theta) * x_coordinates + \
            np.cos(self.theta) * y_coordinates
        return np.array([new_x_coordinates, new_y_coordinates])


class RectangularAperture(RotatableAperture):
    """
    A rectangular aperture.

    Parameters
    ----------
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    theta: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    length: float, meters
        The length of the aperture in the y-direction. 
    width: float, meters
        The length of the aperture in the x-direction. 
    """
    length: float
    width: float


    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float,
            theta       : float, 
            length      : float, 
            width       : float, 
            occulting   : bool, 
            softening   : bool) -> Layer: 
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
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        length: float, meters 
            The length of the aperture in the y-direction.
        width: float, meters
            The length of the aperture in the x-direction.
        """
        super().__init__(x_offset, y_offset, theta, occulting, softening)
        self.length = np.asarray(length).astype(float)
        self.width = np.asarray(width).astype(float)


    def _metric(self: Layer, coordinates: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        coordinates = self._rotate(self._translate(coordinates))  
        x_mask = self._soften(- np.abs(coordinates[0]) + self.length / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


    def largest_extent(self, coordinates: Array) -> float:
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
        return np.hypot(np.array([self.length / 2., self.width / 2.]))


class SquareAperture(RotatableAperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this one tracks less parameters.

    Parameters
    ----------
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    theta: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    width: float, meters
        The side length of the square. 
    """
    width: float
   
 
    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float,
            theta       : float, 
            width       : float, 
            occulting   : bool, 
            softening   : bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        width: float, meters
            The side length of the square. 
        """
        super().__init__(x_offset, y_offset, theta, occulting, softening)
        self.width = np.asarray(width).astype(float)


    def _metric(self: Layer, coordinates: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        coordinates = self._rotate(self._translate(coordinates))
        x_mask = self._soften(- np.abs(coordinates[0]) + self.width / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


    def largest_extent(self, coordinates: Array) -> float:
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
        return np.sqrt(2) * self.width / 2.


# NOTE: This is not yet ready for deployment. The _metric method needs 
# to be re-written. 
class HexagonalAperture(RotatableAperture):
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
        theta: Array = np.linspace(0, 2 * np.pi, 6, endpoint=False).reshape((6, 1, 1))
        rmax: float = self.rmax

        m1: Array = np.tan(theta).reshape((6, 1, 1))
        m2: Array = (-1. / np.tan(theta)).reshape((6, 1, 1))

        # (x1, y1) is in the centre of the segment, whereas the 
        # (x2, y2) is some point in the coordinate system and 
        # (x3, y3) is the closest point on the edge to (x2, y2).
        # See the following figure,
        #        _
        #     _-- |
        #  _--    | (x1, y1)
        # <_------o 
        #   --_   | (x3, y3)
        #      --_o- - - - - -o (x2, y2)
        
        x1: Array = (rmax * np.cos(theta)).reshape((6, 1, 1))
        y1: Array = (rmax * np.sin(theta)).reshape((6, 1, 1))
        
        x2: Array = np.tile(coords[0], (6, 1, 1))
        y2: Array = np.tile(coords[1], (6, 1, 1))

        x3: Array = (y2 + m2 * x2 - y1 - m1 * x1) / (m2 - m1)
        y3: Array = m1 * (x3 - x1) + y1

        dist: Array = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        return self._soften(dist).prod(axis=0)


class CompositeAperture(eqx.Module, abc.ABC):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name.

    Parameters:
    -----------
    apertures: dict(str, Layer)
        The apertures that make up the compound aperture. 
    """
    apertures: dict

    def __init__(self: Layer, apertures: dict) -> Layer:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        """
        self.apertures = apertures


    def __getitem__(self: Layer, key: str) -> Layer:
        """
        Get one of the apertures from the collection using a name 
        based lookup.
        
        Parameters:
        -----------
        key: str
            The name of the aperture to lookup. See the class doc
            string for more information.
        
        Returns:
        --------
        layer: Layer
            The layer that was stored under the name `key`. 
        """
        return self.apertures[key]


    def __setitem__(self, key: str, value: Layer) -> None:
        """
        Assign a new value to one of the aperture mirrors.
        Parameters
        ----------
        key : str
            The name of the segement to replace for example "B1-7".
        value : Layer
            The new value to assign to that segement.
        """
        self.apertures[key] = value


    def __call__(self, parameters: dict) -> dict:
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
        wavefront = wavefront.multiply_amplitude(
            self._aperture(
                wavefront.pixel_coordinates()))
        parameters["Wavefront"] = wavefront
        return parameters


    @abc.abstractmethod
    def _aperture(self: Layer, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
            An aperture generated by combining all of the sub 
            apertures that were stored. 
        """


class CompoundAperture(eqx.Module):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `CompoundAperture` contains overlapping apertures that 
    may or may not be occulting. The goal is mainly to represent
    `AnnularAperture`s that have `UniformSpider`s embedded. This
    class should not be used to represent multiple apertures 
    that are not connected. Doing so will result in a zero 
    output.

    Parameters:
    -----------
    apertures: dict(str, Layer)
        The apertures that make up the compound aperture. 
    """


    def __init__(self: Layer, apertures: dict) -> Layer:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        """
        super.__init__(apertures)


    def _aperture(self, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
            An aperture generated by combining all of the sub 
            apertures that were stored. 
        """
        return np.stack([ap._aperture(coordinates) 
            for ap in self._apertures.values()]).prod(axis=0)


class MultiAperture(eqx.Module):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `MultiAperture` is used to represent apertures that are 
    not overlapping. We can add `CompoundAperture`s into 
    `MultiAperture` to create a combination of the two affects.

    Attributes
    ----------
    apertures : dict(str, Layer)
        The apertures that make up the compound aperture. 
    """


    def __init__(self: Layer, apertures: dict) -> Layer:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        """
        super.__init__(apertures)


    def _aperture(self, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
            An aperture generated by combining all of the sub 
            apertures that were stored. 
        """
        return np.stack([ap._aperture(coordinates) 
            for ap in self._apertures.values()]).sum(axis=0)


