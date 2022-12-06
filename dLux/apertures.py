import equinox as eqx
import jax.numpy as np
import jax 
import dLux
import abc
import functools
from typing import TypeVar


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront
Aperture = TypeVar("Aperture")


__all__ = ["Aperture", "CompoundAperture", "SquareAperture", 
    "RectangularAperture", "CircularAperture", "AnnularAperture",
    "MultiAperture", "RotatableAperture", "HexagonalAperture"]


class ApertureLayer(dLux.optics.OpticalLayer, abc.ABC):
    """
    """


class AbstractDynamicAperture(ApertureLayer, abc.ABC):
    """
    """
    # NOTE: Is this where the x_offset and the y_offset belong?
    # Well let me imagine that it is thus, then we encounter
    # problems with the AberratedAperture. Hmmm,


class StaticAperture(StaticAperture, abc.ABC):
    """
    """


class DynamicAperture(DynamicAperture, abc.ABC):
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
    

    def __init__(self   : Aperture, 
            x_offset    : float, 
            y_offset    : float, 
            occulting   : bool, 
            softening   : bool) -> Aperture:
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
        self.softening = 1. if softening else np.inf
        self.occulting = bool(occulting)


    def get_centre(self: Aperture) -> Array:
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


    def _soften(self: Aperture, distances: Array) -> Array:
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
    def _metric(self: Aperture, distances: Array) -> Array:
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


    def _aperture(self: Aperture, coordinates: Array) -> Array:
        """
        Compute the array representing the aperture. 


        """
        aperture = self._metric(coordinates)
        if self.occulting:
            aperture = (1 - aperture)

        return aperture


    def __call__(self: Aperture, wavefront: Wavefront) -> Wavefront:
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
        coords = wavefront.pixel_coordinates()
        aperture = self._aperture(coords)
        return wavefront.multiply_amplitude(aperture)


    @abc.abstractmethod
    def largest_extent(self) -> float:
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


    def compute_aperture_normalised_coordinates(self: Aperture, 
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
        return self._translate(coordinates) / self.largest_extent()


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


    def __init__(self   : Aperture, 
            x_offset    : float,  
            y_offset    : float, 
            rmax        : float, 
            rmin        : float, 
            occulting   : bool, 
            softening   : bool) -> Aperture:
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


    def _metric(self: Aperture, coordinates: Array) -> Array:
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


    def largest_extent(self: Aperture) -> float:
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
   
 
    def __init__(self   : Aperture, 
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


    def largest_extent(self: Aperture) -> float:
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


    def __init__(self   : Aperture, 
            x_offset    : float, 
            y_offset    : float, 
            theta       : float, 
            occulting   : bool, 
            softening   : bool) -> Aperture:
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


    def _rotate(self: Aperture, coordinates: Array) -> Array:
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


    def compute_aperture_normalised_coordinates(self: Aperture, 
            coords : Array) -> Array:
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
        return self._rotate(super().compute_aperture_normalised_coordinates(coords))


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


    def __init__(self   : Aperture, 
            x_offset    : float, 
            y_offset    : float,
            theta       : float, 
            length      : float, 
            width       : float, 
            occulting   : bool, 
            softening   : bool) -> Aperture: 
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


    def _metric(self: Aperture, coordinates: Array) -> Array:
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


    def largest_extent(self) -> float:
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
        return np.hypot(self.length / 2., self.width / 2.)


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
   
 
    def __init__(self   : Aperture, 
            x_offset    : float, 
            y_offset    : float,
            theta       : float, 
            width       : float, 
            occulting   : bool, 
            softening   : bool) -> Aperture:
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


    def _metric(self: Aperture, coordinates: Array) -> Array:
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


    def largest_extent(self) -> float:
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


    def __init__(self   : Aperture, 
            x_offset    : float, 
            y_offset    : float, 
            theta       : float, 
            rmax        : float,
            softening   : bool,
            occulting   : bool) -> Aperture:
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


    def get_rmax(self: Aperture) -> float:
        """
        Returns
        -------
        max_radius : float, meters
            The distance from the centre of the hexagon to one of 
            the vertices.
        """
        return self.rmax


    def largest_extent(self: Aperture) -> float:
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


    def _metric(self: Aperture, coords: Array) -> Array:
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
        rmax: float = np.sqrt(3.) / 2. * self.rmax

        m: Array = (-1. / np.tan(theta)).reshape((6, 1, 1))
        
        x1: Array = (rmax * np.cos(theta)).reshape((6, 1, 1))
        y1: Array = (rmax * np.sin(theta)).reshape((6, 1, 1))
        
        x: Array = np.tile(coords[0], (6, 1, 1))
        y: Array = np.tile(coords[1], (6, 1, 1))
        
        dist: Array = (y - y1 - m * (x - x1)) / np.sqrt(1 + m ** 2)
        dist: Array = (1. - 2. * (theta <= np.pi)) * dist
        lines: Array = self._soften(dist)

        return lines.prod(axis=0)


class CompositeAperture(eqx.Module, abc.ABC):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name.

    Parameters:
    -----------
    apertures: dict(str, Aperture)
        The apertures that make up the compound aperture. 
    """
    apertures: dict


    def __init__(self: Aperture, apertures: dict) -> Aperture:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Aperture} where the Aperture is a subclass of the 
            Aperture.
        """
        self.apertures = apertures


    def __getitem__(self: Aperture, key: str) -> Aperture:
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
        layer: Aperture
            The layer that was stored under the name `key`. 
        """
        return self.apertures[key]


    def __setitem__(self, key: str, value: Aperture) -> None:
        """
        Assign a new value to one of the aperture mirrors.
        Parameters
        ----------
        key : str
            The name of the segement to replace for example "B1-7".
        value : Aperture
            The new value to assign to that segement.
        """
        self.apertures[key] = value


    def __call__(self, wavefront: Wavefront) -> Wavefront:
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
        wavefront = wavefront.multiply_amplitude(
            self._aperture(
                wavefront.pixel_coordinates()))
        return parameters


    @abc.abstractmethod
    def _aperture(self: Aperture, coordinates: Array) -> Array:
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
    apertures: dict(str, Aperture)
        The apertures that make up the compound aperture. 
    """


    def __init__(self: Aperture, apertures: dict) -> Aperture:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Aperture} where the Aperture is a subclass of the 
            Aperture.
        """
        super().__init__(apertures)


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


class MultiAperture(CompositeAperture):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `MultiAperture` is used to represent apertures that are 
    not overlapping. We can add `CompoundAperture`s into 
    `MultiAperture` to create a combination of the two affects.

    Attributes
    ----------
    apertures : dict(str, Aperture)
        The apertures that make up the compound aperture. 
    """


    def __init__(self: Aperture, apertures: dict) -> Aperture:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Aperture} where the Aperture is a subclass of the 
            Aperture.
        """
        super().__init__(apertures)


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
            for ap in self.apertures.values()]).sum(axis=0)


    def to_list(self: Aperture) -> list:
        """
        Returns:
        --------
        layers: list
            A list of `Aperture` objects that comprise the 
            `MultiAperture`.
        """
        return list(self.apertures.values())


class PolygonalAperture(RotatableAperture):
    """
    A general representation of a pefect polygonal aperture. 
    Each side of the aperture should be the same length. There
    are some pre-existing implementations for some of the more 
    common cases. This is designed for the exceptions that are 
    less common. 

    Parameters:
    -----------
    nsides: Int
        The number of sides.
    rmax: Float
        The radius of the smallest circle that can fully contain the 
        aperture. 
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
    nsides: Int
    rmax: Float


    def __init__(
            self        : Layer,
            x_offset    : Float,
            y_offset    : Float,
            theta       : Float,
            rmax        : Float,
            nsides      : Int,
            occulting   : bool,
            softening   : bool) -> Layer:
        """
        """
        self.rmax = np.asarray(rmax).astype(float)
        self.nsides = int(nsides)
        super().__init__(x_offset, y_offset, theta, occulting, softening)


    def _perp_dist_from_line(
            self    : Layer, 
            point   : Array, 
            grad    : Float, 
            coords  : Array) -> Array:
        """
        """
        x, y = coords[0], coords[1]
        x1, y1 = point[0], point[1]
        return (y - y1 - grad * (x - x1)) / np.sqrt(1 + grad ** 2)


    def _grad_from_two_points(
            self    : Layer,
            point_1 : Array,
            point_2 : Array)-> Array:
        """
        """
        x1, y1 = point_1[0], point_1[1]
        x2, y2 = point_2[0], point_2[1]
        return (y2 - y1) / (x2 - x1)


    def _aperture(self: Layer, coords: Array) -> Array:
        """
        """
        coords: Array = self._rotate(self._translate(coords))
        theta: Array = np.linspace(0., 2. * np.pi, self.nsides, endpoint=False)

        m: Array = (-1. / np.tan(theta)).reshape((self.nsides, 1, 1))
        x1: Array = (rmax * np.cos(theta)).reshape((6, 1, 1))
        y1: Array = (rmax * np.sin(theta)).reshape((6, 1, 1))

        dist: Array = (y - y1 - m * (x - x1)) / np.sqrt(1 + m ** 2)
        dist: Array = (1. - 2. * (theta <= np.pi)) * dist
        

#class PolygonalAperture(Aperture):
#    """
#    """
#    x : float
#    y : float
#    alpha : float
#
#
#    def __init__(self : Layer, pixels : int, pixel_scale : float, 
#            vertices : Matrix, theta : float = 0., phi : float = 0., 
#            magnification : float = 1.) -> Layer:
#        """
#        Parameters
#        ----------
#        pixels : int
#            The number of pixels that the entire compound aperture
#            is to be generated over. 
#        pixel_scale : float
#    
#        vertices : float
#    
#        theta : float, radians
#            The angle that the aperture is rotated from the positive 
#            x-axis. By default the horizontal sides are parallel to 
#            the x-axis.
#        phi : float, radians
#            The angle that the y-axis is rotated away from the 
#            vertical. This results in a sheer. 
#        magnification : float
#            A factor by which to enlarge or shrink the aperture. 
#            Should only be very small amounts in typical use cases.
#        """
#        x, y, alpha = self._vertices(vertices)
#        x_offset, y_offset = self._offset(vertices)
#        self.x = np.asarray(x).astype(float)
#        self.y = np.asarray(y).astype(float)
#        self.alpha = np.asarray(alpha).astype(float)
#        super().__init__(pixels, x_offset, y_offset, theta, phi,
#            magnification, pixel_scale)
#
#
#    def _wrap(self : Layer, array : Vector, order : Vector) -> tuple:
#        """
#        Re-order an array and duplicate the first element as an additional
#        final element. Satisfies the postcondition `wrapped.shape[0] ==
#        array.shape[0] + 1`. This is just a helper method to simplify 
#        external object and is not physically important (Only invoke 
#        this method if you know what you are doing)
#
#        Parameters
#        ----------
#        array : Vector
#            The 1-dimensional vector to sort and append to. Must be one 
#            dimesnional else unexpected behaviour can occur.
#        order : Vector
#            The new order for the elements of `array`. Will be accessed 
#            by invoking `array.at[order]` hence `order` must be `int`
#            `dtype`.
#
#        Returns
#        -------
#        wrapped : Vector
#            `array` with `array[0]` appended to the end. The dimensions
#            of `array` are also expanded twofold so that the final
#            shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
#            This is just for the vectorisation demanded later in the 
#            code.
#        """
#        _array = np.zeros((array.shape[0] + 1,))\
#            .at[:-1]\
#            .set(array.at[order].get())\
#            .reshape(-1, 1, 1)
#        return _array.at[-1].set(_array[0])
#        
#
#    def _vertices(self : Layer, vertices : Matrix) -> tuple:
#        """
#        Generates the vertices that are compatible with the rest of 
#        the transformations from the raw data vertices.
#
#        Parameters
#        ----------
#        vertices : Matrix, meters
#            The vertices loaded from the WebbPSF module. 
#
#        Returns
#        -------
#        x, y, angles : tuple 
#            The vertices in normalised positions and wrapped so that 
#            they can be used in the generation of the compound aperture.
#            The `x` is the x coordinates of the vertices, the `y` is the 
#            the y coordinates and `angles` is the angle of the vertex. 
#        """
#        _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
#        _y = (vertices[:, 1] - np.mean(vertices[:, 1]))
#
#        _angles = np.arctan2(_y, _x)
#        _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)
#
#        # By default the `np.arctan2` function returns values within the 
#        # range `(-np.pi, np.pi)` but comparisons are easiest over the 
#        # range `(0, 2 * np.pi)`. This is where the logic implemented 
#        # above comes from. 
#
#        order = np.argsort(_angles)
#
#        x = self._wrap(_x, order)
#        y = self._wrap(_y, order)
#        angles = self._wrap(_angles, order).at[-1].add(2 * np.pi)
#
#        # The final `2 * np.pi` is designed to make sure that the wrap
#        # of the first angle is within the angular coordinate system 
#        # associated with the aperture. By convention this is the
#        # range `angle[0], angle[0] + 2 * np.pi` what this means in 
#        # practice is that the first vertex appearing in the array 
#        # is used to chose the coordinate system in angular units. 
#
#        return x, y, angles
#
#
#    def _offset(self : Layer, vertices : Matrix) -> tuple:
#        """
#        Get the offsets of the coordinate system.
#
#        Parameters
#        ----------
#        vertices : Matrix 
#            The unprocessed vertices loaded from the JWST data file.
#            The correct shape for this array is `vertices.shape == 
#            (2, number_of_vertices)`. 
#        pixel_scale : float, meters
#            The physical size of each pixel along one of its edges.
#
#        Returns 
#        -------
#        x_offset, y_offset : float, meters
#            The x and y offsets in physical units. 
#        """
#        x_offset = np.mean(vertices[:, 0])
#        y_offset = np.mean(vertices[:, 1])
#        return x_offset, y_offset
#
#
#    # TODO: number_of_pixels can be moved out as a parameter
#    def _rad_coordinates(self : Layer) -> tuple:
#        """
#        Generates the vectorised coordinate system associated with the 
#        aperture.
#
#        Parameters
#        ----------
#        phi_naught : float 
#            The angle substending the first vertex. 
#
#        Returns 
#        -------
#        rho, theta : tuple[Tensor]
#            The stacked coordinate systems that are typically passed to 
#            `_segments` to generate the segments.
#        """
#        cartesian = self._coordinates()
#        positions = cartesian_to_polar(cartesian)
#
#        # rho = positions[0] * self.get_pixel_scale()
#        rho = positions[0]        
#
#        theta = positions[1] 
#        theta += 2 * np.pi * (positions[1] < 0.)
#        theta += 2 * np.pi * (theta < self.alpha[0])
#
#        rho = np.tile(rho, (6, 1, 1))
#        theta = np.tile(theta, (6, 1, 1))
#        return rho, theta
#
#
#    def _edges(self : Layer, rho : Tensor, theta : Tensor) -> Tensor:
#        """
#        Generate lines connecting adjacent vertices.
#
#        Parameters
#        ----------
#        rho : Tensor, meters
#            Represents the radial distance of every point from the 
#            centre of __this__ aperture. 
#        theta : Tensor, Radians
#            The angle associated with every point in the final bitmap.
#
#        Returns
#        -------
#        edges : Tensor
#            The edges represented as a Bitmap with the points inside the 
#            edge marked as 1. and outside 0. The leading axis contains 
#            each unique edge and the corresponding matrix is the bitmap.
#        """
#        # This is derived from the two point form of the equation for 
#        # a straight line (eq. 1)
#        # 
#        #           y_2 - y_1
#        # y - y_1 = ---------(x - x_1)
#        #           x_2 - x_1
#        # 
#        # This is rearranged to the form, ay - bx = c, where:
#        # - a = (x_2 - x_1)
#        # - b = (y_2 - y_1)
#        # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
#        # we can then drive the transformation to polar coordinates in 
#        # the usual way through the substitutions; y = r sin(theta), and 
#        # x = r cos(theta). The equation is now in the form 
#        #
#        #                  c
#        # r = ---------------------------
#        #     a sin(theta) - b cos(theta) 
#        #
#        a = (self.x[1:] - self.x[:-1])
#        b = (self.y[1:] - self.y[:-1])
#        c = (a * self.y[:-1] - b * self.x[:-1])
#
#        linear = c / (a * np.sin(theta) - b * np.cos(theta))
#        #return rho < (linear * self.get_pixel_scale())
#        return rho < linear
#
#
#    def _wedges(self : Layer, theta : Tensor) -> Tensor:
#        """
#        The angular bounds of each segment of an individual hexagon.
#
#        Parameters
#        ----------
#        theta : Tensor, Radians
#            The angle away from the positive x-axis of the coordinate
#            system associated with this aperture. Please note that `theta`
#            May not start at zero. 
#
#        Returns 
#        -------
#        wedges : Tensor 
#            The angular bounds associated with each pair of vertices in 
#            order. The leading axis of the Tensor steps through the 
#            wedges in order arround the circle. 
#        """
#        return (self.alpha[:-1] < theta) & (theta < self.alpha[1:])
#
#
#    def _segments(self : Layer, theta : Tensor, rho : Tensor) -> Tensor:
#        """
#        Generate the segments as a stacked tensor. 
#
#        Parameters
#        ----------
#        theta : Tensor
#            The angle of every pixel associated with the coordinate system 
#            of this aperture. 
#        rho : Tensor
#            The radial positions associated with the coordinate system 
#            of this aperture. 
#
#        Returns 
#        -------
#        segments : Tensor 
#            The bitmaps corresponding to each vertex pair in the vertices.
#            The leading dimension contains the unique segments. 
#        """
#        edges = self._edges(rho, theta)
#        wedges = self._wedges(theta)
#        return (edges & wedges).astype(float)
#        
#
#    def _aperture(self : Layer) -> Matrix:
#        """
#        Generate the BitMap representing the aperture described by the 
#        vertices. 
#
#        Returns
#        -------
#        aperture : Matrix 
#            The Bit-Map that represents the aperture. 
#        """
#        rho, theta = self._rad_coordinates()
#        segments = self._segments(theta, rho)
#        return segments.sum(axis=0)
#
