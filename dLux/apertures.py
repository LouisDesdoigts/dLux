from typing import TypeVar
import matplotlib.pyplot as pyplot 
import equinox as eqx
import jax.numpy as np
import jax 
import dLux
import abc
import functools

Array = TypeVar("Array")
Layer = TypeVar("Layer")
Tensor = TypeVar("Tensor")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")


__all__ = ["Aperture", "CompoundAperture", "SoftEdgedAperture", 
    "SquareAperture", "SoftEdgedSquareAperture", "RectangularAperture",
    "SoftEdgedRectangularAperture", "CircularAperture", 
    "SoftEdgedCircularAperture", "AnnularAperture", 
    "SoftEdgedAnnularAperture"]


def _distance_from_line(self, gradient: float, intercept: float) -> Matrix:
    x, y = self.coordinates[0], self.coordinates[1]
    return np.abs(y - gradient * x - intercept) / np.sqrt(1 + gradient ** 2)


def _distance_from_circle(self, radius: float, centre: Vector) -> Matrix:
    translated_coordinates = self.coordinates + centre.reshape((2, 1, 1))
    radial_coordinates = dLux.polar2cart(translated_coordinates)[0]
    return radial_coordinates - radius


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
    theta : float, radians
        The angle of rotation from the positive x-axis. 
    phi : float, radians
        The rotation of the y-axis away from the vertical and torward 
        the negative x-axis. 
    """
    occulting: bool 
    softening: bool
    x_offset : float
    y_offset : float
    

    def __init__(self, x_offset : float, y_offset : float, 
            occulting: bool, softening: bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        """
        self.x_offset = np.asarray(x_offset).astype(float)
        self.y_offset = np.asarray(y_offset).astype(float)
        self.occulting = bool(occulting)
        self.softening = bool(softening)


    def get_centre(self) -> tuple:
        """
        Returns 
        -------
        x, y : tuple(float, float) meters
            The coodinates of the centre of the aperture. The first 
            element of the tuple is the x coordinate and the second 
            is the y coordinate.
        """
        return np.array([self.x_offset, self.y_offset])


    def _translate(self, coordinates: Tensor) -> Tensor:
        return coordinates - self.get_centre().reshape((2, 1, 1))



    def _soften(self, distances: Array) -> Array:
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
        steepness = distances.shape[0]
        return (np.tanh(steepness * distances) + 1.) / 2.


    @abc.abstractmethod
    def _hardened_metric(self, distances: Array) -> Array:
        """
        """


    @abc.abstractmethod
    def _softened_metric(self, distances: Array) -> Array:
        """
        """


    def _aperture(self, coordinates: Array) -> Array:
        aperture = jax.lax.cond(self.softening,
            self._softened_metric,
            self._hardened_metric,
            coordinates)

        aperture = jax.lax.cond(self.occulting,
            lambda aperture: 1 - aperture,
            lambda aperture: aperture,
            aperture)

        return aperture


    # TODO: Remove the basis naming convention. 
    def set_x_offset(self, x : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The x coordinate of the centre of the hexagonal
            aperture.
        """
        return eqx.tree_at(lambda basis : basis.x_offset, self, x)


    def set_y_offset(self, y : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The y coordinate of the centre of the hexagonal
            aperture.
        """
        return eqx.tree_at(lambda basis : basis.y_offset, self, y)


    def __call__(self, parameters : dict) -> dict:
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
            self._aperture(
                wavefront.get_pixel_coordinates()))
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


    def __init__(self, x_offset : float,  y_offset : float, 
            rmax : float, rmin : float, occulting: bool, 
            softening: bool) -> Layer:
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
        """
        super().__init__(x_offset, y_offset, occulting, softening)
        self.rmax = np.asarray(rmax).astype(float)
        self.rmin = np.asarray(rmin).astype(float)


    def _hardened_metric(self, coordinates: Array) -> Array:
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
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cart2polar(coordinates[0], coordinates[1])[0]
        return ((coordinates <= self.rmax) \
            & (coordinates > self.rmin)).astype(float)


    def _softened_metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cart2polar(coordinates[0], coordinates[1])[0]
        return self._soften(coordinates - self.rmin) * \
            self._soften(- coordinates + self.rmax)
      

class CircularAperture(Aperture):
    """
    A circular aperture represented as a binary array.

    Parameters
    ----------
    radius: float, meters
        The radius of the opening. 
    """
    radius: float
   
 
    def __init__(self, x_offset: float, y_offset: float,
            radius: float, occulting: bool, softening: float) -> Array:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        radius: float, meters 
            The radius of the aperture.
        """
        super().__init__(x_offset, y_offset, occulting, softening)
        self.radius = np.asarray(radius).astype(float)


    def _hardened_metric(self, coordinates: Array) -> Array:
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
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cart2polar(coordinates[0], coordinates[1])[0]
        return (coordinates <= self.radius).astype(float)


    def _softened_metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(coordinates)
        coordinates = dLux.utils.cart2polar(coordinates[0], coordinates[1])[0]
        return self._soften(- coordinates + self.radius)


class RotatableAperture(Aperture):
    theta: float

    def __init__(self, x_offset, y_offset, theta, occulting, softening):
        super().__init__(x_offset, y_offset, occulting, softening)
        self.theta = np.asarray(theta).astype(float)


    def _rotate(self, coordinates: Tensor) -> Tensor:
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
    length: float, meters
        The length of the aperture in the y-direction. 
    width: float, meters
        The length of the aperture in the x-direction. 
    """
    length: float
    width: float


    def __init__(self, x_offset: float, y_offset: float,
            theta: float, length: float, width: float, occulting: bool, 
            softening: bool): 
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
        length: float, meters 
            The length of the aperture in the y-direction.
        width: float, meters
            The length of the aperture in the x-direction.
        """
        super().__init__(x_offset, y_offset, theta, occulting, softening)
        self.length = np.asarray(length).astype(float)
        self.width = np.asarray(width).astype(float)

    def _hardened_metric(self, coordinates: Array) -> Array:
        """
        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._translate(self._rotate(coordinates))
        x_mask = np.abs(coordinates[0]) < (self.length / 2.)
        y_mask = np.abs(coordinates[1]) < (self.width / 2.)    
        return (y_mask * x_mask).astype(float)     


    def _softened_metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(self._rotate(coordinates))  
        x_mask = self._soften(- np.abs(coordinates[0]) + self.length / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


class SquareAperture(RotatableAperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this obe tracks less parameters.

    Parameters
    ----------
    width: float, meters
        The side length of the square. 
    """
    width: float
   
 
    def __init__(self, x_offset: float, y_offset: float,
            theta: float, width: float, occulting: bool, 
            softening: bool):
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
        width: float, meters
            The side length of the square. 
        """
        super().__init__(x_offset, y_offset, theta, occulting, softening)
        self.width = np.asarray(width).astype(float)
    

    def _hardened_metric(self, coordinates: Array) -> Array:
        """
        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._translate(self._rotate(coordinates))
        x_mask = np.abs(coordinates[0]) < (self.width / 2.)
        y_mask = np.abs(coordinates[1]) < (self.width / 2.)
        return (x_mask * y_mask).astype(float)


    def _softened_metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(self._rotate(coordinates))
        x_mask = self._soften(- np.abs(coordinates[0]) + self.width / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


class CompoundAperture(eqx.Module):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. For example:


    Attributes
    ----------
    apertures : dict(str, Layer)
        The apertures that make up the compound aperture. 
    """
    apertures : dict    


    def __init__(self, apertures: dict) -> Layer:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        """
        self.apertures = apertures


    def __getitem__(self, key: str) -> Layer:
        """
        Get one of the apertures from the collection using a name 
        based lookup.
        
        Parameters
        ----------
        key : str
            The name of the aperture to lookup. See the class doc
            string for more information.
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


    def _aperture(self, coordinates: Array) -> Array:
        """
        Returns 
        -------
        aperture : Matrix
            An aperture generated by combining all of the sub 
            apertures that were stored. 
        """
        apertures = []
        for aperture in self.apertures.values():
            apertures.append(aperture._aperture(coordinates))
        return np.stack(apertures).prod(axis=0)


    def __call__(self, params: dict) -> dict:
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
            self._aperture(
                wavefront.get_pixel_coordinates()))
        parameters["Wavefront"] = wavefront
        return parameters


class Spider(equinox.Module, abc.ABC):
    """
    An abstraction on the concept of an optical spider for a space telescope.
    These are the things that hold up the secondary mirrors. For example,

    Parameters
    ----------
    number_of_pixels: int
        The number of pixels along one edge of the image used to represent the 
        spider. 
    radius_of_spider: float, meters
        The physical width of the spider. For the moment it is assumed to be 
        embedded within a circular aperture.         
    width_of_image: float, meters
        The width of the image. If you wish to pad the array representation of
        the spider then set this to the padding factor multiplied by the 
        radius_of_spider parameter. 
    center_of_spicer: Array, meters 
        The [x, y] center of the spider.
    """  
    width_of_image: float
    number_of_pixels: int
    radius_of_spider: float
    centre_of_spider: float


    def __init__(
            self: Layer, 
            width_of_image: float,
            number_of_pixels: int, 
            radius_of_spider: float,
            centre_of_spider: float) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels: int
            The number of pixels along one edge of the image used to represent 
            the spider. 
        radius_of_spider: float, meters
            The physical width of the spider. For the moment it is assumed to be 
            embedded within a circular aperture.         
        width_of_image: float, meters
            The width of the image. If you wish to pad the array representation of
            the spider then set this to the padding factor multiplied by the 
            radius_of_spider parameter. 
        center_of_spicer: Array, meters 
            The [x, y] center of the spider.
        """
        self.number_of_pixels = number_of_pixels
        self.width_of_image = np.asarray(width_of_image).astype(float)
        self.centre_of_spider = np.asarray(centre_of_spider).astype(float)
        self.radius_of_spider = np.asarray(radius_of_spider).astype(float)


    def _coordinates(self: Layer) -> float:
        """
        Generates a coordinate grid representing the positions of the pixels 
        relative to the centre of the spider. The representation that we 
        use is cartesian. 
    
        Returns 
        -------
        coordinates: float, meters
            The pixel coordinates.
        """
        pixel_scale = self.width_of_image / self.number_of_pixels
        pixel_centre = self.centre_of_spider / pixel_scale
        pixel_coordinates = dLux.utils.get_pixel_positions(
            self.number_of_pixels, pixel_centre[0], pixel_centre[1])
        return pixel_coordinates * pixel_scale  
 

    def _rotate(self: Layer, image: float, angle: float) -> float:
        """
        Rotate a set of coordinates by an amount angle. 
    
        Parameters
        ----------
        image: float, meters
            The physical coordinates for the pixel position as generated by 
            self._coordinates(). This should be a tensor with x then y along 
            the leading axis. 
        angle: float, radians
            The amount to rotate the coordinate system by. 

        Returns 
        -------
        coordinates: float, meters
            The rotate physical coordinate system. This will be a tensor with 
            x then y along the leading axis. 
        """
        coordinates = self._coordinates()
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)]])
        return np.apply_along_axis(np.matmul, 0, coordinates, rotation_matrix) 


    def _strut(self: Layer, angle: float, width: float) -> float:
        """
        Generates a representation of a single strut in the spider. This is 
        more complex than you might imagine since the strut can point in 
        any direction. 

        Parameters
        ----------
        angle: float, radians
            The angle that this strut points as measured from the positive 
            x-axis in radians. 
        width: float, meters
            The width of the strut in meters. Note: a large amount of effort 
            is made to make the edge soft so that autodiff does not achieve 
            infinite gradients and that means that in the output the exact 
            edge is not well defined.

        Returns
        -------
        strut: float
            The soft edged strut. 
        """
        coordinates = self._rotate(self._coordinates(), angle)
        distance = np.where(coordinates[0] > 0., np.abs(coordinates[1]), np.inf)
        spider = self._sigmoid(distance, width)
        return spider        

    
    # TODO: 
    def _sigmoid(self: Layer, distance: float, width: float) -> float:
        """
        The name is a misnomer but it has been kept for legacy reasons. 
        This is a routine that is used in the soft edging of the images. 
        
        Parameters
        ----------
        distance: float, meters
            A matrix representing the distance of each pixel from a point 
            line or otherwise a shape. Some useful hints for using this are 
            that setting regions to np.inf will convert them into exactly 1.
            i.e. inside a box. 
        width: float, meters
            This roughly represents the amount of soft edging. The best way to 
            use this parameter is by trying a few different values until you 
            find one that you like.

        Returns
        -------
        image: float 
            The soft edged image, taken from distance. 
        """
        steepness = self.number_of_pixels
        return (np.tanh(steepness * (distance - width)) + 1.) / 2.


    @abc.abstractmethod
    def _spider(self: Layer) -> float:
        """
        Represent the spider as an number_of_pixels by umber_of_pixels array. 

        Returns 
        -------
        spider: float
            The soft edged array representation of the spider. 
        """
        pass 

    
    @abc.abstractmethod
    def __call__(self: Layer, params: dict) -> dict:
        """
        Apply the spider to a wavefront. 

        Parameters
        ----------
        params: dict
            A dictionary of parameters that must contain a "Wavefront" entry.
        
        Returns 
        -------
        params: dict
            The same dictionary of parameter with an updated "Wavefront" entry.
        """
        pass 


class UniformSpider(Spider):
    """
    A spider with equally-spaced, equal-width struts. This is of course the 
    most common and simplest implementation of a spider. Gradients can be 
    taken with respect to the width of the struts and the global rotation 
    as well as the centre of the spider inherited from 

    Parameters
    ----------
    number_of_struts: int 
        The number of struts to equally space around the circle. This is not 
        a differentiable parameter. 
    width_of_struts: float, meters
        The width of each strut. 
    rotation: float, radians
        A global rotation to apply to the entire spider. 
    """
    number_of_struts: int
    width_of_struts: float
    rotation: float


    def __init__(
            self: Layer, 
            width_of_image: float,
            number_of_pixels: int, 
            radius_of_spider: float,
            centre_of_spider: float,
            number_of_struts: int, 
            width_of_struts: float, 
            rotation: float) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels: int
            The number of pixels along one edge of the image used to represent 
            the spider. 
        radius_of_spider: float, meters
            The physical width of the spider. For the moment it is assumed to 
            be embedded within a circular aperture.         
        width_of_image: float, meters
            The width of the image. If you wish to pad the array representation 
            of the spider then set this to the padding factor multiplied by the 
            radius_of_spider parameter. 
        center_of_spicer: Array, meters 
            The [x, y] center of the spider.
        number_of_struts: int 
            The number of struts to equally space around the circle. This is not 
            a differentiable parameter. 
        width_of_struts: float, meters
            The width of each strut. 
        rotation: float, radians
            A global rotation to apply to the entire spider.
        """ 
        super().__init__(
            width_of_image, 
            number_of_pixels, 
            radius_of_spider,
            centre_of_spider)
        self.number_of_struts = number_of_struts
        self.rotation = np.asarray(rotation).astype(float)
        self.width_of_struts = np.asarray(width_of_struts).astype(float)


    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def _strut(self: Layer, angle: float, width: float) -> float:
        """
        A vectorised routine for constructing the struts. This has been done 
        to improve the performance of the program, and simply differs to 
        the super-class implementation. 

        Parameters
        ----------
        angle: float, radians
            The angle that this strut points as measured from the positive 
            x-axis in radians. 
        width: float, meters
            The width of the strut in meters.

        Returns
        -------
        strut: float
            The soft edged strut.
        """
        return super()._strut(angle, width)


    def _spider(self: Layer) -> float:
        """
        Represents the spider in a square array. Each strut is placed equally 
        around the circumference like a wll cut pizza. All the struts have the
        same width and a global rotation.
    
        Returns
        -------
        spider: float
            The array representation of the spider. 
        """
        angles = np.linspace(0, 2 * np.pi, self.number_of_struts, 
            endpoint=False)
        angles += self.rotation

        struts = self._strut(angles, self.width_of_struts)
        spider = np.prod(struts, axis=0)

        coordinates = self._coordinates()
        radial_coordinates = np.hypot(coordinates[0], coordinates[1])

        radial_distance = np.abs(radial_coordinates - self.radius_of_spider)\
            .at[radial_coordinates > self.radius_of_spider]\
            .set(-np.inf)

        soft_edge = self.width_of_image / self.number_of_pixels
        radial_soft_edge = self._sigmoid(radial_distance, soft_edge)

        return radial_soft_edge * spider
        
 
    def __call__(self: Layer, params: dict) -> dict:
        """
        Apply the spider to a wavefront, as it propagates through the spider. 

        Parameters
        ----------
        params: dict
            A dictionary of parameters that contains a "Wavefront" key. 

        Returns 
        -------
        params: dict 
            The same dictionary with the "Wavefront" value updated.
        """
        aperture = self._spider()
        wavefront = params["Wavefront"]
        wavefront = wavefront\
            .set_amplitude(wavefront.get_amplitude() * aperture)\
            .set_phase(wavefront.get_phase() * aperture)
        params["Wavefront"] = wavefront
        return params
