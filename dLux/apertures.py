import equinox as eqx
import jax.numpy as np
import jax 
import dLux
import abc
import functools
from typing import TypeVar


Array = TypeVar("Array")
Layer = TypeVar("Layer")
Tensor = TypeVar("Tensor")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")


__all__ = ["Aperture", "CompoundAperture", "SquareAperture", 
    "RectangularAperture", "CircularAperture", "AnnularAperture"]


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
    """
    occulting: bool 
    softening: float
    x_offset: float
    y_offset: float
    

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
        self.softening = jax.lax.cond(softening, lambda: 1., lambda: np.inf)
        self.occulting = bool(occulting)


    def get_centre(self) -> tuple:
        """
        Returns 
        -------
        x, y : tuple(float, float) meters
            The coodinates of the centre of the aperture. The first 
            element of the tuple is the x coordinate and the second 
            is the y coordinate.
        """
        return np.array([self.x_offset, -self.y_offset])


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
        steepness = self.softening * distances.shape[-1]
        return (np.tanh(steepness * distances) + 1.) / 2.


    @abc.abstractmethod
    def _metric(self, distances: Array) -> Array:
        """
        """


    def _aperture(self, coordinates: Array) -> Array:
        """
        """
        aperture = jax.lax.cond(self.occulting,
            lambda aperture: 1 - aperture,
            lambda aperture: aperture,
            self._metric(coordinates))

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
        return eqx.tree_at(lambda basis : basis.x_offset, 
            self, np.asarray(x).astype(float))


    def get_x_offset(self) -> float:
        return self.x_offset


    def set_y_offset(self, y : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The y coordinate of the centre of the hexagonal
            aperture.
        """
        return eqx.tree_at(lambda basis : basis.y_offset, 
            self, np.asarray(y).astype(float))


    def get_y_offset(self) -> float:
        """
        Returns:
        --------
        y_offset: float, meters
            
        """
        return self.y_offset


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
        wavefront = wavefront.multiply_amplitude(
            self._aperture(
                wavefront.pixel_coordinates()))
        parameters["Wavefront"] = wavefront
        return parameters

    def largest_extent(self, coordinates : Tensor) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

        Parameters
        ----------
        coordinates : Tensor
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        largest_extent : float
            The maximum distance from centre to edge of aperture
        """

        aperture = self._aperture(coordinates)

        #TODO: check between here and basis function where the flipping in y axis should go
        coordinates = coordinates.at[1].set(coordinates[1][::-1,:])# have to flip in y dir for meshgrid to cartesian


        x_offset = self.get_x_offset()
        y_offset = self.get_y_offset()


        x_coords_of_app = coordinates[0][aperture > 0.5] - x_offset
        y_coords_of_app = coordinates[1][aperture > 0.5] - y_offset 

        trans_rho = dLux.utils.cartesian_to_polar(np.array([x_coords_of_app, y_coords_of_app]))[0]
        largest_extent = np.max(trans_rho)

        return largest_extent

    def compute_aperture_normalised_coordinates(self, coordinates : Tensor) -> Array:
        """
        Shift a set of wavefront coodinates to be centered on the aperture and scaled such that
        the radial distance is 1 to the edge of the aperture, returned in polar form

        Parameters
        ----------
        coordinates : Tensor
            The cartesian coordinates to generate the aperture on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  
        
        Returns
        -------
        coordinates : Tensor
            the radial coordinates centered on the centre of the aperture and scaled such that they are 1
            at the maximum extent of the aperture
            The dimensions of the tensor are be `(2, npix, npix)`
        """
        
        # TODO: check where flips should go
        coordinates = coordinates.at[1].set(coordinates[1][::-1,:])

        x_offset = self.get_x_offset()
        y_offset = self.get_y_offset()

        # This is the translation and scaling of the normalised coordinate system. 
        # translate and then multiply by 1 / largest_extent.
        trans_coords = coordinates - np.array([x_offset, y_offset]).reshape((2, 1, 1))
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

    def largest_extent(self, coordinates : Tensor) -> float:
        return self.radius

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
        coordinates = self._rotate(self._translate(coordinates))  
        x_mask = self._soften(- np.abs(coordinates[0]) + self.length / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask

    def largest_extent(self, coordinates: Tensor) -> float:
        return np.hypot(np.array([self.length, self.width]))


class SquareAperture(RotatableAperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this one tracks less parameters.

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
        coordinates = self._rotate(self._translate(coordinates))
        x_mask = self._soften(- np.abs(coordinates[0]) + self.width / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


    def largest_extent(self, coordinates: Tensor) -> float:
        return np.sqrt(2) * self.width / 2.

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
    use_prod  : bool    


    def __init__(self, apertures: dict, use_prod : bool) -> Layer:
        """
        Parameters
        ----------
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        use_prod : bool
            A flag to indicate if the product or max should be used to combine multiple apertures
        """
        self.apertures = apertures
        self.use_prod = use_prod


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
        if self.use_prod:
            return np.stack(apertures).prod(axis=0)
        else:  
            return np.stack(apertures).max(axis=0)


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

