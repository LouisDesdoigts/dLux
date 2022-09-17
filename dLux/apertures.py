from typing import TypeVar
from dLux.utils import (get_positions_vector, get_pixel_positions)
from abc import ABC, abstractmethod 
import equinox as eqx
import jax.numpy as np
import jax 
import functools

config.update("jax_enable_x64", True)

Array = TypeVar("Array")
Layer = TypeVar("Layer")
Tensor = TypeVar("Tensor")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")


__all__ = ["Aperture", "CompoundAperture", "SoftEdgedAperture", 
    "SquareAperture", "SoftEdgedSquareAperture", "RectangularAperture",
    "SoftEdgedRectangularAperture", "CircularAperture", 
    "SofteEdgedCircularAperture", "AnnularAperture", 
    "SoftEdgedAnnularAperture"]


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
    theta = np.arctan2(-coordinates[1], coordinates[0])
    return np.array([rho, theta])


class Aperture(eqx.Module, ABC):
    """
    An abstract class that defines the structure of all the concrete
    apertures. An aperture is represented by an array, usually in the
    range of 0. to 1.. Values in between can be used to represent 
    soft edged apertures and intermediate surfaces. 

    Attributes
    ----------
    pixels : int
        The number of pixels along one edge of the array which 
        represents the aperture.
    x_offset : float, meters
        The x coordinate of the centre of the aperture.
    y_offset : float, meters
        The y coordinate of the centre of the aperture.
    theta : float, radians
        The angle of rotation from the positive x-axis. 
    phi : float, radians
        The rotation of the y-axis away from the vertical and torward 
        the negative x-axis. 
    magnification : float
        The radius of the aperture. The radius belongs to the smallest
        circle that completely contains the aperture. For the math
        nerds the infimum of the set of circles whose union with the
        aperture is the aperture. 
    pixel_scale : float, meters per pixel 
        The length along one side of a square pixel. 
    """
    pixels : int
    x_offset : float
    y_offset : float
    theta : float
    phi : float
    magnification : float
    pixel_scale : float # Not gradable
    

    def __init__(self : Layer, number_of_pixels : int,
            x_offset : float, y_offset : float, theta : float,
            phi : float, magnification : float,
            pixel_scale : float) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels : int
            The number of pixels along one side of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The dimension along one edge of the pixel. At present 
            only square (meter) pixels are supported. 
        """
        self.pixels = int(number_of_pixels)
        self.x_offset = np.asarray(x_offset).astype(float)
        self.y_offset = np.asarray(y_offset).astype(float)
        self.theta = np.asarray(theta).astype(float)
        self.phi = np.asarray(phi).astype(float)
        self.magnification = np.asarray(magnification).astype(float)
        self.pixel_scale = float(pixel_scale)

    
    @abstractmethod
    def _aperture(self : Layer) -> Array:
        """
        Generate the aperture array as an array. 
        
        Parameters
        ----------
        coordinates : Tensor
            The coordinate system over which to generate the aperture 
            The leading dimesnion of the tensor should be the x and y
            coordinates in that order. 

        Returns
        -------
        aperture : Array[Float]
            The aperture. If these values are confined between 0. and 1.
            then the physical interpretation is the transmission 
            coefficient of that pixel. 
        """


    def get_pixel_scale(self: Layer) -> float:
        """
        Returns
        -------
        pixel_scale : float, meters per pixel
            The length along one side of the a square picel used in
            the constuction of the aperture.
        """
        return self.pixel_scale        


    def get_npix(self : Layer) -> int:
        """
        Returns
        -------
        pixels : int
            The number of pixels that parametrise this aperture.
        """
        return self.pixels


    def get_centre(self : Layer) -> tuple:
        """
        Returns 
        -------
        x, y : tuple(float, float) meters
            The coodinates of the centre of the aperture. The first 
            element of the tuple is the x coordinate and the second 
            is the y coordinate.
        """
        return self.x_offset, self.y_offset


    def get_rotation(self : Layer) -> float:
        """
        Returns 
        -------
        theta : float, radians 
            The angle of rotation of the aperture away from the 
            positive x-axis. 
        """
        return self.theta


    def get_shear(self : Layer) -> float:
        """
        Returns 
        -------
        phi : float, radians
            The angle that the y-axis of the coordinate system of 
            the aperture has been rotated towards the negative
            x-axis. This corresponds to a shear. 
        """
        return self.shear


    def get_magnification(self : Layer) -> float:
        """
        Returns
        -------
        magnification : float
            A proportionality factor indicating the magnification 
            of the aperture. 
        """
        return self.magnification


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
        return 1 / self.get_magnification() * coordinates


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
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]])            
        return np.apply_along_axis(np.matmul, 0, coordinates, 
            rotation_matrix) 


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
            .set(coordinates[0] - coordinates[1] * np.tan(self.phi)) 


    def _coordinates(self : Layer) -> Tensor:
        """
        Generate the transformed coordinate system for the aperture.

        Returns
        -------
        coordinates : Tensor
            The coordinate system in the rectilinear view, with the
            x and y coordinates stacked above one another.
        """
        x_pixel_offset = self.x_offset / self.pixel_scale
        y_pixel_offset = self.y_offset / self.pixel_scale
        coordinates = self._shear(
            self._rotate(
                self._magnify(
                    self.get_pixel_scale() * \
                        get_pixel_positions(self.pixels, 
                            x_pixel_offset, y_pixel_offset))))
        return coordinates


    def set_theta(self : Layer, theta : float) -> Layer:
        """
        Parameters
        ----------
        theta : float
            The angle of rotation from the positive x-axis.  

        Returns
        -------
        basis : HexagonalBasis 
            The rotated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.theta, self, theta)


    def set_magnification(self : Layer, rmax : float) -> Layer:
        """
        Parameters
        ----------
        rmax : float
            The radius of the smallest circle that can completely 
            enclose the aperture.

        Returns
        -------
        basis : HexagonalBasis
            The magnified hexagonal basis.
        """
        return eqx.tree_at(lambda basis : basis.magnification, self, rmax)


    def set_shear(self : Layer, phi : float) -> Layer:
        """
        Parameters
        ----------
        phi : float
            The angle of shear from the positive y-axis.

        Returns
        -------
        basis : HexagonalBasis
            The sheared hexagonal basis.
        """
        return eqx.tree_at(lambda basis : basis.phi, self, phi)      


    def set_x_offset(self : Layer, x : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The x coordinate of the centre of the hexagonal
            aperture.

        Returns
        -------
        basis : HexagonalBasis
            The translated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.x_offset, self, x)


    def set_y_offset(self : Layer, y : float) -> Layer:
        """
        Parameters
        ----------
        x : float
            The y coordinate of the centre of the hexagonal
            aperture.

        Returns
        -------
        basis : HexagonalBasis
            The translated hexagonal basis. 
        """
        return eqx.tree_at(lambda basis : basis.y_offset, self, y)


    def __call__(self : Layer, parameters : dict) -> dict:
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
        # TODO: There should probably be a multiplication of the phase 
        # as well. 
        wavefront = wavefront.mulitply_amplitude(self._aperture())
        parameters["Wavefront"] = wavefront
        return parameters


class SoftEdgedAperture(Aperture, ABC):
    """
    Apertures that have hard edges can result in undefined gradients. 
    To combat this annoying behaviour we have added the soft edged apertures
    which run add a few non-binary pixels at the border. 
    """


    def _soft_edge(self: Layer, image: Array) -> Array:
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
        steepness = self.pixels
        return (np.tanh(steepness * image) + 1.) / 2.


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


    def __init__(self : Layer, npix : int, x_offset : float, 
            y_offset : float, theta : float, phi : float, 
            magnification : float, pixel_scale : float,
            rmax : float, rmin : float) -> Layer:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        rmax : float, meters
            The outer radius of the annular aperture. 
        rmin : float, meters
            The inner radius of the annular aperture. 
        """
        super().__init__(npix, x_offset, y_offset, theta, phi, 
            magnification, pixel_scale)
        self.rmax = np.asarray(rmax).astype(float)
        self.rmin = np.asarray(rmin).astype(float)


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
        coordinates = cartesian_to_polar(self._coordinates())
        return ((coordinates <= self.rmax) \
            & (coordinates > self.rmin)).astype(float)


class SoftEdgedAnnularAperture(SoftEdgedAperture):
    """
    An annular aperture (see dLux.AnnularAperture) however, the edges have 
    several soft pixels (i.e. between 1. and 0.). The goal is to improve the 
    stability of gradients taken through the aperture. 

    Parameters
    ----------
    rmin: float, meters
        The radius of the inner edge of the annular opening. 
    rmax: float, meters
        The radius of the outer edge of the annular opening. 
    """
    rmin: float
    rmax: float


    def __init__(self: Layer, pixels: int, x_offset: float, y_offset: float, 
            theta: float, phi: float, magnification: float, pixel_scale: float,
            rmin: float, rmax: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        rmax : float, meters
            The outer radius of the annular aperture. 
        rmin : float, meters
            The inner radius of the annular aperture. 
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.rmin = np.asarray(rmin).astype(float)
        self.rmax = np.asarray(rmax).astype(float)


    def _aperture(self: Layer) -> Array:
        """
        Generates the aperture. There should be around three (depends on the
        scale), non-binary pixels at the edges.

        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = cartesian_to_polar(self._coordinates())[0]
        inner = self._soft_edge(- coordinates + self.rmin)
        outer = self._soft_edge(coordinates - self.rmax)
        return inner * outer


class CircularAperture(Aperture):
    """
    A circular aperture represented as a binary array.

    Parameters
    ----------
    radius: float, meters
        The radius of the opening. 
    """
    radius: float
   
 
    def __init__(self: Layer, pixels: int, x_offset: float, y_offset: float, 
            theta: float, phi: float, magnification: float, pixel_scale: float,
            radius: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        radius: float, meters 
            The radius of the aperture.
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.radius = np.asarray(radius).astype(float)


    def _aperture(self: Layer) -> Array:
        """
        Returns
        -------
        aperture: Array
            The aperture represented as a pixel array.
        """
        coordinates = cartesian_to_polar(self._coordinates())[0]
        return coordinates < self.radius


class SoftEdgedCircularAperture(SoftEdgedAperture):
    """
    A circular aperture that is stabalised against numerical gradient 
    calculations. 

    Parameters
    ----------
    radius: float, meters 
        The radius of the opening. 
    """
    radius: float


    def __init__(self: Layer, pixels: int, x_offset: float, y_offset: float, 
            theta: float, phi: float, magnification: float, pixel_scale: float,
            radius: float) -> Layer:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        radius: float, meters 
            The radius of the aperture.
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.radius = np.asarray(radius).astype(float)


    def _aperture(self: Layer) -> Array:        
        """
        Generates the apperature as a square array. Note: there is a layer of 
        non-binary pixels near the edge to stabalise the numerical gradients. 

        Returns
        -------
        aperture: Array
            The aperture.
        """
        coordinates = cartesian_to_polar(self._coordinates())[0]
        return self._soft_edge(self.radius - coordinates)


class RectangularAperture(Aperture):
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


    def __init__(self: Layer, pixels: int, x_offset: float, y_offset: float,
            theta: float, phi: float, magnification: float,
            pixel_scale: float, length: float, width: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        length: float, meters 
            The length of the aperture in the y-direction.
        width: float, meters
            The length of the aperture in the x-direction.
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.length = np.asarray(length).astype(float)
        self.width = np.asarray(width).astype(float)


    def _aperture(self: Layer) -> Array:
        """
        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._coordinates()
        x_mask = np.abs(coordinates[0]) < (self.length / 2.)
        y_mask = np.abs(coordinates[1]) < (self.width / 2.)    
        return y_mask * x_mask        


class SoftEdgedRectangularAperture(SoftEdgedAperture):
    """
    A rectangular aperture with soft-edges to make it friendly for 
    sutomatic differentiation. 

    Parameters
    ----------
    length: float, meters
        The length of the aperture in the y-direction. 
    width: float, meters
        The length of the aperture in the x-direction. 
    """
    length: float
    width: float 


    def __init__(self: Layer, pixels: int, x_offset: float, y_offset: float,
            theta: float, phi: float, magnification: float, 
            pixel_scale: float, length: float, width: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        length: float, meters 
            The length of the aperture in the y-direction.
        width: float, meters
            The length of the aperture in the x-direction.
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.length = np.asarray(length).astype(float) 
        self.width = np.asarray(width).astype(float)


    def _aperture(self: Layer) -> Array:
        """
        Generates the aperture. There should be around three (depends on the
        scale), non-binary pixels at the edges.

        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._coordinates()
        x_mask = self._soft_edge(- np.abs(coordinates[0]) + self.length / 2.) 
        y_mask = self._soft_edge(- np.abs(coordinates[1]) + self.width / 2.)    
        return y_mask * x_mask        
    

class SquareAperture(Aperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this obe tracks less parameters.

    Parameters
    ----------
    width: float, meters
        The side length of the square. 
    """
    width: float
   
 
    def __init__(self: Layer, pixels: float, x_offset: float, y_offset: float,
            theta: float, phi: float, magnification: float, 
            pixel_scale: float, width: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        width: float, meters
            The side length of the square. 
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification,
            pixel_scale)
        self.width = np.asarray(width).astype(float)
    

    def _aperture(self: Layer) -> Array:
        """
        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._coordinates()
        x_mask = np.abs(coordinates[0]) < (self.width / 2.)
        y_mask = np.abs(coordinates[1]) < (self.width / 2.)
        return x_mask * y_mask


class SoftEdgedSquareAperture(SoftEdgedAperture):
    """
    A square aperture with non-binary pixels near the edges to improve the 
    differential stability of the program. 

    Parameters
    ----------
    width: float, meters
        The side length of the square. 
    """   
    width: float


    def __init__(self: Layer, pixels: float, x_offset: float, y_offset: float,
            theta: float, phi: float, magnification: float, pixel_scale: float,
            width: float) -> Array:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        width: float, meters
            The side length of the square. 
        """
        super().__init__(pixels, x_offset, y_offset, theta, phi, magnification, 
            pixel_scale)
        self.width = np.asarray(width).astype(float)


    def _aperture(self: Layer) -> Array:
        """
        Generates the aperture. There should be around three (depends on the
        scale), non-binary pixels at the edges.

        Returns
        -------
        aperture: Array
            The array representation of the aperture. 
        """
        coordinates = self._coordinates()
        x_mask = self._soft_edge(- np.abs(coordinates[0]) + self.width / 2.)
        y_mask = self._soft_edge(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


class HexagonalAperture(Aperture):
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


    def __init__(self : Layer, npix : int, x_offset : float, 
            y_offset : float, theta : float, phi : float, 
            magnification : float, pixel_scale : float,
            rmax : float) -> Layer:
        """
        Parameters
        ----------
        npix : int
            The number of layers along one edge of the array that 
            represents this aperture.
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        phi : float, radians
            The rotation of the y-axis away from the vertical and 
            torward the negative x-axis measured from the vertical.
        magnification : float
            The scaling of the aperture. 
        pixel_scale : float, meters per pixel
            The length of one side of a square pixel. Defines the 
            physical size of the array representing the aperture.
        rmax : float, meters
            The distance from the center of the hexagon to one of
            the vertices. . 
        """
        super().__init__(npix, x_offset, y_offset, theta, phi, 
            magnification, pixel_scale)
        self.rmax = np.asarray(rmax).astype(float)


    def get_rmax(self : Layer) -> float:
        """
        Returns
        -------
        max_radius : float, meters
            The distance from the centre of the hexagon to one of 
            the vertices.
        """
        return self.rmax


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
        x_centre, y_centre = self.get_centre()
        number_of_pixels = self.get_npix()
        maximum_radius = self.get_rmax()

        x, y = get_pixel_positions(number_of_pixels, -x_centre,
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


class PolygonalAperture(Aperture):
    """
    """
    x : float
    y : float
    alpha : float


    def __init__(self : Layer, pixels : int, pixel_scale : float, 
            vertices : Matrix, theta : float = 0., phi : float = 0., 
            magnification : float = 1.) -> Layer:
        """
        Parameters
        ----------
        pixels : int
            The number of pixels that the entire compound aperture
            is to be generated over. 
        pixel_scale : float
    
        vertices : float
    
        theta : float, radians
            The angle that the aperture is rotated from the positive 
            x-axis. By default the horizontal sides are parallel to 
            the x-axis.
        phi : float, radians
            The angle that the y-axis is rotated away from the 
            vertical. This results in a sheer. 
        magnification : float
            A factor by which to enlarge or shrink the aperture. 
            Should only be very small amounts in typical use cases.
        """
        x, y, alpha = self._vertices(vertices)
        x_offset, y_offset = self._offset(vertices)
        self.x = np.asarray(x).astype(float)
        self.y = np.asarray(y).astype(float)
        self.alpha = np.asarray(alpha).astype(float)
        super().__init__(pixels, x_offset, y_offset, theta, phi,
            magnification, pixel_scale)


    def _wrap(self : Layer, array : Vector, order : Vector) -> tuple:
        """
        Re-order an array and duplicate the first element as an additional
        final element. Satisfies the postcondition `wrapped.shape[0] ==
        array.shape[0] + 1`. This is just a helper method to simplify 
        external object and is not physically important (Only invoke 
        this method if you know what you are doing)

        Parameters
        ----------
        array : Vector
            The 1-dimensional vector to sort and append to. Must be one 
            dimesnional else unexpected behaviour can occur.
        order : Vector
            The new order for the elements of `array`. Will be accessed 
            by invoking `array.at[order]` hence `order` must be `int`
            `dtype`.

        Returns
        -------
        wrapped : Vector
            `array` with `array[0]` appended to the end. The dimensions
            of `array` are also expanded twofold so that the final
            shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
            This is just for the vectorisation demanded later in the 
            code.
        """
        _array = np.zeros((array.shape[0] + 1,))\
            .at[:-1]\
            .set(array.at[order].get())\
            .reshape(-1, 1, 1)
        return _array.at[-1].set(_array[0])
        

    def _vertices(self : Layer, vertices : Matrix) -> tuple:
        """
        Generates the vertices that are compatible with the rest of 
        the transformations from the raw data vertices.

        Parameters
        ----------
        vertices : Matrix, meters
            The vertices loaded from the WebbPSF module. 

        Returns
        -------
        x, y, angles : tuple 
            The vertices in normalised positions and wrapped so that 
            they can be used in the generation of the compound aperture.
            The `x` is the x coordinates of the vertices, the `y` is the 
            the y coordinates and `angles` is the angle of the vertex. 
        """
        _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
        _y = (vertices[:, 1] - np.mean(vertices[:, 1]))

        _angles = np.arctan2(_y, _x)
        _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)

        # By default the `np.arctan2` function returns values within the 
        # range `(-np.pi, np.pi)` but comparisons are easiest over the 
        # range `(0, 2 * np.pi)`. This is where the logic implemented 
        # above comes from. 

        order = np.argsort(_angles)

        x = self._wrap(_x, order)
        y = self._wrap(_y, order)
        angles = self._wrap(_angles, order).at[-1].add(2 * np.pi)

        # The final `2 * np.pi` is designed to make sure that the wrap
        # of the first angle is within the angular coordinate system 
        # associated with the aperture. By convention this is the
        # range `angle[0], angle[0] + 2 * np.pi` what this means in 
        # practice is that the first vertex appearing in the array 
        # is used to chose the coordinate system in angular units. 

        return x, y, angles


    def _offset(self : Layer, vertices : Matrix) -> tuple:
        """
        Get the offsets of the coordinate system.

        Parameters
        ----------
        vertices : Matrix 
            The unprocessed vertices loaded from the JWST data file.
            The correct shape for this array is `vertices.shape == 
            (2, number_of_vertices)`. 
        pixel_scale : float, meters
            The physical size of each pixel along one of its edges.

        Returns 
        -------
        x_offset, y_offset : float, meters
            The x and y offsets in physical units. 
        """
        x_offset = np.mean(vertices[:, 0])
        y_offset = np.mean(vertices[:, 1])
        return x_offset, y_offset


    # TODO: number_of_pixels can be moved out as a parameter
    def _rad_coordinates(self : Layer) -> tuple:
        """
        Generates the vectorised coordinate system associated with the 
        aperture.

        Parameters
        ----------
        phi_naught : float 
            The angle substending the first vertex. 

        Returns 
        -------
        rho, theta : tuple[Tensor]
            The stacked coordinate systems that are typically passed to 
            `_segments` to generate the segments.
        """
        cartesian = self._coordinates()
        positions = cartesian_to_polar(cartesian)

        # rho = positions[0] * self.get_pixel_scale()
        rho = positions[0]        

        theta = positions[1] 
        theta += 2 * np.pi * (positions[1] < 0.)
        theta += 2 * np.pi * (theta < self.alpha[0])

        rho = np.tile(rho, (6, 1, 1))
        theta = np.tile(theta, (6, 1, 1))
        return rho, theta


    def _edges(self : Layer, rho : Tensor, theta : Tensor) -> Tensor:
        """
        Generate lines connecting adjacent vertices.

        Parameters
        ----------
        rho : Tensor, meters
            Represents the radial distance of every point from the 
            centre of __this__ aperture. 
        theta : Tensor, Radians
            The angle associated with every point in the final bitmap.

        Returns
        -------
        edges : Tensor
            The edges represented as a Bitmap with the points inside the 
            edge marked as 1. and outside 0. The leading axis contains 
            each unique edge and the corresponding matrix is the bitmap.
        """
        # This is derived from the two point form of the equation for 
        # a straight line (eq. 1)
        # 
        #           y_2 - y_1
        # y - y_1 = ---------(x - x_1)
        #           x_2 - x_1
        # 
        # This is rearranged to the form, ay - bx = c, where:
        # - a = (x_2 - x_1)
        # - b = (y_2 - y_1)
        # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
        # we can then drive the transformation to polar coordinates in 
        # the usual way through the substitutions; y = r sin(theta), and 
        # x = r cos(theta). The equation is now in the form 
        #
        #                  c
        # r = ---------------------------
        #     a sin(theta) - b cos(theta) 
        #
        a = (self.x[1:] - self.x[:-1])
        b = (self.y[1:] - self.y[:-1])
        c = (a * self.y[:-1] - b * self.x[:-1])

        linear = c / (a * np.sin(theta) - b * np.cos(theta))
        #return rho < (linear * self.get_pixel_scale())
        return rho < linear


    def _wedges(self : Layer, theta : Tensor) -> Tensor:
        """
        The angular bounds of each segment of an individual hexagon.

        Parameters
        ----------
        theta : Tensor, Radians
            The angle away from the positive x-axis of the coordinate
            system associated with this aperture. Please note that `theta`
            May not start at zero. 

        Returns 
        -------
        wedges : Tensor 
            The angular bounds associated with each pair of vertices in 
            order. The leading axis of the Tensor steps through the 
            wedges in order arround the circle. 
        """
        return (self.alpha[:-1] < theta) & (theta < self.alpha[1:])


    def _segments(self : Layer, theta : Tensor, rho : Tensor) -> Tensor:
        """
        Generate the segments as a stacked tensor. 

        Parameters
        ----------
        theta : Tensor
            The angle of every pixel associated with the coordinate system 
            of this aperture. 
        rho : Tensor
            The radial positions associated with the coordinate system 
            of this aperture. 

        Returns 
        -------
        segments : Tensor 
            The bitmaps corresponding to each vertex pair in the vertices.
            The leading dimension contains the unique segments. 
        """
        edges = self._edges(rho, theta)
        wedges = self._wedges(theta)
        return (edges & wedges).astype(float)
        

    def _aperture(self : Layer) -> Matrix:
        """
        Generate the BitMap representing the aperture described by the 
        vertices. 

        Returns
        -------
        aperture : Matrix 
            The Bit-Map that represents the aperture. 
        """
        rho, theta = self._rad_coordinates()
        segments = self._segments(theta, rho)
        return segments.sum(axis=0)


class CompoundAperture(eqx.Module):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. For example:

    >>> x_sep = 0.1
    >>> width = 0.005
    >>> height = 0.2
    >>> first_slit = RectangularAperture(
    ...     npix=1008, width=width, height=height, 
    ...     x_offset = -x_sep/2, y_offset=0.,
    ...     theta=0., phi=0., magnification=0.)
    >>> second_slit = RectangularAperture(
    ...     npix=1008, width=width, height=height, 
    ...     x_offset = x_sep/2, y_offset=0.,
    ...     theta=0., phi=0., magnification=0.)
    >>> apertures = {"Right": first_slit, "Left": second_slit}
    >>> double_slit = CompoundAperture(apertures)
    >>> double_slit["Right"]

    Attributes
    ----------
    npix : int
        The number of pixels to be used to represent the entire 
        set of apertures.
    pixel_scale : float, meters per pixel
        The length of the side of one pixel in the image. 
    apertures : dict(str, Layer)
        The apertures that make up the compound aperture. 
    """
    npix : int
    pixel_scale : float
    apertures : dict    


    def __init__(self : Layer, number_of_pixels : int, 
            pixel_scale : float, apertures : dict) -> Layer:
        """
        Parameters
        ----------
        number_of_pixels : int
            The number of pixels used to represent the compound 
            aperture. 
        pixel_scale : float, meters per pixel
            The length of one edge of a pixel in the rendered 
            aperture.
        apertures : dict
            The aperture objects stored in a dictionary of type
            {str : Layer} where the Layer is a subclass of the 
            Aperture.
        """
        self.npix = int(number_of_pixels)
        self.pixel_scale = float(pixel_scale)
        self.apertures = apertures


    def __getitem__(self : Layer, key : str) -> Layer:
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


    def __setitem__(self : Layer, key : str, value : Layer) -> None:
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


    def _aperture(self : Layer) -> Matrix:
        """
        Returns 
        -------
        aperture : Matrix
            An aperture generated by combining all of the sub 
            apertures that were stored. 
        """
        apertures = []
        for aperture in self.apertures.values():
            apertures.append(aperture._aperture())
        return np.stack(apertures).sum(axis=0)


    def get_npix(self : Layer) -> int:
        """
        Returns
        -------
        pixels : int
            The number of pixels along one edge of the output image.
        """
        return self.npix
