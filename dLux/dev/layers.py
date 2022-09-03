"""
src/dev/layers.py
-----------------
Development script for the new layers structure.
"""

from constants import *
from matplotlib import pyplot
from typing import TypeVar
from dLux.utils import (get_radial_positions, get_pixel_vector, 
    get_pixel_positions)
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


MAX_DIFF = 4


# TODO: CLIMB 


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


def factorial(n : int) -> int:
    """
    Calculate n! in a jax friendly way. Note that n == 0 is not a 
    safe case.  

    Parameters
    ----------
    n : int
        The integer to calculate the factorial of.

    Returns
    n! : int
        The factorial of the integer
    """
    return jax.lax.exp(jax.lax.lgamma(n + 1.))


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
    def _aperture(self : Layer, coordinates : int) -> Array:
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


    def _translate(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Offset the coordinate system by prespecified amounts in both
        the `x` and `y` directions. 

        Parameters
        ----------
        coordinates : Tensor
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Tensor
            The translated coordinate system. 
        """
        return coordinates\
            .at[0]\
            .set(coordinates[0] + self.x_offset)\
            .at[1]\
            .set(coordinates[1] + self.y_offset)


    def _coordinates(self : Layer) -> Tensor:
        """
        Generate the transformed coordinate system for the aperture.

        Returns
        -------
        coordinates : Tensor
            The coordinate system in the rectilinear view, with the
            x and y coordinates stacked above one another.
        """
        coordinates = self._shear(
            self._rotate(
                self._translate(
                    self._magnify(
                        self.get_pixel_scale() * \
                            get_pixel_positions(self.get_npix())))))
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
        coordinates = cartesian_to_polar(self._coordinates())
        return ((coordinates <= self.rmax) \
            & (coordinates > self.rmin)).astype(float)


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
        self.rmax = rmax


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
        x_centre, y_centre = self._get_centre()
        number_of_pixels = self.get_npix()
        maximum_radius = self.get_rmax()

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


class JWSTPrimaryApertureSegment(Aperture):
    """
    A dLux implementation of the JWST primary aperture segment.
    The segments are sketched and drawn below:

                            +---+
                           *     *
                      +---+  B1   +---+
                     *     *     *     *
                +---+  C6   +---+  C1   +---+
               *     *     *     *     *     *
              +  B6   +---+  A1   +---+  B2   +
               *     *     *     *     *     *
                +---+  A6   +---+  A2   +---+
               *     *     *     *     *     *
              +  C5   +---+       +---+  C2   +
               *     *     *     *     *     * 
                +---+  A5   +---+  A3   +---+
               *     *     *     *     *     *   
              +  B5   +---+  A4   +---+  B3   +
               *     *     *     *     *     *
                +---+  C4   +---+  C3   +---+
                     *     *     *     *
                      +---+  B4   +---+
                           *     *         
                            +---+    

    The data for the vertices is retrieved from WebbPSF and the 
    syntax for using the class is as follows:

    >>> npix = 1008 # Total number of pixels for the entire primary
    >>> appix = 200 # Pixels for this specific aperture. 
    >>> C1 = JWSTPrimaryApertureSegment("C1-1", npix, appix)
    >>> aperture = C1._aperture()

    If you want to only model one mirror then appix and npix can be 
    set to the same. The assumption is that the entire aperture is 
    going to be modelled. 

    To use the aperture to generate an orthonormal basis on the not 
    quite a hexagon we use the following code. 

    >>> basis = Basis(C1(), nterms)._basis()

    To learn the rotation, shear and other parameters of the mirror 
    we can provide this functionality to the constructor of the 
    aperture. For example:
    
    >>> C1 = JWSTPrimaryApertureSegment(
    ...     segement : str = "C1-1",
    ...     number_of_pixels : int = npix,
    ...     aperture_pixels : int = appix,
    ...     rotation : float = 0.1,
    ...     shear : float = 0.1,
    ...     magnification = 1.001)
    >>> basis = Basis(npix, nterms, C1)._basis()   

    The generation of zernike polynomials and there orthonormalisation
    is an archilies heal of the project, currently runnig much slower 
    than equivalent implementations and there is ongoing work into 
    optimising this but for now you are unfortunate.  

    Attributes
    ----------
    segement : str
        The string code for the segment that is getting modelled. 
        The format for the string is the "Ln" where "L" is the 
        letter code and "n" is the number. The format "Ln-m" will 
        also work where m is an integer mapping of "Ln" to the 
        integers. The way it works is A1, A2, A3, A4, A5 and A6 map 
        to 1, 2, 3, 4, 5 and 6. B1, B2, B3, B4, B5 and B6 map to 
        7, 9, 11, 13, 15, 17 and Cn map to the remaining numbers.
    x_offset : float, meters
        The x offset of the centre. This is automatically calculated
        in the consturctor but can be changed and optimised.
    y_offset : float, meters
        The y offset of the centre. This is automatically calculated 
        in the constructor but can be changed and optimised. 
    theta : float, Radians
        The angle of rotation from the positive x axis.
    phi : float
        The angle of shear. If the initial shape of the aperture is
        as shown in fig. 1, then the sheered aperture is as shown 
        in fig. 2.
                                  |
                                +---+
                               *  |  *
                          <---+---+---+--->
                               *  |  *
                                +---+                       
                                  |
                    fig 1. The unsheered aperture.

                                  |  / 
                                  +---+
                                * |/  *
                          <---+---+---+--->
                              *  /| *
                              +---+                       
                               /  |
                    fig 2. The sheered aperture. 

    magnification : float
        The multiplicative factor indicating the size of the aperture
        from the initial.
    pixel_scale : float, meters per pixel
        The is automatically calculated. DO NOT MODIFY THIS VALUE.  
    x : Vector[float], meters
        The x coordinates of the vertices. It is not recomended that 
        gradients be taken with respect to these. Also note that these
        are measured with respect to the centre of the coordinate 
        system with the origin at `x_offset`, `y_offset`.
    y : Vector[float]
        The y coordinates of the vertices. It is not recommended that
        gradients be take with respect to these. As with `x` these 
        are centred on the coordinate system with the origin at 
        `x_offset`, `y_offset`.
    alpha : Vector[float], radians.   
        The angular position of each vertex counterclockwise around
        the shape.            
    """
    segment : str 
    x : float
    y : float
    alpha : float


    def __init__(self : Layer, segment : str, pixels : int,
            pixel_scale : float, theta : float = 0., phi : float = 0., 
            magnification : float = 1.) -> Layer:
        """
        Parameters
        ----------
        segement : str
            The James Webb primary mirror section to modify. The 
            format of this string is "Ln", where "L" is a letter 
            "A", "B" or "C" and "n" is a number. See the class
            documentation for more detail.
        pixels : int
            The number of pixels that the entire compound aperture
            is to be generated over. 
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
        vertices = self._load(segment)
        x, y, alpha = self._vertices(vertices)
        x_offset, y_offset = self._offset(vertices)
        self.segment = str(segment)
        self.x = np.asarray(x).astype(float)
        self.y = np.asarray(y).astype(float)
        self.alpha = np.asarray(alpha).astype(float)
        super().__init__(pixels, x_offset, y_offset, theta, phi,
            magnification, pixel_scale)


    # TODO: Does this functionality really need to be separate. 
    # consider moving into the function below.              
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
        # A wedge simply represents the angular bounds of the aperture
        # I have demonstrated below with a hexagon but understand that
        # these bounds are _purely_ angular (see fig 1.)
        #
        #               +-------------------------+
        #               |                ^^^^^^^^^|
        #               |     +--------+^^^^^^^^^^|
        #               |    /        /^*^^^^^^^^^|
        #               |   /        /^^^*^^^^^^^^|
        #               |  /        /^^^^^*^^^^^^^|
        #               | +        +^^^^^^^+^^^^^^|
        #               |  *              /       |
        #               |   *            /        |
        #               |    *          /         |
        #               |     +--------+          |
        #               +-------------------------+
        #               figure 1: The angular bounds 
        #                   between the zeroth and the 
        #                   first vertices. 
        #
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
        # TODO: consider storing the vertices as a parameter to 
        # avoid reloading them every time. 
        rho, theta = self._rad_coordinates()
        segments = self._segments(theta, rho)
        return segments.sum(axis=0)


    def _load(self : Layer, segment : str):
        """
        Load the desired segment from the WebbPSF data. 

        Parameters
        ----------
        segment : str
            The segment that is desired to load. Should be in the 
            form "Ln". See the class doc string for more detail.

        Returns
        -------
        vertices : Matrix, meters
            The vertice information in any order with the shape
            (2, 6).
        """
        return dict(JWST_PRIMARY_SEGMENTS)[segment]


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


class JWSTPrimaryAperture(CompoundAperture):
    """
    Generates the full James Webb primary aperture. 
    """

    def __init__(self : Layer, number_of_pixels : int) -> Layer:
        """
        Generate the full primary aperture of the James Webb space 
        telescope. This constructor initialises default values for 
        the segements. This means that they are not rotated magnified
        or sheared. 

        Parameters
        ----------
        number_of_pixels : int
            The number of pixels to display the the entire aperture
            over.
        """
        pixel_scale = self._pixel_scale(number_of_pixels)

        SEGMENTS = [
            "A1-1", "A2-2", "A3-3", "A4-4", "A5-5", "A6-6", 
            "B1-7", "B2-9", "B3-11", "B4-13", "B5-15", "B6-17", 
            "C1-8", "C2-10", "C3-12", "C4-14", "C5-16", "C6-18"]

        apertures = dict()
        for segment in SEGMENTS:
            apertures[segment] = JWSTPrimaryApertureSegment(segment, 
                    number_of_pixels, pixel_scale)
        
        super().__init__(number_of_pixels, pixel_scale, apertures)
        

    def _pixel_scale(self : Layer, npix : int) -> float:
        """
        The physical dimesnions of a pixel along one edge. 

        Parameters
        ----------
        npix : int
            The number of pixels along one edge of the output image 

        Returns
        -------
        pixel_scale : float, meters
            The physical length along one edge of a pixel. 
        """
        B1 = JWST_PRIMARY_SEGMENTS[6][1]
        B4 = JWST_PRIMARY_SEGMENTS[9][1]
        height = B1[:, 1].max() - B4[:, 1].min()
        return height / npix


class Basis(eqx.Module):
    """
    _Abstract_ class representing a basis fixed over an aperture 
    that is used to optimise and learn aberations in the aperture. 
    
    Attributes
    ----------
    npix : int
        The number of pixels along the edge of the square array 
        representing each term in the basis.
    nterms : int
        The number of basis vectors to generate. This is determined
        by passing along the noll indices until the number is 
        reached (inclusive).
    aperture : Layer
        The aperture over which to generate the basis. This should 
        be an implementation of the abstract base class `Aperture`.
    """
    npix : int
    nterms : int    
    aperture : Layer


    def __init__(self : Layer, nterms : int,
            aperture : Layer) -> Layer:
        """
        Parameters
        ----------
        nterms : int
            The number of basis terms to generate. This determines the
            length of the leading dimension of the output Tensor.
        aperture : Layer
            The aperture to generate the basis over. This must be 
            an implementation of the abstract subclass `Aperture`. 
        """
        self.npix = int(aperture.get_npix())
        self.nterms = int(nterms)
        self.aperture = aperture


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
        basis = self._basis()
        with open(file_name, "w") as save:
            save.write(basis)


    @functools.partial(jax.vmap, in_axes=(None, 0))
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
        n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
        smallest_j_in_row = n * (n + 1) / 2 + 1 
        number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
        sign_of_shift = -(j & 1) + ~(j & 1) + 2
        base_case = (n & 1)
        m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
        return n, m


    def _radial_zernike(self : Layer, n : int, m : int,
            rho : Matrix) -> Tensor:
        """
        The radial zernike polynomial.

        Parameters
        ----------
        n : int
            The first index number of the zernike polynomial to forge
        m : int 
            The second index number of the zernike polynomial to forge.
        rho : Matrix
            The radial positions of the aperture. Passed as an argument 
            for speed.

        Returns
        -------
        radial : Tensor
            An npix by npix stack of radial zernike polynomials.
        """
        m, n = np.abs(m), np.abs(n)
        upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1
        rho = np.tile(rho, (MAX_DIFF, 1, 1))

        murder_weapon = (np.arange(MAX_DIFF) < upper)

        k = np.arange(MAX_DIFF) * murder_weapon
        coefficients = (-1) ** k * factorial(n - k) / \
            (factorial(k) * \
                factorial(((n + m) / 2).astype(int) - k) * \
                factorial(((n - m) / 2).astype(int) - k))
        radial = coefficients.reshape(MAX_DIFF, 1, 1) *\
            rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1) *\
            murder_weapon.reshape(MAX_DIFF, 1, 1)
         
        return radial.sum(axis=0)


    def _zernikes(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Calculate the zernike basis on a square pixel grid. 

        Parameters
        ----------
        number : int
            The number of zernike basis terms to calculate.
            This is a static argument to jit because the array
            size depends on it.
        pixels : int
            The number of pixels along one side of the zernike image
            for each of the n zernike polynomials.
        coordinates : Tensor
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        zernike : Tensor 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        j = np.arange(1, self.nterms + 1).astype(int)
        n, m = self._noll_index(j)
        coordinates = cartesian_to_polar(coordinates)

        # NOTE: The idea is to generate them here at the higher level 
        # where things will not change and we will be done. 
        rho = coordinates[0]
        theta = coordinates[1]

        aperture = (rho <= 1.).astype(int)

        # In the calculation of the noll coefficient we must define 
        # between the m == 0 and and the m != 0 case. I have done 
        # this in place by casting the logical operation to an int. 

        normalisation_coefficients = \
            (1 + (np.sqrt(2) - 1) * (m != 0).astype(int)) \
            * np.sqrt(n + 1)

        radial_zernikes = np.zeros((self.nterms,) + rho.shape)
        for i in np.arange(self.nterms):
            radial_zernikes = radial_zernikes\
                .at[i]\
                .set(self._radial_zernike(n[i], m[i], rho))

        # When m < 0 we have the odd zernike polynomials which are 
        # the radial zernike polynomials multiplied by a sine term.
        # When m > 0 we have the even sernike polynomials which are 
        # the radial polynomials multiplies by a cosine term. 
        # To produce this result without logic we can use the fact
        # that sine and cosine are separated by a phase of pi / 2
        # hence by casting int(m < 0) we can add the nessecary phase.
        out_shape = (self.nterms, 1, 1)

        theta = np.tile(theta, out_shape)
        m = m.reshape(out_shape)
        phase_mod = (m < 0).astype(int) * np.pi / 2
        phase = np.cos(np.abs(m) * theta - phase_mod)

        normalisation_coefficients = \
            normalisation_coefficients.reshape(out_shape)
        
        return normalisation_coefficients * radial_zernikes \
            * aperture * phase 


    def _orthonormalise(self : Layer, aperture : Matrix, 
            zernikes : Tensor) -> Tensor:
        """
        The hexike polynomials up until `number_of_hexikes` on a square
        array that `number_of_pixels` by `number_of_pixels`. The 
        polynomials can be restricted to a smaller subset of the 
        array by passing an explicit `maximum_radius`. The polynomial
        will then be defined on the largest hexagon that fits with a 
        circle of radius `maximum_radius`. 
        
        Parameters
        ----------
        aperture : Matrix
            An array representing the aperture. This should be an 
            `(npix, npix)` array. 
        number_of_hexikes : int = 15
            The number of basis terms to generate. 
        zernikes : Tensor
            The zernike polynomials to orthonormalise on the aperture.
            This tensor should be `(nterms, npix, npix)` in size, where 
            the first axis represents the noll indexes. 

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
        pixel_area = aperture.sum()
        basis = np.zeros(zernikes.shape).at[0].set(aperture)
        
        for j in np.arange(1, self.nterms):
            intermediate = zernikes[j] * aperture

            coefficient = -1 / pixel_area * \
                (zernikes[j] * basis[1 : j + 1] * aperture)\
                .sum(axis = (1, 2))\
                .reshape(j, 1, 1) 

            intermediate += (coefficient * basis[1 : j + 1])\
                .sum(axis = 0)
            
            basis = basis\
                .at[j]\
                .set(intermediate / \
                    np.sqrt((intermediate ** 2).sum() / pixel_area))
        
        return basis


    def basis(self : Layer) -> Tensor:
        """
        Generate the basis. Requires a single run after which,
        the basis is cached and can be used with no computational 
        cost.  

        Parameters
        ----------
        aperture : Matrix
            The aperture over which the basis is to be generated. 
        coordinates : Matrix, meters, radians 
            The coordinate system over which to generate the aperture.

        Returns
        -------
        basis : Tensor
            The basis polynomials evaluated on the square arrays
            containing the apertures until `maximum_radius`.
            The leading dimension is `n` long and 
            each stacked array is a basis term. The final shape is:
            `(n, npix, npix)`
        """
        coordinates = self.aperture._coordinates()
        aperture = self.aperture._aperture()
        zernikes = self._zernikes(coordinates)
        return self._orthonormalise(aperture, zernikes)


class CompoundBasis(eqx.Module):
    """
    Interfaces with compound apertures to generate basis over them.
    """
    bases : list

 
    def __init__(self : Layer, nterms : int, 
            compound_aperture : Layer) -> Layer:
        """
        Parameters
        ----------
        nterms : int
            The number of basis terms to generate over each mirror.
            pass a list of integers in the order that the apertures
            appear in compound_aperture.
        compound_aperture : Layer
            The compound aperture to generate a basis over. 
        """
        apertures = compound_aperture.apertures.values()
        if isinstance(nterms, list):
            bases = [Basis(nterm, aperture) \
                for nterm, aperture in zip(nterms, apertures)]
        else:
            bases = [Basis(nterms, aperture) for aperture in apertures]
        self.bases = bases


    def basis(self : Layer) -> Tensor:
        """
        Generate a basis over a compound aperture.

        Returns 
        -------
        basis : Tensor
            The basis represented as `(napp, nterms, npix, npix)`
            array
        """
        return np.stack([basis.basis() for basis in self.bases])
