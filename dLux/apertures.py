import dLux
from abc import ABC, abstractmethod
from jax import numpy as np, lax 


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront


__all__ = ["CircularAperture", "SquareAperture", 
    "HexagonalAperture", "RegularPolygonalAperture", 
    "IrregularPolygonalAperture", "StaticAperture",
    "AberratedAperture", "StaticAberratedAperture"]


two_pi: float = 2. * np.pi



class ApertureLayer(dLux.optics.OpticalLayer, ABC):
    """
    The ApertureLayer groups together all of the functionality 
    that is associated with the apertures. Very little of this
    functionality is actually implemented by itself because 
    implementation varies amongst the subclasses. It is a 
    layer within the class Heirachy that exists almost purely 
    as a classification.

    Attributes:
    -----------
    name: str
        The address of this ApertureLayer within the optical 
        system. 
    """

    
    def __init__(self: dLux.optics.OpticalLayer) -> dLux.optics.OpticalLayer:
        """
        Automatically assigns the name of the layer to be the 
        class name. 
        """
        self.name = self.__class__.__name__


class AbstractDynamicAperture(ApertureLayer, ABC):
    """
    AbstractDynamicAperture:
    ------------------------
    This class also provides a level of classification without 
    implementing any functionality. It was created so that 
    `DynamicAberratedAperture` and `DynamicAperture` could 
    inherit from a common base. 

    Attributes:
    -----------
    centre: Array, meters
        The (x, y) coordinate of the centre of the aperture.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    centre: Array
    strain: Array
    compression: Array
    rotation: Array
    

    def __init__(
            self        : ApertureLayer, 
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.) -> ApertureLayer:
        """
        The default aperture is dis-allows the learning of all 
        parameters. 

        Parameters
        ----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__()
        self.centre = np.asarray(centre).astype(float)
        self.strain = np.asarray(strain).astype(float)
        self.compression = np.asarray(compression).astype(float)
        self.rotation = np.asarray(rotation).astype(float)


    def _rotate(self: ApertureLayer, coords: Array) -> Array:
        """
        Rotate the coordinate system by a pre-specified amount,
        `self.rotation`

        Parameters:
        -----------
        coords : Array
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns:
        --------
        coordinates : Array
            The rotated coordinate system. 
        """
        x, y = coords[0], coords[1]
        new_x = np.cos(self.rotation) * x + np.sin(self.rotation) * y
        new_y = -np.sin(self.rotation) * x + np.cos(self.rotation) * y
        return np.array([new_x, new_y])


    def _translate(self: ApertureLayer, coords: Array) -> Array:
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
        return coords - self.centre[:, None, None]


    def _strain(self: ApertureLayer, coords: Array) -> Array:
        """
        Apply a strain to the coordinate system. 

        Parameters:
        -----------
        coords: Array
            The coordinates to apply the strain to. 

        Returns:
        --------
        coords: Array 
            The strained coordinate system. 
        """
        trans_coords: Array = np.transpose(coords, (0, 2, 1))
        return coords + trans_coords * self.strain[:, None, None]


    def _compress(self: ApertureLayer, coords: Array) -> Array:
        """
        Apply a compression to the coordinates.

        Parameters:
        -----------
        coords: Array, meters
            The uncompressed coordinates. 

        Returns:
        --------
        coords: Array, meters
            The compressed coordinates. 
        """
        return coords * self.compression[:, None, None]


    def _coordinates(self: ApertureLayer, coords: Array) -> Array:
        """
        Transform the paraxial coordinates into the coordinate
        system of the aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinates of the `Wavefront`. 

        Returns:
        --------
        coords: Array, meters
            The coordinates of the `Aperture`.
        """
        is_trans = (self.centre != np.zeros((2,), float)).any()
        coords: Array = lax.cond(is_trans,
            lambda: self._translate(coords),
            lambda: coords)

        is_compr: bool = (self.compression != np.ones((2,), float)).any()
        coords: Array = lax.cond(is_compr,
            lambda: self._compress(coords),
            lambda: coords)

        is_strain: bool = (self.strain != np.zeros((2,), float)).any()
        coords: Array = lax.cond(is_strain,
            lambda: self._strain(coords),
            lambda: coords)

        is_rot: bool = (self.rotation != 0.)
        coords: Array = lax.cond(is_rot,
            lambda: self._rotate(coords),
            lambda: coords)

        return coords


class DynamicAperture(AbstractDynamicAperture, ABC):
    """
    An abstract class that defines the structure of all the concrete
    apertures. An aperture is represented by an array, usually in the
    range of 0. to 1.. Values in between can be used to represent 
    soft edged apertures and intermediate surfaces. 

    Attributes:
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
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
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    occulting: bool 
    softening: Array
    

    def __init__(self   : ApertureLayer, 
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        The default aperture is dis-allows the learning of all 
        parameters. 

        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre,
            strain = strain,
            compression = compression,
            rotation = rotation)
        self.softening = 1. if softening else 1e32
        self.occulting = bool(occulting)


    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Apply the aperture to an incoming wavefront.

        Parameters:
        -----------
        wavefront: Wavefront
            The wavefront before encountering the aperture.

        Returns:
        --------
        wavefront: Wavefront
            The wavefront after encountering the aperture.
        """
        coords = wavefront.pixel_coordinates
        aperture = self._aperture(coords)
        return wavefront.multiply_amplitude(aperture)


    @abstractmethod
    def _extent(self: ApertureLayer) -> Array: # pragma: no cover
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
        _extent : float
            The maximum distance from centre to edge of aperture
        """


    @abstractmethod
    def _metric(self: ApertureLayer, distances: Array) -> Array: # pragma: no cover
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


    def _soften(self: ApertureLayer, distances: Array) -> Array:
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


    def _aperture(self: ApertureLayer, coords: Array) -> Array:
        """
        Compute the array representing the aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system of the wavefront.

        Returns:
        --------
        aperture: Array 
            The aperture.
        """
        coords: Array = self._coordinates(coords) 
        aperture: Array = self._metric(coords)

        if self.occulting:
            aperture: Array = (1. - aperture)

        return aperture


    def _normalised_coordinates(self: ApertureLayer, 
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
        return self._coordinates(coordinates) / self._extent()


class AnnularAperture(DynamicAperture):
    """
    A circular aperture, parametrised by the number of pixels in
    the array. By default this is a hard edged aperture but may be 
    in future modifed to provide soft edges. 

    Attributes:
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rmax: Array, meters
        Outer radius of aperture.
    rmin: Array, meters
        Inner radius of aperture.
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    """
    rmin: Array
    rmax: Array


    def __init__(self   : ApertureLayer, 
            rmax        : Array, 
            rmin        : Array, 
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters
        ----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
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
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression, 
            occulting = occulting, 
            softening = softening)
        self.rmax = np.asarray(rmax).astype(float)
        self.rmin = np.asarray(rmin).astype(float)


    def _metric(self: ApertureLayer, coords: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metriiic: Array
            The "distance" from the aperture. 
        """
        # TODO: Optimise this slightly by calling hypot directly.
        coords = dLux.utils.cartesian_to_polar(coords)[0]
        return self._soften(coords - self.rmin) * \
            self._soften(- coords + self.rmax)


    def _extent(self: ApertureLayer) -> Array:
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
        _extent : float
            The maximum distance from centre to edge of aperture
        """
        return self.rmax
      
      
class CircularAperture(DynamicAperture):
    """
    A circular aperture represented as a binary array.

    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
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
   
 
    def __init__(self   : ApertureLayer, 
            radius      : Array, 
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            occulting   : bool = False, 
            softening   : bool = False) -> Array:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
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
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression, 
            occulting = occulting, 
            softening = softening)
        self.radius = np.asarray(radius).astype(float)


    def _metric(self: ApertureLayer, coords: Array) -> Array:
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
        # TODO: Optimisation here.
        coords = dLux.utils.cartesian_to_polar(coords)[0]
        return self._soften(- coords + self.radius)


    def _extent(self: ApertureLayer) -> float:
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
        _extent : float
            The maximum distance from centre to edge of aperture
        """
        return self.radius


class RectangularAperture(DynamicAperture):
    """
    A rectangular aperture.

    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
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
    length: float
    width: float


    def __init__(self   : ApertureLayer, 
            length      : Array, 
            width       : Array, 
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer: 
        """
        Parameters
        ----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
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
        super().__init__(
            centre = centre, 
            strain = strain,
            compression = compression,
            rotation = rotation, 
            occulting = occulting, 
            softening = softening)
        self.length = np.asarray(length).astype(float)
        self.width = np.asarray(width).astype(float)


    def _metric(self: ApertureLayer, coords: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        x_mask = self._soften(- np.abs(coords[0]) + self.length / 2.)
        y_mask = self._soften(- np.abs(coords[1]) + self.width / 2.)
        return x_mask * y_mask


    def _extent(self: ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        return np.hypot(self.length / 2., self.width / 2.)


class SquareAperture(DynamicAperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this one tracks less parameters.

    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
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
   
 
    def __init__(self   : ApertureLayer, 
            width       : Array, 
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer: 
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        width: float, meters
            The length of the aperture in the x-direction.
        """
        super().__init__(
            centre = centre, 
            strain = strain,
            compression = compression,
            rotation = rotation, 
            occulting = occulting, 
            softening = softening)
        self.width = np.asarray(width).astype(float)


    def _metric(self: ApertureLayer, coords: Array) -> Array:
        """
        Measures the distance from the edges of the aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinates of the `Wavefront`.

        Returns:
        --------
        metric: Array
            The "distance" from the aperture. 
        """
        x_mask = self._soften(- np.abs(coords[0]) + self.width / 2.)
        y_mask = self._soften(- np.abs(coords[1]) + self.width / 2.)
        return x_mask * y_mask


    def _extent(self: ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        return np.sqrt(2) * self.width / 2.


class PolygonalAperture(DynamicAperture, ABC):
    """
    An abstract class that represents all `PolygonalApertures`.
    The structure here is more than a little strange. Most of 
    the pre-implemented `PolygonalApertures` do **not** inherit
    from `PolygonalAperture`. This is because most of the
    behaviour that is defined by `PolygonalAperture` is related
    to general cases. For apertures, the generality results in 
    a loss of speed. For example, this may be caused because
    a specific symmetry of the shape cannot be exploited. As 
    a result, more optimal implementations could be created 
    directly. Since, the pre-implemented `Aperture` classes 
    that are polygonal share no behaviour with the 
    `PolygonalAperture` it made more sense to separate them 
    out. 
    
    Implementation Notes: A lot of the code that is provided 
    was carefully hand vectorised. In general, where a shape 
    change is applied to an array the new array is given the 
    prefix `bc` standing for "broadcastable".

    Attributes:
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    
    def __init__(self   : ApertureLayer, 
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
    
    
    def _perp_dists_from_lines(
            self: ApertureLayer, 
            m   : float, 
            x1  : float, 
            y1  : float,
            x   : float, 
            y   : float) -> float:
        """
        Calculate the perpendicular distance of a set of points (x, y) from
        a line parametrised by a gradient m and a point (x1, y1). Notice, 
        I am using x and y separately because the instructions cannot be vectorised
        accross them combined. This function can take any number of points.
        
        Parameters:
        -----------
        m: float, None (meters / meter)
            The gradient of the line.
        x1: float, meters
            The x coordinate of a single point that lies on the line.
        y1: float, meters
            The y coordinate of a single point that lies on the line. 
        x: float, meters
            A set of coordinates that you wish to calculate the distance to 
            from the line. 
        y: float, meters
            A set of coordinates that you wish to calculate the distance to 
            from the line. Must have the same dimensions as x.
        
        Returns:
        --------
        dists: float, meters
            The distance of the points (x, y) from the line. Has the same 
            shape as x and y.
        """
        inf_case = (x - x1)
        gen_case = (m * inf_case - (y - y1)) / np.sqrt(1 + m ** 2)
        return np.where(np.isinf(m), inf_case, gen_case)
    
    
    def _grad_from_two_points(
            self: ApertureLayer, 
            xs  : float, 
            ys  : float) -> float:
        """
        Calculate the gradient of the chord that connects two points. 
        Note: This is distinct from `_grads_from_many_points` in that
        it does not wrap arround.
        
        Parameters:
        -----------
        xs: float, meters
            The x coordinates of the points.
        ys: float, meters
            The y coordinates of the points.
            
        Returns:
        --------
        m: float, None (meters / meter)
            The gradient of the chord that connects the two points.
        """
        return (ys[1] - ys[0]) / (xs[1] - xs[0])
    
    
    def _offset(
            self        : ApertureLayer, 
            theta       : float, 
            threshold   : float) -> float:
        """
        Transform the angular range of polar coordinates so that 
        the new lowest angle is offset. The final range should be 
        $[\\phi, \\phi + 2 \\pi]$ where $\\phi$ represents the 
        `threshold`. 
        
        Parameters:
        -----------
        theta: float, radians
            The angular coordinates.
        threshold: float
            The amount to offset the coordinates by.
        
        Returns:
        --------
        theta: float, radians 
            The offset coordinate system.
        """
        comps = (theta < threshold).astype(float)
        return theta + comps * two_pi
    
    
    def _is_orig_left_of_edge(
            self: ApertureLayer, 
            ms  : float, 
            xs  : float, 
            ys  : float) -> int:
        """
        Determines whether the origin is to the left or the right of 
        the edge. The edge(s) in this case are defined by a set of 
        gradients, m and points (xs, ys).
        
        Parameters:
        -----------
        ms: float, None (meters / meter)
            The gradient of the edge(s).
        xs: float, meters
            A set of x coordinates that lie along the edges. 
            Must have the same shape as ms. 
        ys: float, meters
            A set of y coordinates that lie along the edges.
            Must have the same shape as ms.
            
        Returns:
        --------
        is_left: int
            1 if the origin is to the left else -1.
        """
        # NOTE: see class docs.
        bc_orig = np.array([[0.]])
        dist_from_orig = self._perp_dists_from_lines(ms, xs, ys, bc_orig, bc_orig)
        return np.sign(dist_from_orig)
    
    
    def _make_wedges(self: ApertureLayer, off_phi: float, sorted_theta: float) -> float:
        """
        Wedges are used to isolate the space between two vertices in the 
        angular plane. 
        
        Parameters:
        -----------
        off_phi: float, radians
            The angular coordinates that have been correctly offset so 
            that the minimum angle corresponds to the first vertex.
            Note that this particular offset is not unique as any offset
            that is two pi greater will also work.
        sorted_theta: float, radians
            The angles of the vertices sorted from lowest to highest. 
            Implementation Note: The sorting is required for other 
            functions that are typically called together. As a result 
            it has not been internalised. This is a helper function 
            that is not designed to be called in general. This should 
            have the correct shape to be braodcast. This usually involves 
            expanding it to have two extra dimensions. 
            
        Returns:
        --------
        wedges: float
            A stack of binary (float) arrays that represent the angles 
            bounded by each consecutive pair of vertices.
        """
        next_sorted_theta = np.roll(sorted_theta, -1).at[-1].add(two_pi)
        greater_than = (off_phi >= sorted_theta)
        less_than = (off_phi < next_sorted_theta)
        wedges = greater_than & less_than
        return wedges.astype(float)


class IrregularPolygonalAperture(PolygonalAperture):
    """
    The default aperture is dis-allows the learning of all 
    parameters. 

    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    vertices: Array, meters
        The location of the vertices of the aperture.
    """
    vertices: Array
    
    
    def __init__(self   : ApertureLayer, 
            vertices    : Array,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.),
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        vertices: Array, meters
            The location of the vertices of the aperture.
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
        
        vertices = np.array(vertices).astype(float)
        shape = vertices.shape
        is_corr_shape = (shape[0] > shape[1]) and (shape[1] == 2)

        assert is_corr_shape, "Make sure that the vertices are (n, 2)"

        self.vertices = vertices
            
    
    def _grads_from_many_points(self: ApertureLayer, x1: float, y1: float) -> float:
        """
        Given a set of points, calculate the gradient of the line that 
        connects those points. This function assumes that the points are 
        provided in the order they are to be connected together. Notice 
        that we also assume there are more than two points, but more can 
        be provided in which case the shape is assumed to be closed. The 
        output has the same shape as the input and does not check for 
        infinite (vertical) gradients.
        
        Due to the intensly vectorised nature of this code it is ofen 
        necessary to provided the parameters with expanded dimensions. 
        This may be achieved using `x1[:, None, None]` or 
        `x1.reshape((-1, 1, 1))` or `np.expand_dims(x1, (1, 2))`.
        There is no major performance difference between the different
        methods of reshaping. 
        
        Parameters:
        -----------
        x1: float, meters
            The x coordinates of the points that are to be connected. 
        y1: float, meters
            The y coordinates of the points that are to be connected. 
            Must have the same shape as x. 
            
        Returns:
        --------
        ms: float, None (meters / meter)
            The gradients of the lines that connect the vertices. The 
            vertices wrap around to form a closed shape whatever it 
            may look like. 
        """
        x_diffs = x1 - np.roll(x1, -1)
        y_diffs = y1 - np.roll(y1, -1)
        return y_diffs / x_diffs
    
    
    def _extent(self: ApertureLayer) -> float:
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
        extent : float
            The maximum distance from centre to edge of aperture
        """
        verts: float = self.vertices
        dist_to_verts: float = np.hypot(verts[:, 1], verts[:, 0])
        return np.max(dist_to_verts)
    
    
    def _metric(self: ApertureLayer, coords: float) -> float:
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
        # NOTE: see class docs.
        bc_x1 = self.vertices[:, 0][:, None, None]
        bc_y1 = self.vertices[:, 1][:, None, None]

        bc_x = coords[0][None, :, :]
        bc_y = coords[1][None, :, :]

        theta = np.arctan2(bc_y1, bc_x1)
        offset_theta = self._offset(theta, 0.)

        sorted_inds = np.argsort(offset_theta.flatten())

        sorted_x1 = bc_x1[sorted_inds]
        sorted_y1 = bc_y1[sorted_inds]
        sorted_theta = offset_theta[sorted_inds]   
        sorted_m = self._grads_from_many_points(sorted_x1, sorted_y1)

        phi = self._offset(np.arctan2(bc_y, bc_x), sorted_theta[0])

        dist_from_edges = self._perp_dists_from_lines(sorted_m, sorted_x1, sorted_y1, bc_x, bc_y)  
        wedges = self._make_wedges(phi, sorted_theta)
        dist_sgn = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)

        flat_dists = (dist_sgn * dist_from_edges * wedges).sum(axis=0)
        return self._soften(flat_dists)


class RegularPolygonalAperture(PolygonalAperture):
    """
    An optiisation that can be applied to generate
    regular polygonal apertures without using their 
    vertices. 
    
    Attributes:
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    nsides: int
        The number of sides that the aperture has. 
    rmax: float, meters
        The radius of the smallest circle that can completely 
        enclose the aperture. 
    """
    nsides: int
    rmax: float
        
    
    def __init__(self   : ApertureLayer, 
            nsides      : int,
            rmax        : float,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.),
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
        self.nsides = int(nsides)
        self.rmax = np.array(rmax).astype(float)
        
        
    def _extent(self: ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        return self.rmax
        
    
    def _metric(self: ApertureLayer, coords: float) -> float:
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
        x: float = coords[0]
        y: float = coords[1]

        neg_pi_to_pi_phi: float = np.arctan2(y, x) 
        alpha: float = np.pi / self.nsides
            
        i: int = np.arange(self.nsides)[:, None, None] # Dummy index
        bounds: float = 2. * i * alpha
        phi: float = self._offset(neg_pi_to_pi_phi, bounds[0])
            
        wedges: float = self._make_wedges(phi, bounds)
        ms: float = -1 / np.tan(2. * i * alpha + alpha)
        xs: float = self.rmax * np.cos(2. * i * alpha)
        ys: float = self.rmax * np.sin(2. * i * alpha)
        dists: float = self._perp_dists_from_lines(ms, xs, ys, x, y)
        inside: float = self._is_orig_left_of_edge(ms, xs, ys)
         
        dist: float = (inside * dists * wedges)
        return self._soften(dist.sum(axis=0))


class HexagonalAperture(RegularPolygonalAperture):
    """
    Generate a hexagonal aperture, parametrised by rmax. 
   
    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    rmax : float, meters
       The infimum of the radii of the set of circles that fully 
       enclose the hexagonal aperture. In other words the distance 
       from the centre to one of the vertices. 
    """
    rmax : float
    
    
    def __init__(self   : ApertureLayer, 
            rmax        : float,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.),
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            nsides = 6,
            rmax = rmax,
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)

# TODO: See if this code can be used to fix the bugs that 
#       I am encountering with the rotations. I believe 
#       that it will and is probably faster. For now forge
#       ahead. 
#coords: Array = self._rotate(self._translate(coords))
#theta: Array = np.linspace(0, 2 * np.pi, 6, endpoint=False).reshape((6, 1, 1)) + np.pi / 6.
#rmax: float = np.sqrt(3.) / 2. * self.rmax

#m: Array = (-1. / np.tan(theta)).reshape((6, 1, 1))

#x1: Array = (rmax * np.cos(theta)).reshape((6, 1, 1))
#y1: Array = (rmax * np.sin(theta)).reshape((6, 1, 1))

#x: Array = np.tile(coords[0], (6, 1, 1))
#y: Array = np.tile(coords[1], (6, 1, 1))

#dist: Array = (y - y1 - m * (x - x1)) / np.sqrt(1 + m ** 2)
#dist: Array = (1. - 2. * (theta <= np.pi)) * dist
#lines: Array = self._soften(dist)

#return lines.prod(axis=0)


class CompositeAperture(AbstractDynamicAperture):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name.
    
    This class should be used if you want to learn the parameters
    of the entire aperture without learning the individual components.
    This is often going to be useful for pupils with spiders since 
    the connection implies that changes to once are likely to 
    affect one another.

    Attributes:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    apertures: dict
    

    def __init__(self   : ApertureLayer, 
            apertures   : dict,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.)) -> ApertureLayer:
        """
        The default aperture is dis-allows the learning of all 
        parameters. 

        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        apertures : dict
           The aperture objects stored in a dictionary of type
           {str : Aperture} where the Aperture is a subclass of the 
           Aperture.
        """
        super().__init__(
            centre = centre,
            strain = strain, 
            compression = compression,
            rotation = rotation)
        self.apertures = apertures


    def __getitem__(self: ApertureLayer, key: str) -> ApertureLayer:
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


    def __setitem__(self, key: str, value: ApertureLayer) -> None:
        """
        Assign a new value to one of the aperture mirrors.
        Parameters
        ----------
        key : str
           The name of the segement to replace for example "B1-7".
        value : ApertureLayer
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
               wavefront.pixel_coordinates))
        return wavefront


    @abstractmethod
    def _aperture(self: ApertureLayer, coordinates: Array) -> Array: # pragma: no cover
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


class CompoundAperture(CompositeAperture):
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
    
        
    This class should be used if you want to learn the parameters
    of the entire aperture without learning the individual components.
    This is often going to be useful for pupils with spiders since 
    the connection implies that changes to once are likely to 
    affect one another.

    Attributes:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """


    def __init__(
            self        : ApertureLayer,
            apertures   : dict,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.)) -> ApertureLayer:
        """
        Parameters:
        -----------
        apertures: dict(str, Aperture)
           The apertures that make up the compound aperture. 
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(apertures,
            centre = centre,
            strain = strain,
            compression = compression,
            rotation = rotation)
        

    def _aperture(self, coords: Array) -> Array:
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
        coords: float = self._coordinates(coords)
        aps: float = np.stack([ap._aperture(coords) for ap in self.apertures.values()])
        return aps.prod(axis=0)


class MultiAperture(CompositeAperture):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `MultiAperture` is used to represent apertures that are 
    not overlapping. We can add `CompoundAperture`s into 
    `MultiAperture` to create a combination of the two affects.

    Attributes:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """


    def __init__(
            self        : ApertureLayer,
            apertures   : dict,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.)) -> ApertureLayer:
        """
        Parameters:
        -----------
        apertures: dict(str, Aperture)
           The apertures that make up the compound aperture. 
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(apertures,
            centre = centre,
            strain = strain,
            compression = compression,
            rotation = rotation)


    def _aperture(self, coords: Array) -> Array:
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
        coords: float = self._coordinates(coords)
        aps: float = np.stack([ap._aperture(coords) for ap in self.apertures.values()])
        return aps.sum(axis=0)


class Spider(DynamicAperture, ABC):
    """
    An abstraction on the concept of an optical spider for a space telescope.
    These are the things that hold up the secondary mirrors. 

    Attributes:
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    
    
    def __init__(self   : ApertureLayer, 
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.), 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = True,
            softening = softening)
 
 
    def _strut(
            self    : ApertureLayer, 
            angle   : float, 
            coords  : Array) -> Array:
        """
        Generates a representation of a single strut in the spider. This is 
        more complex than you might imagine since the strut can point in 
        any direction. 
 
        Parameters
        ----------
        angle: float, radians
            The angle that this strut points as measured from the positive 
            x-axis in radians. 
 
        Returns
        -------
        strut: float
            The soft edged strut. 
        """
        x, y = coords[0], coords[1]
        perp = np.tan(angle)
        gradient = np.tan(angle)
        dist = np.abs(y - gradient * x) / np.sqrt(1 + gradient ** 2)
        theta = np.arctan2(y, x) + np.pi 
        theta = np.where(theta > angle, theta - angle, theta + 2 * np.pi - angle)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)
        strut = np.where((theta > np.pi / 2.) & (theta < 3. * np.pi / 2.), 1., dist)
        return strut


    def _extent(self: ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        raise NotImplementedError("The `Spider` class and its derivatives" +\
            "are not designed to be used with the `AberatedAperture` class." +\
            "If this is part of a `CompoundAperture` place the " +\
            "`AberratedAperture`s into the `CompoundAperture` not the " +\
            "other way arround.")


class UniformSpider(Spider):
    """
    A spider with equally-spaced, equal-width struts. This is of course the 
    most common and simplest implementation of a spider. Gradients can be 
    taken with respect to the width of the struts and the global rotation 
    as well as the centre of the spider.
 
    Attributes: 
    -----------
    centre: Array, meters
        The (x, y) centre of the coordinate system in the wavefront
        coordinate system.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: Array, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    number_of_struts: int 
        The number of struts to equally space around the circle. This is not 
        a differentiable parameter. 
    width_of_struts: float, meters
        The width of each strut. 
    """
    number_of_struts: int
    width_of_struts: float


    def __init__(self   : ApertureLayer, 
            num_struts  : int,
            strut_width : float,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.),
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters:
        -----------
        centre: Array, meters
            The (x, y) centre of the coordinate system in the wavefront
            coordinate system.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: Array, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        number_of_struts: int 
            The number of struts to equally space around the circle. This is not 
            a differentiable parameter. 
        width_of_struts: float, meters
            The width of each strut. 
        """ 
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            softening = softening)
        self.number_of_struts = int(num_struts)
        self.width_of_struts = np.asarray(strut_width).astype(float)
 
 
    def _metric(self: ApertureLayer, coords: Array) -> Array:
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
        coords: Array
            The paraxial coordinates of the wavefront.

        Returns:
        --------
        non_occ_ap: Array 
            This is essential the final step in processing to produce
            the aperture. What is returned is the non-occulting 
            version of the aperture. 
        """
        coords = self._coordinates(coords)
        angles = np.linspace(0, 2 * np.pi, self.number_of_struts, endpoint=False)
        angles += self.rotation
        struts = np.array([self._strut(angle, coords) for angle in angles]) - self.width_of_struts / 2.
        softened = self._soften(struts)
        return softened.prod(axis=0)


class AberratedAperture(ApertureLayer):
    """
    An class representing an `Aperture` defined
    with a basis. The basis is a set of polynomials that are 
    orthonormal over the surface of the aperture (usually). 
    These can be used to represent any aberation on the surface
    of the aperture. In general, the basis should only be defined 
    on apertures that have a surface such as a mirror or phase 
    plate ect. It isn't really possible to have aberrations on 
    an opening. This rule may be broken to learn the atmosphere 
    above a telescope but whether or not this is a good idea 
    remains to be seen.
 
    Attributes:
    -----------
    basis_funcs: list[callable]
        A list of functions that represent the basis. The exact
        polynomials that are represented will depend on the shape
        of the aperture. 
    aperture: ApertureLayer
        The aperture on which the basis is defined. Must be a 
        subcclass of the `Aperture` class.
    coeffs: list[floats]
        The coefficients of the basis terms. By learning the 
        coefficients only the amount of time that is required 
        for the learning process is significantly reduced.
    """
    basis_funcs: list
    aperture: ApertureLayer
    coeffs: Array
    nterms: int
 
 
    def __init__(self   : ApertureLayer, 
            noll_inds   : Array,
            coeffs      : Array,
            aperture    : ApertureLayer) -> ApertureLayer: 
        """
        Parameters:
        -----------
        noll_inds: List[int]
            The noll indices are a scheme for indexing the Zernike
            polynomials. Normally these polynomials have two 
            indices but the noll indices prevent an order to 
            these pairs. All basis can be indexed using the noll
            indices based on `n` and `m`. 
        aperture: ApertureLayer
            The aperture that the basis is defined on. The shape 
            of this aperture defines what the polynomials are. 
        coeffs: Array
            The coefficients of the basis vectors. 
        """
        assert not aperture.occulting
        assert isinstance(aperture, AbstractDynamicAperture)

        if isinstance(aperture, RegularPolygonalAperture):
            n: int = aperture.nsides
            self.basis_funcs = [self.jth_polike(j, n) for j in noll_inds]
        else:
            self.basis_funcs = [self.jth_zernike(j) for j in noll_inds]

        super().__init__()
        self.aperture = aperture
        self.nterms = int(len(coeffs))
        self.coeffs = np.asarray(coeffs).astype(float)
 
 
    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Apply the aperture and the abberations to the wavefront.  
 
        Parameters:
        -----------
        params: dict
            A dictionary containing the key "Wavefront".
 
        Returns:
        --------
        params: dict 
            A dictionary containing the key "wavefront".
        """
        coords: Array = wavefront.pixel_positions()
        opd: Array = self._opd(coords)
        aperture: Array = self.aperture._aperture(coords)
        return wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)


    def noll_index(self: ApertureLayer, j: int) -> tuple:
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


    def jth_radial_zernike(self: ApertureLayer, n: int, m: int) -> callable:
       """
       The radial zernike polynomial.

       Parameters
       ----------
       n : int
           The first index number of the zernike polynomial to forge
       m : int 
           The second index number of the zernike polynomial to forge.

       Returns
       -------
       radial : Tensor
           An npix by npix stack of radial zernike polynomials.
       """
       MAX_DIFF = 5
       m, n = np.abs(m), np.abs(n)
       upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1

       k = np.arange(MAX_DIFF)
       mask = (k < upper).reshape(MAX_DIFF, 1, 1)
       coefficients = (-1) ** k * dLux.utils.math.factorial(n - k) / \
           (dLux.utils.math.factorial(k) * \
               dLux.utils.math.factorial(((n + m) / 2).astype(int) - k) * \
               dLux.utils.math.factorial(((n - m) / 2).astype(int) - k))

       def _jth_radial_zernike(rho: list) -> list:
           rho = np.tile(rho, (MAX_DIFF, 1, 1))
           coeffs = coefficients.reshape(MAX_DIFF, 1, 1)
           rads = rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1)
           return (coeffs * mask * rads).sum(axis = 0)
               
       return _jth_radial_zernike


    def jth_polar_zernike(self: ApertureLayer, n: int, m: int) -> callable:
       """
       Generates a function representing the polar component 
       of the jth Zernike polynomial.

       Parameters:
       -----------
       n: int 
           The first index number of the Zernike polynomial.
       m: int 
           The second index number of the Zernike polynomials.

       Returns:
       --------
       polar: Array
           The polar component of the jth Zernike polynomials.
       """
       is_m_zero = (m != 0).astype(int)
       norm_coeff = (1 + (np.sqrt(2) - 1) * is_m_zero) * np.sqrt(n + 1)

       # When m < 0 we have the odd zernike polynomials which are 
       # the radial zernike polynomials multiplied by a sine term.
       # When m > 0 we have the even sernike polynomials which are 
       # the radial polynomials multiplies by a cosine term. 
       # To produce this result without logic we can use the fact
       # that sine and cosine are separated by a phase of pi / 2
       # hence by casting int(m < 0) we can add the nessecary phase.

       phase_mod = (m < 0).astype(int) * np.pi / 2

       def _jth_polar_zernike(theta: list) -> list:
           return norm_coeff * np.cos(np.abs(m) * theta - phase_mod)

       return _jth_polar_zernike  


    def jth_zernike(self: ApertureLayer, j: int) -> callable:
        """
        Calculate the zernike basis on a square pixel grid. 
     
        Parameters
        ----------
        noll_index: int
            The noll index corresponding to the zernike to generate.
            The first ten zernikes have been computed analytically 
            and are available via the `PreCompZernikeBasis` class. 
            This is only for doing zernike terms that are of higher 
            order and not centered.
     
        Returns
        -------
        zernike : Tensor 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        n, m = self.noll_index(j)
     
        def _jth_zernike(coords: list) -> list:
            polar_coords = dLux.utils.cartesian_to_polar(coords)
            rho = polar_coords[0]
            theta = polar_coords[1]
            aperture = rho <= 1.
            _jth_rad_zern = self.jth_radial_zernike(n, m)
            _jth_pol_zern = self.jth_polar_zernike(n, m)
            return aperture * _jth_rad_zern(rho) * _jth_pol_zern(theta)
        
        return _jth_zernike 


    def jth_polike(self: ApertureLayer, j: int, n: int) -> callable:
        """
        The jth polike as a function. 
     
        Parameters:
        -----------
        j: int
            The noll index of the requested zernike.
        n: int
            The number of sides on the regular polygon.
     
        Returns:
        --------
        hexike: callable
            A function representing the jth hexike that is evaluated 
            on a cartesian coordinate grid. 
        """
        _jth_zernike = self.jth_zernike(j)
     
        def _jth_polike(coords: Array) -> Array:
            polar: float = dLux.utils.cartesian_to_polar(coords)
            rho: float = polar[0]
            alpha: float = np.pi / n
            phi: float = polar[1] + alpha 
            wedge: float = np.floor((phi + alpha) / (2. * alpha))
            u_alpha: float = phi - wedge * (2 * alpha)
            r_alpha: float = np.cos(alpha) / np.cos(u_alpha)
            return 1 / r_alpha * _jth_zernike(coords / r_alpha)
     
        return _jth_polike
 
 
    def _basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 
 
        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. Normally the basis is 
            cropped to be just on the aperture however, this 
            step is not necessary except for in visualisation. 
            It has been removed to save some time in the 
            calculations. 
        """
        coords: float = self.aperture._normalised_coordinates(coords)
        ikes: float = np.stack([h(coords) for h in self.basis_funcs])
        
        is_reg_pol: bool = isinstance(self.aperture, RegularPolygonalAperture)
        is_circ: bool = isinstance(self.aperture, CircularAperture)

        if is_circ or is_reg_pol:
            return ikes

        aperture: float = self.aperture._aperture(coords)
        ikes: float = self._orthonormalise(aperture, ikes)

        return ikes 
 

    def _opd(self: ApertureLayer, coords: Array) -> Array:
        """
        Calculate the optical path difference that is caused 
        by the basis and the aberations that it represents. 
 
        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 
 
        Returns:
        --------
        opd: Array
            The optical path difference associated with much of 
            the path. 
        """
        basis: Array = self._basis(coords)
        opd: Array = np.dot(basis.T, self.coeffs)
        return opd


    def _orthonormalise(self: ApertureLayer, 
            aperture: Array, 
            zernikes: Array) -> Array:
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
        shape = zernikes.shape
        width = shape[-1]
        basis = np.zeros(shape).at[0].set(aperture)
 
        for j in np.arange(1, self.nterms):
            intermediate = zernikes[j] * aperture
            coefficient = np.zeros((self.nterms, 1, 1), dtype=float)
            mask = (np.arange(1, self.nterms) > j + 1).reshape((-1, 1, 1))
 
            coefficient = -1 / pixel_area * \
                (zernikes[j] * basis[1:] * aperture * mask)\
                .sum(axis = (1, 2))\
                .reshape(-1, 1, 1) 

            intermediate += (coefficient * basis[1:] * mask).sum(axis = 0)
            
            basis = basis\
                .at[j]\
                .set(intermediate / \
                    np.sqrt((intermediate ** 2).sum() / pixel_area))
        
        return basis


class StaticAperture(ApertureLayer):
    """
    This layer is designed to increase the speed, when parameters 
    are not getting learned. It pre-calculates the aperture array 
    which is stored and then simply applies it. 

    Attributes:
    -----------
    aperture: Array
        The aperture represented as an array.
    """
    aperture: Array


    def __init__(
            self        : ApertureLayer, 
            aperture    : ApertureLayer, 
            npix        : int, 
            pix_scale   : float) -> ApertureLayer:
        """
        Parameters:
        -----------
        aperture: ApertureLayer
            An instance of DynamicAperture. 
        npix: int
            The number of pixels used to represent the wavefront 
            coordinate system.
        pixel_scale: float, meters / pixel
            The pixel scale of the wavefront coordinate system.
        """
        assert isinstance(aperture, DynamicAperture)

        super().__init__()
        coords: float = dLux.utils.get_pixel_coordinates(npix, pix_scale)
        self.aperture: float = aperture._aperture(coords)


    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Apply the aperture to the wavefront.

        Parameters:
        -----------
        wavefront: Wavefront
            The wavefront that is passing through the aperture.

        Returns:
        --------
        wavefront: Wavefront
            The wavefront after passing through the aperture
        """
        return wavefront.multiply_amplitude(self.aperture)
    

class StaticAberratedAperture(ApertureLayer):
    """
    This layer is designed to increase the speed, when parameters 
    are not getting learned. It pre-calculates the aperture and
    basis arrays which is stored and then applied. 

    Attributes:
    -----------
    aperture: Array
        The aperture represented as an array.
    basis: Array 
        The basis represented as an array.
    """
    aperture: Array
    basis: Array


    def __init__(
            self        : ApertureLayer, 
            aperture    : ApertureLayer, 
            npix        : int, 
            pix_scale   : float) -> ApertureLayer:
        """
        Parameters:
        -----------
        aperture: ApertureLayer
            An instance of DynamicAperture. 
        npix: int
            The number of pixels used to represent the wavefront 
            coordinate system.
        pixel_scale: float, meters / pixel
            The pixel scale of the wavefront coordinate system.
        """
        assert isinstance(aperture, AberratedAperture)

        super().__init__()
        coords: float = dLux.utils.get_pixel_coordinates(npix, pix_scale)
        self.aperture: float = aperture.aperture._aperture(coords)
        self.basis: float = aperture._basis(coords)


    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Apply the aperture to the wavefront.

        Parameters:
        -----------
        wavefront: Wavefront
            The wavefront that is passing through the aperture.

        Returns:
        --------
        wavefront: Wavefront
            The wavefront after passing through the aperture
        """
        return wavefront\
            .multiply_amplitude(self.aperture)\
            .add_phase(self.basis)


