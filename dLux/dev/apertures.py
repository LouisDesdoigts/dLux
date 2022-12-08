import equinox as eqx
import matplotlib as mpl 
import matplotlib.pyplot as plt
import jax.numpy as np
import jax 
import dLux
import abc
import functools
from typing import TypeVar, Int


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront


__all__ = ["Aperture", "CompoundAperture", "SquareAperture", 
    "RectangularAperture", "CircularAperture", "AnnularAperture",
    "MultiAperture", "RotatableAperture", "HexagonalAperture"]


def test_plots_of_aps(aps: dict) -> None:
    """
    A formalisation of the common testing routine that I have
    been using. This will be removed from the production code. 

    Parameters:
    -----------
    aps: dict
        The apertures with descriptive titles.
    """
    npix = 128
    width = 2.
    coords = dLux.utils.get_pixel_coordinates(npix, width / npix)
    fig, axes = plt.subplots(1, len(aps))
    for i, ap in enumerate(aps):
        axes[i].set_title(ap)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        _map = axes[i].imshow(aps[ap]._aperture(coords))
        fig.colorbar(_map, ax=axes[i])

    plt.show()


class ApertureLayer(dLux.optics.OpticalLayer, abc.ABC):
    """
    The ApertureLayer groups together all of the functionality 
    that is associated with the apertures. Very little of this
    functionality is actually implemented by itself because 
    implementation varies amongst the subclasses. It is a 
    layer within the class Heirachy that exists almost purely 
    as a classification.

    Parameters:
    -----------
    name: String
        The address of this ApertureLayer within the optical 
        system. 
    """

    
    def __init__(self: dLux.optics.OpticalLayer) -> dLux.optics.OpticalLayer:
        """
        Automatically assigns the name of the layer to be the 
        class name. 
        """
        self.name = self.__class__.__name__


class AbstractDynamicAperture(ApertureLayer, abc.ABC):
    """
    AbstractDynamicAperture:
    ------------------------
    This class also provides a level of classification without 
    implementing any functionality. It was created so that 
    `DynamicAberratedAperture` and `DynamicAperture` could 
    inherit from a common base. 
    """
    # Now I am regretting this structure and think that I will 
    # just go with another structure that does make use 
    # of mutliple inheritance 
    #
    # The thinking is that we have `StaticAperture`, `ShapedAperture` and 
    # `AberratedAperture` then we can combine these as we wish to the 
    # correct affect
    #
    # Let's just stick to the plan. 


class DynamicAperture(AbstractDynamicAperture, abc.ABC):
    """
    An abstract class that defines the structure of all the concrete
    apertures. An aperture is represented by an array, usually in the
    range of 0. to 1.. Values in between can be used to represent 
    soft edged apertures and intermediate surfaces. 

    Attributes
    ----------
    centre: float, meters
        The x coordinate of the centre of the aperture.
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
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    occulting: bool 
    softening: Array
    centre: Array
    strain: Array
    compression: Array
    rotation: Array
    

    def __init__(self   : ApertureLayer, 
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> Aperture:
        """
        The default aperture is dis-allows the learning of all 
        parameters. 

        Parameters
        ----------
        centre: float, meters
            The centre of the coordinate system along the x-axis.
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
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__()
        self.centre = np.asarray(centre).astype(float)
        self.strain = np.asarray(strain).astype(float)
        self.compression = np.asarray(compression).astype(float)
        self.rotation = np.asarray(rotation).astype(float)
        self.softening = 1. if softening else np.inf
        self.occulting = bool(occulting)


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
    def _extent(self: ApertureLayer) -> Array:
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


    def _rotate(self: ApertureLayer, coords: Array) -> Array:
        """
        Rotate the coordinate system by a pre-specified amount,
        `self._theta`

        Parameters
        ----------
        coords : Array
            A `(2, npix, npix)` representation of the coordinate 
            system. The leading dimensions specifies the x and then 
            the y coordinates in that order. 

        Returns
        -------
        coordinates : Array
            The rotated coordinate system. 
        """
        x, y = coords[0], coords[1]
        new_x = np.cos(self.rotation) * x + np.sin(self.rotation) * y
        new_y = -np.sin(self.rotation) * x + np.cos(self.rotation) * y
        return np.array([new_x, new_y])


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
        return coordinates - self.centre.reshape(2, 1, 1)


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
        return coords + trans_coords * self.strain.reshape(2, 1, 1)


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
        return coords * self.compression.reshape(2, 1, 1)


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
        coords: Array = jax.lax.cond(is_trans,
            lambda: self._translate(coords),
            lambda: coords)

        is_compr: bool = (self.compression != np.ones((2,), float)).any()
        coords: Array = jax.lax.cond(is_compr,
            lambda: self._compress(coords),
            lambda: coords)

        is_strain: bool = (self.strain != np.zeros((2,), float)).any()
        coords: Array = jax.lax.cond(is_strain,
            lambda: self._strain(coords),
            lambda: coords)

        is_rot: bool = (self.rotation != 0.)
        coords: Array = jax.lax.cond(is_rot,
            lambda: self._rotate(coords),
            lambda: coords)

        return coords


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


    def _normalised_coordinates(self: Aperture, 
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

    Attributes
    ----------
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
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
        centre: float, meters
            The centre of the coordinate system in the paraxial 
            coordinates.
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
      
      
test_plots_of_aps({
    "Occ. Soft": AnnularAperture(1., .5, occulting=True, softening=True),
    "Occ. Hard": AnnularAperture(1., .5, occulting=True),
    "Soft": AnnularAperture(1., .5, softening=True),
    "Hard": AnnularAperture(1., .5),
    "Trans.": AnnularAperture(1., .5, centre=[.5, .5]),
    "Strain": AnnularAperture(1., .5, strain=[.5, 0.]),
    "Compr.": AnnularAperture(1., .5, compression=[.5, 1.])
})

class CircularAperture(DynamicAperture):
    """
    A circular aperture represented as a binary array.

    Parameters
    ----------
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
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
        Parameters
        ----------
        centre: float, meters
            The centre of the coordinate system in the paraxial coordinates.
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


test_plots_of_aps({
    "Occ. Soft": CircularAperture(1., occulting=True, softening=True),
    "Occ. Hard": CircularAperture(1., occulting=True),
    "Soft": CircularAperture(1., softening=True),
    "Hard": CircularAperture(1.),
    "Trans.": CircularAperture(1., centre=[.5, .5]),
    "Strain": CircularAperture(1., strain=[.5, 0.]),
    "Compr.": CircularAperture(1., compression=[.5, 1.])
})

class RectangularAperture(DynamicAperture):
    """
    A rectangular aperture.

    Parameters
    ----------
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
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
        centre: float, meters
            The centre of the coordinate system in the paraxial coordinates.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
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

test_plots_of_aps({
    "Occ. Soft": RectangularAperture(1., .5, occulting=True, softening=True),
    "Occ. Hard": RectangularAperture(1., .5, occulting=True),
    "Soft": RectangularAperture(1., .5, softening=True),
    "Hard": RectangularAperture(1., .5),
    "Trans.": RectangularAperture(1., .5, centre=[.5, .5]),
    "Strain": RectangularAperture(1., .5, strain=[.5, 0.]),
    "Compr.": RectangularAperture(1., .5, compression=[.5, 1.]),
    "Rot.": RectangularAperture(1., .5, rotation=np.pi / 4.)
})

class SquareAperture(DynamicAperture):
    """
    A square aperture. Note: this can also be created from the rectangular 
    aperture class, but this one tracks less parameters.

    Parameters
    ----------
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
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
        Parameters
        ----------
        centre: float, meters
            The centre of the coordinate system in the paraxial coordinates.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
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

test_plots_of_aps({
   "Occ. Soft": SquareAperture(1., occulting=True, softening=True),
   "Occ. Hard": SquareAperture(1., occulting=True),
   "Soft": SquareAperture(1., softening=True),
   "Hard": SquareAperture(1.),
   "Trans.": SquareAperture(1., centre=[.5, .5]),
   "Strain": SquareAperture(1., strain=[.5, 0.]),
   "Compr.": SquareAperture(1., compression=[.5, 1.]),
   "Rot.": SquareAperture(1., rotation=np.pi / 4.)
})


class PolygonalAperture(DynamicAperture, abc.ABC):
    """
    A general representation of a polygonal aperture. 
    This class defines some useful parameters for working
    with the lines and points, that were not needed above where 
    additional optimisations where necessary.

    Parameters:
    -----------
    nsides: Int
        The number of sides.
    rmax: Float
        The radius of the smallest circle that can fully contain the 
        aperture. 
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    """


    def _perp_dist_from_line(
            self    : ApertureLayer, 
            point   : Array, 
            grad    : Array, 
            coords  : Array) -> Array:
        """
        Calculate the distance from a line parametrised by a
        gradient and a point. The mathematical formula for the
        line based on this parametrisation is,

            y - y1 = m(x - x1) (1)

        where m is the gradient and (x1, y1) is the point. The 
        distance perpendicular from the line works out to be,

            d = (y - y1 - m(x - x1)) / (1 + m**2) (2)

        Parameters:
        -----------
        point: Array, meters
            The location of the point that lines on the line and 
            partialy defines it. That is (x1, y1) see (1).
        grad: Array, None (meters / meter)
            The gradient of the line.
        coords: Array, meters
            The point (x, y) in (2). This can be a two dimensional
            array for speed.

        Returns:
        --------
        dist: Array, meters
            The distance of each point in coords from the line 
            given by eq (1)
        """
        x, y = coords[0], coords[1]
        x1, y1 = point[0], point[1]
        return (y - y1 - grad * (x - x1)) / np.sqrt(1 + grad ** 2)


    def _grad_from_two_points(
            self    : Layer,
            point_1 : Array,
            point_2 : Array)-> Array:
        """
        A convinient helper function that calculates the 
        gradient of a chord connecting two points. The formula 
        that is used is,

            m = (y2 - y1) / (x2 - x1) (1)

        Parameters:
        -----------
        point_1: Array, meters
            (x1, y1) in eq (1)
        point_2: Array, meters
            (x2, y2) in eq (2)

        Returns:
        --------
        m: Array
            The gradient of the line connecting (x1, y1) and 
            (x2, y2).
        """
        x1, y1 = point_1[0], point_1[1]
        x2, y2 = point_2[0], point_2[1]
        return (y2 - y1) / (x2 - x1)


# TODO: Implement PolygonalAperture as the abstract base class 
#       with the subclasses RegularPolygonalAperture and 
#       IrregularPolygonalAperture.
class RegularPolygonalAperture(PolygonalAperture):
    """
    A general representation of a polygonal aperture. 
    Each side of the aperture should be the same length. There
    are some pre-existing implementations for some of the more 
    common cases. This is designed for the exceptions that are 
    less common. 

    Par
    -----------
    nsides: Int
        The number of sides.
    rmax: Float
        The radius of the smallest circle that can fully contain the 
        aperture. 
    centre: float, meters
        The centre of the coordinate system in the paraxial coordinates.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    """
    nsides: Int
    rmax: Float


    def __init__(
            self        : ApertureLayer,
            rmax        : Array,
            nsides      : Int,
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer: 
        """
        Parameters:
        -----------
        nsides: Int
            The number of sides.
        rmax: Float
            The radius of the smallest circle that can fully contain the 
            aperture. 
        centre: float, meters
            The centre of the coordinate system in the paraxial coordinates.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
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
            rotation = rotation,
            occulting = occulting, 
            softening = softening)
        self.rmax = np.asarray(rmax).astype(float)
        self.nsides = int(nsides)

    `
    # TODO: Work out some clever way of doing this so that the vertical 
    #       gradient problem is not well, a problem. I think that this 
    #       will have to be done using a `lax.cond`.
    def _aperture(self: ApertureLayer, coords: Array) -> Array:
        """
        Generates a regular polygon with nsides. The zero rotation 
        of the aperture avoids vertical sides as this creates infinite 
        gradients. 

        Parameters:
        -----------
        coords: Array, meters
            The aperture coordinates (already transformed from the 
            wavefront coordinates).
            
        Returns:
        --------
        aperture: Array
            The aperture as an array of values.

        """
        n: int = self.nsides # Abrreviation for line length 
        theta: Array = np.linspace(0., 2. * np.pi, n, endpoint=False)

        m: Array = (-1. / np.tan(theta)).reshape((n, 1, 1))
        x1: Array = (rmax * np.cos(theta)).reshape((n, 1, 1))
        y1: Array = (rmax * np.sin(theta)).reshape((n, 1, 1))

        dist: Array = (y - y1 - m * (x - x1)) / np.sqrt(1 + m ** 2)
        dist: Array = (1. - 2. * (theta <= np.pi)) * dist

        return self._soften(dist)
        

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


    def _extent(self: Aperture) -> float:
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
import jax 
import abc 
import jax.numpy as np
import equinox as eqx
import dLux as dl
import typing 
from typing import List

__all__ = ["AberratedCircularAperture", "AberratedHexagonalAperture"]

Array = typing.TypeVar("Array")
Layer = typing.TypeVar("Layer")
Aperture = typing.TypeVar("Aperture")
CircularAperture = typing.TypeVar("CircularAperture")
HexagonalAperture = typing.TypeVar("HexagonalAperture")


zernikes: list = [
    lambda rho, theta: np.ones(rho.shape, dtype=float),
    lambda rho, theta: 2. * rho * np.sin(theta),
    lambda rho, theta: 2. * rho * np.cos(theta),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.sin(2. * theta),
    lambda rho, theta: np.sqrt(3.) * (2. * rho ** 2 - 1.),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.cos(2. * theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.sin(3. * theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.cos(3. * theta)
]


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


def noll_index(j: int) -> tuple:
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


def jth_radial_zernike(n: int, m: int) -> list:
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
    coefficients = (-1) ** k * factorial(n - k) / \
        (factorial(k) * \
            factorial(((n + m) / 2).astype(int) - k) * \
            factorial(((n - m) / 2).astype(int) - k))

    def _jth_radial_zernike(rho: list) -> list:
        rho = np.tile(rho, (MAX_DIFF, 1, 1))
        coeffs = coefficients.reshape(MAX_DIFF, 1, 1)
        rads = rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1)
        return (coeffs * mask * rads).sum(axis = 0)
            
    return _jth_radial_zernike


def jth_polar_zernike(n: int, m: int) -> list:
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


def jth_zernike(j: int) -> list:
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
    n, m = noll_index(j)

    def _jth_zernike(coords: list) -> list:
        polar_coords = dl.utils.cartesian_to_polar(coords)
        rho = polar_coords[0]
        theta = polar_coords[1]
        aperture = rho <= 1.
        _jth_rad_zern = jth_radial_zernike(n, m)
        _jth_pol_zern = jth_polar_zernike(n, m)
        return aperture * _jth_rad_zern(rho) * _jth_pol_zern(theta)
    
    return _jth_zernike 


# So the current problem is that I need to find some way of passing 
# rmax into the hexike dynamically (do I). Haha just worked it out,
# I normalise the corrdinates first hence I don't need rmax. 
def jth_hexike(j: int) -> callable:
    """
    The jth Hexike as a function. 

    Parameters:
    -----------
    j: int
        The noll index of the requested zernike. 

    Returns:
    --------
    hexike: callable
        A function representing the jth hexike that is evaluated 
        on a cartesian coordinate grid. 
    """
    _jth_zernike = jth_zernike(j)

    def _jth_hexike(coords: Array) -> Array:
        polar = dl.utils.cartesian_to_polar(coords)
        rho, phi = polar[0], polar[1]
        phi = phi + np.pi / 2.
        alpha = np.pi / 6.
        wedge = np.floor((phi + alpha) / (2 * alpha))
        u_alpha = phi - wedge * (2 * alpha)
        r_alpha = np.cos(alpha) / np.cos(u_alpha)

#        plt.title("$R_{\\alpha}$")
#        plt.imshow(r_alpha)
#        plt.colorbar()
#        plt.show()

        return 1 / r_alpha * _jth_zernike(coords / r_alpha)

    return _jth_hexike


class AberratedAperture(eqx.Module, abc.ABC):
    """
    An abstract base class representing an `Aperture` defined
    with a basis. The basis is a set of polynomials that are 
    orthonormal over the surface of the aperture (usually). 
    These can be used to represent any aberation on the surface
    of the aperture. In general, the basis should only be defined 
    on apertures that have a surface such as a mirror or phase 
    plate ect. It isn't really possible to have aberrations on 
    an opening. This rule may be broken to learn the atmosphere 
    above a telescope but whether or not this is a good idea 
    remains to be seen.

    Parameters:
    -----------
    basis_funcs: list[callable]
        A list of functions that represent the basis. The exact
        polynomials that are represented will depend on the shape
        of the aperture. 
    aperture: Aperture
        The aperture on which the basis is defined. Must be a 
        subclass of the `Aperture` class.
    coeffs: list[floats]
        The coefficients of the basis terms. By learning the 
        coefficients only the amount of time that is required 
        for the learning process is significantly reduced.
    """
    basis_funcs: list
    aperture: Aperture
    coeffs: Array


    def __init__(self   : Layer, 
            coeffs      : Array,
            aperture    : Aperture) -> Layer: 
        """
        Parameters:
        -----------
        noll_inds: List[int]
            The noll indices are a scheme for indexing the Zernike
            polynomials. Normally these polynomials have two 
            indices but the noll indices prevent an order to 
            these pairs. All basis can be indexed using the noll
            indices based on `n` and `m`. 
        aperture: Aperture
            The aperture that the basis is defined on. The shape 
            of this aperture defines what the polynomials are. 
        coeffs: Array
            The coefficients of the basis vectors. 
        """
        assert not aperture.occulting
        assert isinstance(aperture, dl.Aperture)
        self.aperture = aperture
        self.coeffs = np.asarray(coeffs).astype(float)


    def _basis(self: Layer, coords: Array) -> Array:
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
        coords: Array = self.aperture._normalised_coordinates(coords)
        return np.stack([h(coords) for h in self.basis_funcs])


    def _opd(self: Layer, coords: Array) -> Array:
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


    def __call__(self, params_dict: dict) -> dict:
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
        wavefront: object = params_dict["Wavefront"]
        coords: Array = wavefront.pixel_positions()
        opd: Array = self._opd(coords)
        aperture: Array = self.aperture._aperture(coords)
        params_dict["Wavefront"] = wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)
        return params_dict


class AberratedCircularAperture(AberratedAperture):
    """
    Parameters:
    -----------
    zernikes: Array
        An array of `jit` compiled zernike basis functions 
        that operate on a set of coordinates. In particular 
        these coordinates correspond to a normalised set 
        of coordinates that are centered at the the centre 
        of the circular aperture with 1. occuring along the 
        radius. 
    coeffs: Array
        The coefficients of the Zernike terms. 
    aperture: Layer
        Must be an instance of `CircularAperture`. This 
        is applied alongside the basis. 
    """


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : CircularAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: CircularAperture
            A `CircularAperture` within which the aberrations are 
            being studied. 
        """
        self.basis_funcs = [jth_zernike(ind) for ind in noll_inds]
        super().__init__(coeffs, aperture)

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.CircularAperture)


class AberratedHexagonalAperture(AberratedAperture):
    """
    Parameters:
    -----------
    Hexikes: Array
        An array of `jit` compiled hexike basis functions 
        that operate on a set of coordinates. In particular 
        these coordinates correspond to a normalised set 
        of coordinates that are centered at the the centre 
        of the circular aperture with 1. occuring along the 
        radius. 
    coeffs: Array
        The coefficients of the Hexike terms. 
    aperture: Layer
        Must be an instance of `HexagonalAperture`. This 
        is applied alongside the basis. 
    """


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : HexagonalAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: HexagonalAperture
            A `HexagonalAperture` within which the aberrations are 
            being studied. 
        """

        self.basis_funcs = [jth_hexike(j) for j in noll_inds]
        super().__init__(coeffs, aperture)

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.HexagonalAperture)


class AberratedArbitraryAperture(AberratedAperture):
    """
    This class is an alternative form of generating a 
    basis over an aperture of any shape. Although not 
    incredibly slow, it is slower than the other methods
    but does not have the shortcomings of numerical 
    instability. It is recomended that this method is 
    used with the `StaticBasis` class.

    Parameters:
    -----------
    basis_funcs: list
        A list of `callable` functions that can be used 
        to produce the basis. 
    coeffs: Array
        The coefficients of the Hexike terms. 
    aperture: Layer
        Must be an instance of `HexagonalAperture`. This 
        is applied alongside the basis. 
    """
    nterms: int


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : HexagonalAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: HexagonalAperture
            A `HexagonalAperture` within which the aberrations are 
            being studied. 
        """

        self.basis_funcs = [jth_zernike(j) for j in noll_inds]
        super().__init__(coeffs, aperture)
        self.nterms = len(noll_inds)

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.Aperture)


    def _orthonormalise(self: Layer, 
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


    def _basis(self: Layer, coords: Array) -> Array:
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
        zern_coords = self.aperture._normalised_coordinates(coords)
        zernikes = np.stack([h(zern_coords) for h in self.basis_funcs])
        aperture = self.aperture._aperture(coords)
        return self._orthonormalise(aperture, zernikes)


class MultiAberratedAperture(eqx.Module):
    """
    This is for disjoint apertures that have multiple components. 
    For example, the James Webb Space Telescope and the Heimdellr
    array. 

    Parameters:
    -----------
    aperture: MutliAperture
        The aperture over which to generate each of the basis. 
    bases: List[Layer]
        A list of `AberratedAperture` objects.
    """
    aperture: Layer
    bases: List[Layer]


    def __init__(self   : Layer, 
            noll_inds   : Array, 
            coeffs      : Array,
            aperture    : Layer) -> Layer: 
        """
        Parameters:
        -----------
        aperture: Layer
            A `MultiAperture` over which the basis will be generated. 
            Each `Aperture` in the `MultiAperture` will be bequeathed
            it's own basis. 
        coeffs: Array
            The coefficients of the basis terms in each aperture.
            The coefficients should be a matrix that is 
            `(nterms, napps)`.
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        """
        bases = []
        for ap, coeff in zip(aperture.to_list(), coeffs):
            if isinstance(ap, dl.HexagonalAperture):
                bases.append(AberratedHexagonalAperture(noll_inds, coeff, ap))
            elif isinstance(ap, dl.CircularAperture):
                bases.append(AberratedCircularAperture(noll_inds, coeff, ap))
            else:
                bases.append(AberratedArbitraryAperture(noll_inds, coeff, ap))

        self.aperture = aperture
        self.bases = bases
        assert isinstance(self.aperture, dl.MultiAperture)


    def _basis(self: Layer, coords: Array) -> Array:
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
        return np.stack([b._basis(coords) for b in self.bases])


    def _opd(self: Layer, coords: Array) -> Array:
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


    def __call__(self, params_dict: dict) -> dict:
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
        wavefront: object = params_dict["Wavefront"]
        coords: Array = wavefront.pixel_positions()
        opd: Array = self._opd(coords)
        aperture: Array = self.aperture._aperture(coords)
        params_dict["Wavefront"] = wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)
        return params_dict


# TODO: I should pre-calculate the _aperture in the init for the 
# AberratedCircularAperture and the AberratedHexagonalAperture
# This is so that I can add a note.
# This is testing code. 
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "seismic"

pixels = 128
nterms = 6

coordinates = dl.utils.get_pixel_coordinates(pixels, 3. / pixels)

num_ikes = 10
num_basis = 3
noll_inds = [i + 1 for i in range(num_ikes)]

# Testing the AberratedCircularAperture specifically
#coeffs = np.ones((num_ikes,), float)
#aper = dl.CircularAperture(1., 0., .5, False, False)
#basis = AberratedCircularAperture(noll_inds, coeffs, aper)
#
#_basis = basis._basis(coordinates)
#_aper = aper._aperture(coordinates)
#
#fig, axes = plt.subplots(2, 5)
#for i in range(num_ikes):
#    col = i % (num_ikes // 2)
#    row = i // (num_ikes // 2)
#
#    axes[row][col].set_title(noll_inds[i])
#    _map = axes[row][col].imshow(_basis[i] * _aper)
#    axes[row][col].set_xticks([])
#    axes[row][col].set_yticks([])
#    axes[row][col].axis("off")
#    fig.colorbar(_map, ax=axes[row][col])
#plt.show()

# So now I need to test the MultiAberratedAperture 
aps = {
    "Right Circ.": dl.CircularAperture(-1., 0., .5, False, False),
    "Centre Hex.": dl.HexagonalAperture(0., 0., 0., .5, False, False),
    "Left Rect.":  dl.SquareAperture(1., 0., 0., 1., False, False)
}

coeffs = np.ones((3, num_ikes), float) 
aper = dl.MultiAperture(aps)
basis = MultiAberratedAperture(noll_inds, coeffs, aper)

_aper = aper._aperture(coordinates)

plt.title("MultiAperture")
plt.imshow(_aper)
plt.colorbar()
plt.show()

_basis = basis._basis(coordinates)
_comb_basis = _basis.sum(axis=0)

fig, axes = plt.subplots(2, 5)
for i in range(num_ikes):
    col = i % (num_ikes // 2)
    row = i // (num_ikes // 2)

    axes[row][col].set_title(noll_inds[i])
    _map = axes[row][col].imshow(_comb_basis[i] * _aper)
    axes[row][col].set_xticks([])
    axes[row][col].set_yticks([])
    fig.colorbar(_map, ax=axes[row][col])
plt.show()

fig = plt.figure()
figs = fig.subfigures(num_basis, 1)
for _basis, _fig, _aper in zip(basis.bases, figs, aper.apertures):
    _fig.suptitle(f"{_basis.__class__.__name__}")
    __basis = _basis._basis(coordinates)
    __aper = aper[_aper]._aperture(coordinates)
    axes = _fig.subplots(2, (num_ikes // 2))

    for i in range(num_ikes):
        col = i % (num_ikes // 2)
        row = i // (num_ikes // 2)

        axes[row][col].set_title(noll_inds[i])
        _map = axes[row][col].imshow(__basis[i] * __aper, vmin=-2, vmax=2)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        _fig.colorbar(_map, ax=axes[row][col])

    # TODO: I need some better way of interfacing with these 
    # things. I think that I need to use an `OrderedDict` or 
    # something along those lines. I could implement this 
    # myself of course. 
plt.show()


# So the goal here is to perform the tests for all the apertures using the 
# `AberratedArbitraryAperture`. 
#aps = {
#    "Sq. Ap.": dl.SquareAperture(0., 0., 0., 1., False, False),
#    "Ann. Ap.": dl.AnnularAperture(0., 0., 1., .5, False, False),
#    "Rect. Ap.": dl.RectangularAperture(0., 0., 0., .5, 1., False, False),
#    "Hex. Ap.": dl.HexagonalAperture(0., 0., 0., 1., False, False)
#}
#
#coeffs = np.ones((num_ikes,), dtype=float)
#bases = {
#    "Squarikes": AberratedArbitraryAperture(noll_inds, coeffs, aps["Sq. Ap."]),
#    "Annikes": AberratedArbitraryAperture(noll_inds, coeffs, aps["Ann. Ap."]),
#    "Rectikes": AberratedArbitraryAperture(noll_inds, coeffs, aps["Rect. Ap."]),
#    "Hexikes": AberratedArbitraryAperture(noll_inds, coeffs, aps["Hex. Ap."])
#}
#
#figure = plt.figure()
#figs = figure.subfigures(4, 1)
#for fig, ap, basis in zip(figs, aps, bases):
#    _basis = bases[basis]._basis(coordinates)
#    _ap = aps[ap]._aperture(coordinates)
#
#    axes = fig.subplots(2, num_ikes // 2)
#    for i in range(num_ikes):
#        row = i // (num_ikes // 2)
#        col = i % (num_ikes // 2)
#
#        fig.suptitle(basis)
#        _map = axes[row][col].imshow(_basis[i] * _ap)
#        axes[row][col].set_xticks([])
#        axes[row][col].set_yticks([])
#        axes[row][col].axis("off")
#        fig.colorbar(_map, ax=axes[row][col]) 
#plt.show()

#aps = {
#    "Default": dl.SquareAperture(0., 0., 0., 1., False, False),
#    "Trans. x": dl.SquareAperture(.5, 0., 0., 1., False, False),
#    "Trans. y": dl.SquareAperture(0., .5, 0., 1., False, False),
#    "Rot.": dl.SquareAperture(0., 0., np.pi / 4., 1., False, False),
#    "Soft": dl.SquareAperture(0., 0., 0., 1., False, True),
#}
#
#coeffs = np.ones((num_ikes,), dtype=float)
#bases = {
#    "Default": AberratedArbitraryAperture(noll_inds, coeffs, aps["Default"]),
#    "Trans. x": AberratedArbitraryAperture(noll_inds, coeffs, aps["Trans. x"]),
#    "Trans. y": AberratedArbitraryAperture(noll_inds, coeffs, aps["Trans. y"]),
#    "Rot.": AberratedArbitraryAperture(noll_inds, coeffs, aps["Rot."]),
#    "Soft": AberratedArbitraryAperture(noll_inds, coeffs, aps["Soft"]),
#}
#
#figure = plt.figure()
#figs = figure.subfigures(len(bases), 1)
#for fig, ap, basis in zip(figs, aps, bases):
#    _basis = bases[basis]._basis(coordinates)
#    _ap = aps[ap]._aperture(coordinates)
#
#    axes = fig.subplots(2, num_ikes // 2)
#    for i in range(num_ikes):
#        row = i // (num_ikes // 2)
#        col = i % (num_ikes // 2)
#
#        fig.suptitle(basis)
#        _map = axes[row][col].imshow(_basis[i] * _ap, cmap=plt.cm.seismic, vmin=-3, vmax=3)
#        axes[row][col].set_xticks([])
#        axes[row][col].set_yticks([])
#        fig.colorbar(_map, ax=axes[row][col]) 
#plt.show()

#circ_ap = dl.CircularAperture(0., 0., 1., False, False)
#basis = AberratedCircularAperture(noll_inds, np.ones((num_ikes,)), circ_ap)
#
#_basis = basis._basis(coordinates)
#_aperture = circ_ap._aperture(coordinates)
#
#fig, axes = plt.subplots(2, num_ikes // 2, figsize=((num_ikes // 2)*4, 2*3))
#for i in range(num_ikes):
#    row = i // (num_ikes // 2)
#    col = i % (num_ikes // 2)
#    _map = axes[row][col].imshow(_basis[i] * _aperture)
#    fig.colorbar(_map, ax=axes[row][col]) 
#
#plt.show()

#hex_ap = dl.HexagonalAperture(0., 0., 0., 1., False, False)
#hex_basis = AberratedHexagonalAperture(noll_inds, np.ones((num_ikes,)), hex_ap)
#
#_basis = hex_basis._basis(coordinates)
#_aperture = hex_ap._aperture(coordinates)
#
#fig, axes = plt.subplots(2, num_ikes // 2, figsize=((num_ikes // 2)*4, 2*3))
#for i in range(num_ikes):
#    row = i // (num_ikes // 2)
#    col = i % (num_ikes // 2)
#    _map = axes[row][col].imshow(_basis[i])
#    fig.colorbar(_map, ax=axes[row][col]) 
#
#plt.show()

# Show the commit has and message
# 
#   `git show -s --format=oneline ref`
#   
# Re-apply a reverted commit
# 
#   `git cherry-pick ref`
#
# Reference commits easily using `HEAD~5`.
import dLux 
import jax.numpy as np
import abc
import typing


Array = typing.TypeVar("Array")
Layer = typing.TypeVar("Layer")


__all__ = ["UniformSpider"]


class Spider(dLux.apertures.Aperture, abc.ABC):
    """
    An abstraction on the concept of an optical spider for a space telescope.
    These are the things that hold up the secondary mirrors. For example,
    """
    def __init__(self, x_offset: float, y_offset: float, softening: bool):
        super().__init__(x_offset, y_offset, False, softening)


    def _strut(self, angle: float, coordinates: Array) -> Array:
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
        x, y = coordinates[0][:, ::-1], coordinates[1][:, :]
        perp = np.tan(angle)
        gradient = np.tan(angle)
        dist = np.abs(y - gradient * x) / np.sqrt(1 + gradient ** 2)
        theta = np.arctan2(y, x) + np.pi 
        theta = np.where(theta > angle, theta - angle, theta + 2 * np.pi - angle)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)

        # So the current problem is that I need to translate the coordinates 
        # around by angle and then return them to the range [0, 2 pi].

        strut = np.where((theta > np.pi / 2.) & (theta < 3. * np.pi / 2.), 1., dist)

        # This is all a hot mess. That is what this is. So how do I fix it?
        # Well it is obviously not trivial. I to take an array of angles and
        # essentially rotate it by angle. So I need to find all of the 
        # points where theta is greater than angle and then I subtract 
        # angle from theta in these points. Where theta is less than
        # angle I want to add two pi - angle. Consider the case of angle equals
        # three pi on two. The fourth quadrant has three pi on two subtracted 
        # giving it a range of zero to pi on two. Yes I think that this will 
        # work. 

        return strut


class UniformSpider(Spider):
    """
    A spider with equally-spaced, equal-width struts. This is of course the 
    most common and simplest implementation of a spider. Gradients can be 
    taken with respect to the width of the struts and the global rotation 
    as well as the centre of the spider.

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
            x_offset: float,
            y_offset: float, 
            number_of_struts: int, 
            width_of_struts: float, 
            rotation: float,
            softening: bool) -> Layer:
        """
        Parameters
        ----------
        radius_of_spider: float, meters
            The physical width of the spider. For the moment it is assumed to 
            be embedded within a circular aperture.         
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
        super().__init__(x_offset, y_offset, softening)
        self.number_of_struts = int(number_of_struts)
        self.rotation = np.asarray(rotation).astype(float)
        self.width_of_struts = np.asarray(width_of_struts).astype(float)


    def _metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(coordinates)
        angles = np.linspace(0, 2 * np.pi, self.number_of_struts, 
            endpoint=False)
        angles += self.rotation
        struts = np.array([self._strut(angle, coordinates) for angle in angles]) - self.width_of_struts / 2.
        softened = self._soften(struts)
        return softened.prod(axis=0)
        
 
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
        aperture = self._aperture()
        wavefront = params["Wavefront"]
        wavefront = wavefront\
            .set_amplitude(wavefront.get_amplitude() * aperture)\
            .set_phase(wavefront.get_phase() * aperture)
        params["Wavefront"] = wavefront
        return params
