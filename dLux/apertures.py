from __future__ import annotations
import dLux
from abc import ABC, abstractmethod
from jax import numpy as np, lax, vmap, Array
from jax.tree_util import tree_map, tree_flatten
from equinox import filter
from dLux.utils import get_pixel_positions, coordinates as c, opd_to_phase, \
    list_to_dictionary
from dLux.utils.units import convert_angular, convert_cartesian


Wavefront = dLux.wavefronts.Wavefront
OpticalLayer = dLux.optics.OpticalLayer
ZernikeBasis = lambda : dLux.aberrations.ZernikeBasis
TransmissiveLayer = lambda : dLux.optics.TransmissiveLayer
AberrationLayer = lambda : dLux.optics.AberrationLayer
TransmissiveOptic = lambda : dLux.optics.TransmissiveOptic


__all__ = ["CircularAperture", "SquareAperture", "HexagonalAperture", 
           "RegularPolygonalAperture", "IrregularPolygonalAperture", 
           "StaticAperture", "AberratedAperture", "StaticAberratedAperture", 
           "AnnularAperture", "RectangularAperture", "CompoundAperture", 
           "MultiAperture", "UniformSpider", "ApertureFactory"]


class ApertureLayer(TransmissiveLayer(), ABC):
    """
    The abstract base class that all aperture layers inherit from. This 
    instatiates the Transmissive class, intialising the name and normalisation
    attributes and providing the correct functionality for the `__call__` 
    method.

    Attributes
    ----------
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """

    
    def __init__(self      : OpticalLayer, 
                 normalise : bool = False,
                 name      : str  = "ApertureLayer",
                 **kwargs) -> ApertureLayer:
        """
        Constructor for the ApertureLayer class, instatiating the OpticalLayer 
        class.

        Parameters
        ----------
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name : str = 'ApertureLayer'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(normalise, name, **kwargs)


    @abstractmethod
    def _transmission(self        : ApertureLayer, 
                      coordinates : Array) -> Array: # pragma: no cover
        """
        Compute the array representing the aperture transmission on the 
        provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        pass


    def get_transmission(self     : ApertureLayer, 
                         npixels  : int, 
                         diameter : float) -> Array:
        """
        Compute the array representing the aperture on a set of coordinates 
        with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the array to calcualte the transmission on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._transmission(coordinates)


    def __call__(self : ApertureLayer, wavefront : Wavefront) -> Wavefront:
        """
        Apply the aperture to an incoming wavefront.

        Parameters
        ----------
        wavefront: Wavefront
            The wavefront before encountering the aperture.

        Returns
        -------
        wavefront: Wavefront
            The wavefront after encountering the aperture.
        """
        wavefront *= self._transmission(wavefront.coordinates)
        if self.normalise:
            return wavefront.normalise()
        return wavefront


class AbstractDynamicAperture(ApertureLayer, ABC):
    """
    Abstract base class instatiating a series of methods designed to generate
    apertures differentiably at run-time. This class primarily implements the 
    coordinate transformations that can be applied to each aperture in order to 
    have fully control over the aperture shape, and apply global transformations
    to the apertures.
    
    Attributes
    ----------
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    centre      : Array
    shear       : Array
    compression : Array
    rotation    : Array
    

    def __init__(self        : ApertureLayer, 
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 normalise   : bool  = False,
                 name        : str   = 'AbstractDynamicAperture'
                 ) -> ApertureLayer:
        """
        Constructor for the AbstractDynamicAperture class.

        Parameters
        ----------
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'AbstractDynamicAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(normalise=normalise, name=name)

        self.centre = np.asarray(centre).astype(float)
        self.shear = np.asarray(shear).astype(float)
        self.compression = np.asarray(compression).astype(float)
        self.rotation = np.asarray(rotation).astype(float)

        dLux.exceptions.validate_eq_attr_dims(self.centre.shape, (2,), "centre")
        dLux.exceptions.validate_eq_attr_dims(self.shear.shape, (2,), "shear")
        dLux.exceptions.validate_eq_attr_dims(
            self.compression.shape, (2,), "compression")
        dLux.exceptions.validate_eq_attr_dims(
            self.rotation.shape, (), "rotation")


    def _coordinates(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Transform the input coordinates into the coordinate system of the 
        aperture. 

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to transform.

        Returns
        -------
        coordinates: Array, meters
            The coordinates of the `Aperture`.
        """
        is_trans = (self.centre != np.zeros((2,), float)).any()
        coordinates = lax.cond(is_trans,
            lambda: c.translate(coordinates, self.centre),
            lambda: coordinates)

        is_compr = (self.compression != np.ones((2,), float)).any()
        coordinates = lax.cond(is_compr,
            lambda: c.compress(coordinates, \
                self.compression),
            lambda: coordinates)

        is_shear = (self.shear != np.zeros((2,), float)).any()
        coordinates = lax.cond(is_shear,
            lambda: c.shear(coordinates, self.shear),
            lambda: coordinates)

        is_rot = (self.rotation != 0.)
        coordinates = lax.cond(is_rot,
            lambda: c.rotate(coordinates, self.rotation),
            lambda: coordinates)

        return coordinates


class DynamicAperture(AbstractDynamicAperture, ABC):
    """
    An abstract base class that implements the methods required to provide soft
    edges to the apertures and generate either transmissive or occulting 
    apertures.

    Attributes
    ----------
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    occulting : bool 
    softening : Array
    

    def __init__(self        : ApertureLayer, 
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool = False,
                 name        : str   = 'DynamicAperture') -> ApertureLayer:
        """
        Constructor for the DynamicAperture class.

        Parameters
        ----------
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'DynamicAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre      = centre,
                         shear       = shear,
                         compression = compression,
                         rotation    = rotation,
                         normalise   = normalise,
                         name        = name)
        self.occulting = bool(occulting)
        self.softening = np.asarray(softening).astype(float) 
        dLux.exceptions.validate_eq_attr_dims((), self.softening.shape, 
                                              "softening")


    @abstractmethod
    def _extent(self : ApertureLayer) -> Array: # pragma: no cover
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for 
        speed.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """


    @abstractmethod
    def _soft_edged(self        : ApertureLayer, 
                    coordinates : Array) -> Array: # pragma: no cover
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """


    @abstractmethod
    def _hard_edged(self        : ApertureLayer, 
                    coordinates : Array) -> Array: # pragma: no cover
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """


    def _soften(self : ApertureLayer, distances : Array) -> Array:
        """
        Converts the distances from an edge into a soft edged transmission array
        using a tanh function.

        Parameters
        ----------
        distances: Array
            The distances from an edge the the aperture.

        Returns
        -------
        transmission: Array
            The softened transmission of the aperture edge based on the input
            distances.
        """
        steepness = 3. / self.softening * distances.shape[-1]
        return (np.tanh(steepness * distances) + 1.) / 2.


    def _transmission(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the array representing the aperture on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        coordinates = self._coordinates(coordinates) 

        aperture = lax.cond(
            (self.softening != 0.).any(),
            lambda coords: self._soft_edged(coords),
            lambda coords: self._hard_edged(coords).astype(float),
            coordinates)

        # TODO: Workout how to recast this using raw logic and see 
        #       if it is faster or not. Need to lok at the `jaxpr` 
        #       for python if statements like this and to try 
        #       work-out if extra stuff is getting done.
        if self.occulting:
            aperture = (1. - aperture)

        return aperture


    def _normalised_coordinates(self        : ApertureLayer, 
                                coordinates : Array) -> Array:
        """
        Shift a set of coodinates to be centered on the aperture and scaled such
        that the radial distance is 1 to the edge of the aperture.

        ### Here

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.
        
        Returns
        -------
        coordinates : Array, meters
            The coordinate system centered on the aperture with radius 
            normalised the maximum distance of an edge from the center.
        """
        return self._coordinates(coordinates) / self._extent()




#################################
### Concrete Aperture Classes ###
#################################
class CircularAperture(DynamicAperture):
    """
    A circular aperture parameterised by its radius.

    Attributes
    ----------
    radius: Array, meters
        The radius of the aperture. 
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    radius : Array

 
    def __init__(self        : ApertureLayer, 
                 radius      : Array, 
                 centre      : Array = np.array([0., 0.]),
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 occulting   : bool = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "CircularAperture",
                 ) -> Array:
        """
        Constructor for the CircularAperture class.

        Parameters
        ----------
        radius: Array, meters 
            The radius of the aperture.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'CircularAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre      = centre, 
                         shear       = shear, 
                         compression = compression, 
                         occulting   = occulting, 
                         softening   = softening,
                         normalise   = normalise,
                         name        = name) 

        self.radius = np.asarray(radius).astype(float)
        dLux.exceptions.validate_eq_attr_dims((), self.radius.shape, "radius")


    def _soft_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        scale = coordinates[0, 1] - coordinates[0, 0]
        coordinates = np.hypot(coordinates[0], coordinates[1])
        return self._soften(- coordinates + self.radius)


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        coordinates = np.hypot(coordinates[0], coordinates[1])
        return (coordinates < self.radius).astype(float)


    def _extent(self : ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        return self.radius
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        radius = convert_cartesian(self.radius, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} Circular Aperture with radius "
                   f"{radius} {cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class AnnularAperture(DynamicAperture):
    """
    An annular aperture defined by its inner and outer radii.

    Attributes
    ----------
    rmax: Array, meters
        Outer radius of aperture.
    rmin: Array, meters
        Inner radius of aperture.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    rmin : Array
    rmax : Array

    
    def __init__(self        : ApertureLayer, 
                 rmax        : Array, 
                 rmin        : Array, 
                 centre      : Array = np.array([0., 0.]),
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "AnnularAperture") -> ApertureLayer:
        """
        Constructor for the AnnularAperture class.

        Parameters
        ----------
        rmax : Array, meters
            The outer radius of the aperture. 
        rmin : Array, meters
            The inner radius of the aperture. 
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'AnnularAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression, 
                         occulting = occulting, 
                         softening = softening,
                         normalise = normalise,
                         name = name)

        self.rmax = np.asarray(rmax).astype(float)
        self.rmin = np.asarray(rmin).astype(float)

        dLux.exceptions.validate_eq_attr_dims((), self.rmax.shape, "rmax")
        dLux.exceptions.validate_eq_attr_dims((), self.rmin.shape, "rmin")


    def _soft_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        coordinates = np.hypot(coordinates[0], coordinates[1])
        return self._soften(coordinates - self.rmin) * \
            self._soften(- coordinates + self.rmax)


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        coordinates = np.hypot(coordinates[0], coordinates[1])
        return ((coordinates > self.rmin) * \
            (coordinates < self.rmax)).astype(float)


    def _extent(self : ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        return self.rmax
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        rmin = convert_cartesian(self.rmin, 'meters', cartesian_units)
        rmax = convert_cartesian(self.rmax, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} Annular Aperture with inner "
                   f"radius {rmin} {cartesian_units} and outer radius {rmax} "
                   f"{cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class RectangularAperture(DynamicAperture):
    """
    A rectangular aperture parameterised by it height and width.

    Attributes
    ----------
    height: Array, meters
        The length of the aperture in the y-direction. 
    width: Array, meters
        The length of the aperture in the x-direction. 
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    height : Array
    width  : Array


    def __init__(self        : ApertureLayer, 
                 height      : Array, 
                 width       : Array, 
                 centre      : Array = np.array([0., 0.]),
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "RectangularAperture") -> ApertureLayer: 
        """
        Constructor for the RectangularAperture class.

        Parameters
        ----------
        height: Array, meters 
            The length of the aperture in the y-direction.
        width: Array, meters
            The length of the aperture in the x-direction.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'RectangularAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear,
                         compression = compression,
                         rotation = rotation, 
                         occulting = occulting, 
                         softening = softening,
                         normalise = normalise,
                         name = name)

        self.height = np.asarray(height).astype(float)
        self.width = np.asarray(width).astype(float)

        dLux.exceptions.validate_eq_attr_dims((), self.height.shape, "height")
        dLux.exceptions.validate_eq_attr_dims((), self.width.shape, "width")


    def _soft_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        y_mask = self._soften(- np.abs(coordinates[1]) + self.height / 2.)
        x_mask = self._soften(- np.abs(coordinates[0]) + self.width / 2.)
        return x_mask * y_mask

    
    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        y_mask = np.abs(coordinates[1]) < self.height / 2.
        x_mask = np.abs(coordinates[0]) < self.width / 2.
        return (x_mask * y_mask).astype(float)


    def _extent(self : ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        return np.hypot(self.height / 2., self.width / 2.)
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        height = convert_cartesian(self.height, 'meters', cartesian_units)
        width = convert_cartesian(self.width, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} Rectangular Aperture with height "
                   f"{height} {cartesian_units} and width {width} "
                   f"{cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class SquareAperture(DynamicAperture):
    """
    A square aperture parameterised by its width.

    Attributes
    ----------
    width: Array, meters
        The side length of the aperture. 
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the
        aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    width : Array

 
    def __init__(self        : ApertureLayer, 
                 width       : Array, 
                 centre      : Array = np.array([0., 0.]),
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "SquareAperture") -> ApertureLayer: 
        """
        Constructor for the SquareAperture class.

        Parameters
        ----------
        width: Array, meters
            The side length of the aperture. 
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'SquareAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear,
                         compression = compression,
                         rotation = rotation, 
                         occulting = occulting, 
                         softening = softening,
                         normalise = normalise,
                         name = name)

        self.width = np.asarray(width).astype(float)

        dLux.exceptions.validate_eq_attr_dims((), self.width.shape, "width")


    def _soft_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        x_mask = self._soften(- np.abs(coordinates[0]) + self.width / 2.)
        y_mask = self._soften(- np.abs(coordinates[1]) + self.width / 2.)
        return x_mask * y_mask


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        x_mask = np.abs(coordinates[0]) < self.width / 2.
        y_mask = np.abs(coordinates[1]) < self.width / 2.
        return (x_mask * y_mask).astype(float)


    def _extent(self : ApertureLayer) -> Array:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        return np.sqrt(2) * self.width / 2.
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        width = convert_cartesian(self.width, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} Rectangular Aperture with width "
                   f"{width} {cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class PolygonalAperture(DynamicAperture, ABC):
    """
    Abstract base class for all polygonal apertures, from which both regular 
    and irregular polygonal apertures inherit from, implementing some shared 
    methods.
    
    Implementation Notes: A lot of the code that is provided was carefully hand 
    vectorised. In general, where a shape change is applied to an array the new 
    array is given the prefix `bc` standing for "broadcastable".

    Attributes
    ----------
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    

    def __init__(self        : ApertureLayer, 
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = 'PolygonalAperture') -> ApertureLayer:
        """
        Constructor for the PolygonalAperture class.

        Parameters
        ----------
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'PolygonalAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         occulting = occulting,
                         softening = softening,
                         normalise = normalise,
                         name = name)
    
    
    def _perp_dists_from_lines(self : ApertureLayer, 
                               m    : float, 
                               x1   : float, 
                               y1   : float,
                               xs   : Array, 
                               ys   : Array) -> Array:
        """
        Calcualtes the perpendicular distance of the cartesian (x, y) 
        coordinates from a line. The line is parameteried by its gradient m and
        a point (x1, y1) that lies on the line.
        
        Parameters
        ----------
        m: float 
            The gradient of the line.
        x1: float, meters
            The x coordinate the point that lies on the line.
        y1: float, meters
            The y coordinate the point that lies on the line.
        xs: Array, meters
            The x coordinates to calculate the distance on.
        ys: Array, meters
            The y coordinates to calculate the distance on.
        
        Returns
        -------
        distances: Array, meters
            The distance of the points (xs, ys) from the line.
        """
        inf_case = (xs - x1)
        gen_case = (m * inf_case - (ys - y1)) / np.sqrt(1 + m ** 2)
        return np.where(np.isinf(m), inf_case, gen_case)
    
    
    def _grad_from_two_points(self : ApertureLayer, 
                              xs   : float, 
                              ys   : float) -> float: # pragma: no cover
        """
        Calculate the gradient of the chord that connects two points. 
        Note: This is distinct from `_grads_from_many_points` in that
        it does not wrap arround.
        
        Parameters
        ----------
        xs: float, meters
            The x coordinates of the two points.
        ys: float, meters
            The y coordinates of the two points.
            
        Returns
        -------
        m: float
            The gradient of the chord that connects the two points.
        """
        return (ys[1] - ys[0]) / (xs[1] - xs[0])
    
    
    def _offset(self      : ApertureLayer, 
                theta     : float, 
                threshold : float) -> float:
        """
        Transform the angular range of polar coordinates so that the new lowest 
        angle is offset. The final range should be $[\\phi, \\phi + 2 \\pi]$ 
        where $\\phi$ represents the `threshold`. 
        
        Parameters
        ----------
        theta: float, radians
            The angular coordinates.
        threshold: float
            The amount to offset the coordinates by.
        
        Returns
        -------
        theta: float, radians 
            The offset coordinate system.
        """
        comps = (theta < threshold).astype(float)
        return theta + comps * 2. * np.pi
    
    
    def _is_orig_left_of_edge(self : ApertureLayer, 
                              ms   : float, 
                              xs   : float, 
                              ys   : float) -> int:
        """
        Determines whether the origin is to the left or the right of the edge. 
        The edge(s) are defined by a set of gradients, ms and points (xs, ys).
        
        Parameters
        ----------
        ms: float
            The gradient of the edge(s).
        xs: float, meters
            The set of x coordinates that lie along the edges. 
        ys: float, meters
            The set of y coordinates that lie along the edges.
            
        Returns
        -------
        is_left: int
            1 if the origin is to the left else -1.
        """
        # NOTE: see class docs.
        bc_orig = np.array([[0.]])
        dist_from_orig = self._perp_dists_from_lines(ms, xs, ys, bc_orig, \
            bc_orig)
        return np.sign(dist_from_orig)
    
    
class IrregularPolygonalAperture(PolygonalAperture):
    """
    An arbitrary aperture parameterised by a set of vertices.

    TODO: Check if the verticies need to be defined in a specific way, based on
    the methods this looks like the case (ie, ordered).

    Attributes
    ----------
    vertices: Array, meters
        The location of the vertices of the aperture.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    vertices : Array
    
    
    def __init__(self        : ApertureLayer, 
                 vertices    : Array,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "IrregularPolygonalAperture"
                 ) -> ApertureLayer:
        """
        Constructor for the IrregularPolygonalAperture class.

        Parameters
        ----------
        vertices: Array, meters
            The location of the vertices of the aperture.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'IrregularPolygonalAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         occulting = occulting,
                         softening = softening,
                         normalise = normalise,
                         name = name)
        
        self.vertices = np.array(vertices).astype(float)
        dLux.exceptions.validate_bc_attr_dims(
            (1, 2), self.vertices.shape, "vertices")
            
    
    def _grads_from_many_points(self : ApertureLayer, 
                                xs   : float, 
                                ys   : float) -> float:
        """
        Given a set of points, calculate the gradient of the line that connects 
        those points. This function assumes that the points are provided in the 
        order they are to be connected together. Notice that we also assume 
        there are more than two points, but more can be provided in which case 
        the shape is assumed to be closed. The output has the same shape as the 
        input and does not check for infinite (vertical) gradients.
        
        Note: Due to the intensly vectorised nature of this code it is ofen 
        necessary to provide the parameters with expanded dimensions. This may 
        be achieved using `x1[:, None, None]` or `x1.reshape((-1, 1, 1))` or 
        `np.expand_dims(x1, (1, 2))`.
        
        Parameters
        ----------
        xs: float, meters
            The x coordinates of the points that are to be connected. 
        ys: float, meters
            The y coordinates of the points that are to be connected. 
            Must have the same shape as x. 
            
        Returns
        -------
        ms: float
            The gradients of the lines that connect the vertices. The vertices 
            wrap around to form a closed shape whatever it may look like. 
        """
        x_diffs = xs - np.roll(xs, -1)
        y_diffs = ys - np.roll(ys, -1)
        return y_diffs / x_diffs
    
    
    def _extent(self : ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        verts = self.vertices
        dist_to_verts = np.hypot(verts[:, 1], verts[:, 0])
        return np.max(dist_to_verts)
    
    
    def _soft_edged(self : ApertureLayer, coordinates : float) -> float:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        # NOTE: see class docs.
        bc_x1 = self.vertices[:, 0][:, None, None]
        bc_y1 = self.vertices[:, 1][:, None, None]

        bc_x = coordinates[0][None, :, :]
        bc_y = coordinates[1][None, :, :]

        theta = np.arctan2(bc_y1, bc_x1)
        offset_theta = self._offset(theta, 0.)

        sorted_inds = np.argsort(offset_theta.flatten())

        sorted_x1 = bc_x1[sorted_inds]
        sorted_y1 = bc_y1[sorted_inds]
        sorted_m = self._grads_from_many_points(sorted_x1, sorted_y1)

        dist_from_edges = self._perp_dists_from_lines(sorted_m, sorted_x1, \
            sorted_y1, bc_x, bc_y)  
        dist_sgn = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)
        soft_edges = self._soften(dist_sgn * dist_from_edges)

        return (soft_edges).prod(axis=0)


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        # NOTE: see class docs.
        bc_x1 = self.vertices[:, 0][:, None, None]
        bc_y1 = self.vertices[:, 1][:, None, None]

        bc_x = coordinates[0][None, :, :]
        bc_y = coordinates[1][None, :, :]

        theta = np.arctan2(bc_y1, bc_x1)
        offset_theta = self._offset(theta, 0.)

        sorted_inds = np.argsort(offset_theta.flatten())

        sorted_x1 = bc_x1[sorted_inds]
        sorted_y1 = bc_y1[sorted_inds]
        sorted_m = self._grads_from_many_points(sorted_x1, sorted_y1)

        dist_from_edges = self._perp_dists_from_lines(sorted_m, sorted_x1, \
            sorted_y1, bc_x, bc_y)  
        dist_sgn = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)
        edges = (dist_from_edges * dist_sgn) > 0.

        return (edges).prod(axis=0)
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = f"Applies a {transmissive} Irregular Polygonal Aperture"
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class RegularPolygonalAperture(PolygonalAperture):
    """
    A regular polygonal aperture defined by its number of sides and the maximum 
    radius to the vertices from its center.
    
    Attributes
    ----------
    nsides: int
        The number of sides of the aperture. 
    rmax: Array, meters
        The maximum radius to the vertices from its center.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    nsides : int
    rmax   : Array
        
    
    def __init__(self        : ApertureLayer, 
                 nsides      : int,
                 rmax        : Array,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "RegularPolygonalAperture"
                 ) -> ApertureLayer:
        """
        Constructor for the RegularPolygonalAperture class.

        Parameters
        ----------
        nsides: int
            The number of sides of the aperture.  
        rmax: Array, meters
            The maximum radius to the vertices from its center.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'RegularPolygonalAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         occulting = occulting,
                         softening = softening,
                         normalise = normalise,
                         name = name)

        self.nsides = int(nsides)
        self.rmax = np.array(rmax).astype(float)

        dLux.exceptions.validate_eq_attr_dims((), self.rmax.shape, "rmax")

        
    def _extent(self : ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        return self.rmax
        
    
    def _soft_edged(self : ApertureLayer, coordinates : float) -> float:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        x = coordinates[0]
        y = coordinates[1]

        neg_pi_to_pi_phi = np.arctan2(y, x) 
        alpha = np.pi / self.nsides
            
        i = np.arange(self.nsides)[:, None, None] # Dummy index
        bounds = 2. * i * alpha
            
        ms = -1 / np.tan(2. * i * alpha + alpha)
        xs = self.rmax * np.cos(2. * i * alpha)
        ys = self.rmax * np.sin(2. * i * alpha)
        dists = self._perp_dists_from_lines(ms, xs, ys, x, y)
        inside = self._is_orig_left_of_edge(ms, xs, ys)
         
        dist = self._soften(inside * dists)
        return dist.prod(axis=0)


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        x = coordinates[0]
        y = coordinates[1]

        neg_pi_to_pi_phi = np.arctan2(y, x) 
        alpha = np.pi / self.nsides
            
        i = np.arange(self.nsides)[:, None, None] # Dummy index
        bounds = 2. * i * alpha
            
        ms = -1 / np.tan(2. * i * alpha + alpha)
        xs = self.rmax * np.cos(2. * i * alpha)
        ys = self.rmax * np.sin(2. * i * alpha)
        dists = self._perp_dists_from_lines(ms, xs, ys, x, y)
        inside = self._is_orig_left_of_edge(ms, xs, ys)
         
        dist = (inside * dists) > 0.
        return dist.prod(axis=0)


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        rmax = convert_cartesian(self.rmax, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} {self.nsides} sided Regular "
                   f"Polygonal Aperture of max radius {rmax:.{sigfigs}} "
                   f"{cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


class HexagonalAperture(RegularPolygonalAperture):
    """
    A hexagonal aperture parameterised by the maximum radius to the vertices 
    from its center.
    
    Attributes
    ----------
    rmax : Array, meters
        The maximum radius to the vertices from its center.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    occulting: bool
        Is the aperture occulting or tranmissive. False results in a tranmissive
        aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    rmax : Array

    
    def __init__(self        : ApertureLayer, 
                 rmax        : Array,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 occulting   : bool  = False, 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = "HexagonalAperture") -> ApertureLayer:
        """
        Constructor for the HexagonalAperture class.

        Parameters
        ----------
        rmax : Array, meters
            The maximum radius to the vertices from its center.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        occulting: bool = False
            Is the aperture occulting or tranmissive. False results in a 
            tranmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'HexagonalAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(nsides = 6,
                         rmax = rmax,
                         centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         occulting = occulting,
                         softening = softening,
                         normalise = normalise,
                         name = name)


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        rmax = convert_cartesian(self.rmax, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)
        transmissive = "transmissive" if not self.occulting else "occulting"

        summary = (f"Applies a {transmissive} Hexagonal Aperture of max radius "
                   f"{rmax:.{sigfigs}} {cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


###############
### Spiders ###
###############
class Spider(DynamicAperture, ABC):
    """
    An abstract class for generating aperture spiders struts.

    Attributes
    ----------
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    
    
    def __init__(self        : ApertureLayer, 
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.), 
                 softening   : Array = np.array(1.),
                 normalise   : bool  = False,
                 name        : str   = 'Spider') -> ApertureLayer:
        """
        Constructor for the Spider class.

        Parameters
        ----------
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'Spider'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         occulting = False,
                         softening = softening,
                         normalise = normalise,
                         name = name)
 
 
    def _strut(self        : ApertureLayer, 
               angle       : float, 
               coordinates : Array) -> Array:
        """
        Generates a representation of a single strut in the spider. 
 
        Parameters
        ----------
        angle: float, radians
            The angle that this strut points from the positive x-axis.
 
        Returns
        -------
        distance: float
            The distance from the center of the strut.
        """
        x, y = coordinates[0], coordinates[1]
        gradient = np.tan(angle)
        dist = np.abs(y - gradient * x) / np.sqrt(1 + gradient ** 2)
        theta = np.arctan2(y, x) + np.pi 
        theta = np.where(theta > angle, theta - angle, theta + 2 * np.pi -
            angle)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)
        strut = np.where((theta > np.pi / 2.) & (theta < 3. * np.pi / 2.), 1.,
            dist)
        return strut


    def _extent(self : ApertureLayer) -> float: # pragma: no cover
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre.

        Returns
        -------
        extent : float
            The maximum distance from the centre to edge of aperture.
        """
        raise NotImplementedError("The `Spider` class and its derivatives "
            "are not designed to be used with the `AberatedAperture` class. "
            "If this is part of a `CompoundAperture` place the "
            "`AberratedAperture`s into the `CompoundAperture` not the "
            "other way arround.")


class UniformSpider(Spider):
    """
    A set of spider struts with equally-spaced, equal-width struts.
 
    Attributes
    ----------
    nstruts: int 
        The number of spider struts.
    strut_width: Array, meters
        The width of each strut. 
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the 
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    nstruts     : int
    strut_width : Array

    
    def __init__(self         : ApertureLayer, 
                 nstruts      : int,
                 strut_width  : Array,
                 centre       : Array = np.array([0., 0.]), 
                 shear        : Array = np.array([0., 0.]),
                 compression  : Array = np.array([1., 1.]),
                 rotation     : Array = np.array(0.),
                 softening    : Array = np.array(1.),
                 normalise    : bool  = False,
                 name         : str   = "UniformSpider") -> ApertureLayer:
        """
        Constructor for the UniformSpider class.

        Parameters
        ----------
        nstruts: int 
            The number of struts to equally space around the circle. This is not 
            a differentiable parameter. 
        strut_width: Array, meters
            The width of each strut. 
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the 
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'UniformSpider'
            The name of the layer, which is used to index the layers dictionary.
        """ 
        super().__init__(centre = centre, 
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         softening = softening,
                         normalise = normalise,
                         name = name)

        self.nstruts = int(nstruts)
        self.strut_width = np.asarray(strut_width).astype(float)

        dLux.exceptions.validate_eq_attr_dims(
            (), self.strut_width.shape, "Width_of_struts")


    def _stacked_struts(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calculates an array of individual struts comprising the full spider 
        aperture on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinate system to calculate the struts on.

        Returns
        -------
        struts: Array
            The array of all the individual struts.
        """
        coordinates = self._coordinates(coordinates)
        angles = np.linspace(0, 2. * np.pi, self.nstruts, endpoint=False)
        angles += self.rotation
        return vmap(self._strut, in_axes=(0, None))(angles, coordinates) 

 
    def _soft_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the soft edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The softed edged aperture shape.
        """
        struts = self._stacked_struts(coordinates) - self.strut_width / 2.
        softened = self._soften(struts)
        return softened.prod(axis=0)


    def _hard_edged(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calcualtes the hard edged aperture shape on the input coordinates.

        Parameters
        ----------
        coordinates: Array, meters
            The coordinates to calculate the aperture shape on.

        Returns
        -------
        aperture: Array
            The hard edged aperture shape.
        """
        struts = self._stacked_struts(coordinates) > self.strut_width / 2. 
        return struts.prod(axis=0)


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        width = convert_cartesian(self.strut_width, 'meters', cartesian_units)
        center = convert_cartesian(self.center, 'meters', cartesian_units)
        rotation = convert_angular(self.rotation, 'radians', angular_units)

        summary = (f"Applies a {self.nstrut} strut spider with widths {width} "
                   f"{cartesian_units}")
        
        if self.softening != np.array(0):
            summary += f" softened by ~{self.softening} pixels"
        if self.center != np.array([0., 0.]):
            summary += f" centred at {center}"
        if self.rotation != np.array(0.):
            summary += f" rotated by {rotation} {angular_units}"
        if self.shear != np.array([0., 0.]):
            summary += f" sheared by {self.shear}"
        if self.compression != np.array([1., 1.]):
            summary += f" compressed by {self.compression}"
        return summary + "."


###################
### Aberrations ###
###################
class AbstractAberratedAperture(AberrationLayer(), ApertureLayer):
    """
    An abstract class for generating apertures with aberrations. This 
    instantiates the coefficients parameter, defining the amplitude of each 
    basis vector of the aberrations.
    
    Attributes
    ----------
    coefficients: Array
        The amplitude of each basis vector of the aberrations.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the
        aperture.
    TODO: Name
    """
    coefficients : Array


    def __init__(self         : AberrationLayer, 
                 coefficients : Array, 
                 name         : str = "AbstractAberratedAperture",
                 **kwargs) -> ApertureLayer:
        
        """
        Constructor for the AbstractAberratedAperture class.

        Parameters
        ----------
        coefficients: Array
            The amplitude of each basis vector of the aberrations.
        name: str = "AbstractAberratedAperture"
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name=name, **kwargs)

        self.coefficients = np.asarray(coefficients).astype(float)
        # NOTE: Dimension checking is complex here becuase AberratedApertures
        # and CompoundApertures must always have 1d coefficeints, but 
        # MultiApertures can have 2d coefficients.


    @abstractmethod
    def _basis(self        : ApertureLayer, 
               coordinates : Array) -> Array: # pragma: no cover
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the basis vectors on.

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """


    @abstractmethod
    def get_basis(self     : ApertureLayer, 
                  npixels  : int, 
                  diameter : float) -> Array: # pragma: no cover
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
 

    @abstractmethod
    def _opd(self        : ApertureLayer, 
             coordinates : Array) -> Array: # pragma: no cover
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the opd on.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """


    @abstractmethod
    def get_opd(self     : ApertureLayer, 
                npixels  : int, 
                diameter : float) -> Array: # pragma: no cover
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates with the specified number of pixels and 
        diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """


class AberratedAperture(AbstractAberratedAperture):
    """
    A class for generating apertures with aberrations. This class generates the
    basis vectors of the aberrations at run time, allowing for the aperture and
    aberrations to be recovered simultaneously.
 
    Attributes
    ----------
    aperture: ApertureLayer
        The aperture on which the aberration basis is defined.
    basis: list[Zernike]
        A list of basis functions that represent the basis. The exact 
        polynomials that are represented will depend on the aperture shape. 
    coefficients: Array
        The amplitude of each basis vector of the aberrations.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    aperture : ApertureLayer
    basis    : ZernikeBasis

    
    def __init__(self         : ApertureLayer, 
                 aperture     : ApertureLayer, 
                 noll_inds    : Array,
                 coefficients : Array = None,
                 name         : str   = "AberratedAperture",
                 **kwargs) -> ApertureLayer: 
        """
        Constructor for the AberratedAperture class.

        Parameters
        ----------
        aperture: ApertureLayer
            The aperture on which the aberration basis is defined.
        noll_inds: List[int]
            The noll indices are a scheme for indexing the Zernike
            polynomials. Normally these polynomials have two 
            indices but the noll indices prevent an order to 
            these pairs. All basis can be indexed using the noll
            indices based on `n` and `m`. 
        coefficients: Array = None
            The amplitude of each basis vector of the aberrations. If nothing 
            is provided, then the coefficients are set to zero.
        name: str = 'AberratedAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        # Ensure aperture is dynamic
        if not isinstance(aperture, DynamicAperture):
            raise TypeError("AberratedApertures can not contain Static, "
                "Compound or Multi Apertures. AberratedApertures can be "
                "placed in Compound or Multi Apertures, which can then be "
                "promoted to Static.")

        # Ensure transmissice
        if aperture.occulting:
            raise ValueError("AberratedApertures can not be occulting.")

        # Set Aperture
        self.aperture = aperture

        # Set Basis
        self.basis = dLux.aberrations.ZernikeBasis(noll_inds)

        # Initialise the coefficinets
        coefficients = np.zeros(len(noll_inds)) if coefficients is None \
            else np.asarray(coefficients).astype(float)

        super().__init__(coefficients=coefficients, name=name, **kwargs)
        
        # Dimensionality check
        dLux.exceptions.validate_bc_attr_dims(
            noll_inds.shape, self.coefficients.shape, "coefficients")
 

    def __call__(self : ApertureLayer, wavefront : Wavefront) -> Wavefront:
        """
        Apply the aperture and the abberations to the wavefront.  
 
        Parameters
        ----------
        wavefront: Wavefront
            The wavefront that is passing through the aperture.

        Returns
        -------
        wavefront: Wavefront
            The wavefront after passing through the aperture.
        """
        # Calculate aperture and opd
        coordinates = wavefront.coordinates
        transmission = self.aperture._transmission(coordinates)
        phase = wavefront.phase + self._opd(coordinates) * wavefront.wavenumber
        amplitude = transmission * wavefront.amplitude
        wavefront = wavefront.set(['amplitude', 'phase'], [amplitude, phase])

        # This must be done after the amplitude is updated because we dont
        # know what aperture have been applied upstream
        if self.aperture.normalise:
            wavefront = wavefront.normalise()
        return wavefront
 

    def _transmission(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the array representing the aperture on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        return self.aperture._transmission(coordinates)
        

    def get_transmission(self     : ApertureLayer, 
                         npixels  : int, 
                         diameter : float) -> Array:
        """
        Compute the array representing the aperture on a set of coordinates 
        with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the array to calcualte the transmission on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self.aperture._transmission(coordinates)


    def _basis(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the basis vectors on.

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        coordinates = self.aperture._normalised_coordinates(coordinates)

        if isinstance(self.aperture, RegularPolygonalAperture):
            ikes = self.basis.calculate_basis(coordinates, self.aperture.nsides)
        else:
            ikes = self.basis.calculate_basis(coordinates)

        is_reg_pol = isinstance(self.aperture, RegularPolygonalAperture)
        is_circ = isinstance(self.aperture, CircularAperture)

        if is_circ or is_reg_pol:
            return ikes

        aperture = self.aperture._transmission(coordinates)
        return self._orthonormalise(aperture, ikes) 


    def get_basis(self     : ApertureLayer, 
                  npixels  : int, 
                  diameter : float) -> Array:
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._basis(coordinates)
 

    def _opd(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the opd on.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        basis = self._basis(coordinates)
        return (basis * self.coefficients[:, None, None]).sum(axis=0)


    def get_opd(self : ApertureLayer, npixels : int, diameter : float) -> Array:
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates with the specified number of pixels and 
        diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._opd(coordinates)


    def _orthonormalise(self     : ApertureLayer, 
                        aperture : Array, 
                        zernikes : Array) -> Array:
        """
        Orthonomalises the zernike polynomials on the aperture.
        
        Parameters
        ----------
        aperture : Array
            An array representing the aperture.
        zernikes : Array
            The zernike polynomials to orthonormalise on the aperture.
 
        Returns
        -------
        basis : Array
            The orthonormalised zernike polynomials evaluated on the aperture.
        """
        pixel_area = aperture.sum()
        shape = zernikes.shape
        basis = np.zeros(shape).at[0].set(aperture)
 
        nterms = len(zernikes)
        # for j in np.arange(1, nterms):
        for j in range(nterms):
            intermediate = zernikes[j] * aperture
            coefficient = np.zeros((nterms, 1, 1), dtype=float)
            mask = (np.arange(1, nterms) > j + 1).reshape((-1, 1, 1))
 
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
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        summary = super().summary(angular_units, cartesian_units, sigfigs)
        return summary[:-1] + f" with {len(self.coefficients)} aberrations."


###########################
### Composite Apertures ###
###########################
class CompositeAperture(AbstractDynamicAperture):
    """
    An abstract class used to combine multiple apertures so that more complex
    apertures can have global transformations applied to them. Two examples 
    would be a pupil with spiders holding the secondary mirror or an aperture
    mask.

    Attributes
    ----------
    apertures: dict(str, Aperture)
       The sub-apertures that make up the full aperture. 
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    apertures : dict


    def __init__(self        : ApertureLayer, 
                 apertures   : list,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 normalise   : bool  = False,
                 name        : str   = 'CompositeAperture') -> ApertureLayer:
        """
        Constructor for the CompositeAperture class.

        Parameters
        ----------
        apertures: dict(str, Aperture)
            The sub-apertures that make up the full aperture.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'CompositeAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(centre = centre,
                         shear = shear, 
                         compression = compression,
                         rotation = rotation,
                         normalise = normalise,
                         name = name)
        
        for aperture in apertures:
            if not isinstance(aperture, (ApertureLayer, 
                AbstractAberratedAperture)):
                raise TypeError("All the apertures should be ApertureLayers.")
            if isinstance(aperture, AbstractStaticAperture):
                raise TypeError("StaticApertures cannot be put into " + \
                    "Compound or Multi Apertures. Please promote the " + \
                    "Compound or Multi Aperture to a StaticAperture.")

        self.apertures = list_to_dictionary(apertures, ordered=False)


    def _stacked_apertures(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Calculates an array of individual apertures comprising the compound 
        aperture on the input coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the apertures on.

        Returns
        -------
        apertures: Array
            The array of all the individual apertures.
        """
        # Get coordinates and define leaf function
        coordinates = self._coordinates(coordinates)
        _leaf = lambda ap: isinstance(ap, ApertureLayer)

        # Get Apertures
        get_aperture = lambda ap: ap._transmission(coordinates)
        aps = tree_map(get_aperture, self.apertures, is_leaf=_leaf)

        # Construct Aperture
        return np.array(list(aps.values()))


    @abstractmethod
    def _transmission(self        : ApertureLayer, 
                  coordinates : Array) -> Array: # pragma: no cover
        """
        Compute the array representing the aperture on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        aperture : Array 
            The array representing the transmission of the combined 
            sub-apertures. 
        """
    

    def get_transmission(self     : ApertureLayer, 
                         npixels  : int, 
                         diameter : float) -> Array:
        """
        Compute the array representing the aperture on a set of coordinates 
        with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._transmission(coordinates)


    def _basis(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the basis vectors on.

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        coordinates = self._coordinates(coordinates)
        aberrated = self._aberrated_apertures()
        _leaf = lambda ap: isinstance(ap, ApertureLayer)
        get_basis = lambda ap: ap._basis(coordinates)
        basis = tree_map(get_basis, aberrated, is_leaf=_leaf)
        return np.squeeze(np.array(tree_flatten(basis)[0]))

    
    def get_basis(self     : ApertureLayer, 
                  npixels  : int, 
                  diameter : float) -> Array:
        """
        Compute the basis vectors of the aperture aberrations on the provided 
        coordinates with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._basis(coordinates)


    def _coefficients(self : ApertureLayer) -> Array:
        """
        Returns the coefficients of the stored aberrated apertures.

        Returns 
        -------
        coefficients : Array
           The coefficients of the aberrated sub-aperture/apertures
        """
        aberrated = self._aberrated_apertures()
        _leaf = lambda ap: isinstance(ap, ApertureLayer)
        get_coeffs = lambda ap: ap.coefficients
        coeffs = tree_map(get_coeffs, aberrated, is_leaf=_leaf)
        return np.squeeze(np.array(tree_flatten(coeffs)[0]))

    
    @property
    def coefficients(self : ApertureLayer) -> Array:
        """
        Returns the coefficinets of the stored aberrated apertures.

        Returns
        -------
        coefficients : Array
           The coefficients of the aberrated sub-aperture/apertures
        """
        return self._coefficients()


    def _opd(self : ApertureLayer, coordinates : Array) -> Array:        
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the opd on.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        # Get the aberrated aperture in a list
        aberrated = self._aberrated_apertures()

        # Check for an aberrated aperture
        if not any(aberrated):
            return np.array(0)
        
        # Define leaf and get coordinates
        _leaf = lambda ap: isinstance(ap, ApertureLayer)
        coordinates = self._coordinates(coordinates)

        # Get basis
        get_basis = lambda ap: ap._basis(coordinates)
        basis = tree_map(get_basis, aberrated, is_leaf=_leaf)
        basis = np.array(tree_flatten(basis)[0])

        # Get coeffs
        get_coeffs = lambda ap: ap.coefficients
        coeffs = tree_map(get_coeffs, aberrated, is_leaf=_leaf)
        coeffs = np.array(tree_flatten(coeffs)[0])

        # Calculate opd
        return (basis * coeffs[:, :, None, None]).sum((0, 1))
        

    def get_opd(self : ApertureLayer, npixels : int, diameter : float) -> Array:
        """
        Compute the total optical path difference of the aperture aberrations 
        on the provided coordinates with the specified number of pixels and 
        diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the aperture in meters. 

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)
        return self._opd(coordinates)


    def _aberrated_apertures(self : ApertureLayer) -> list:
        """
        Returns the individual apertures with aberrations.

        Returns
        -------
        apertures: list[AberratedApertures]
            The list of apertures with aberrations.
        """
        # Define leaf fn
        is_aberrated = lambda leaf: isinstance(leaf, AberratedAperture)

        # Get aberrated apertures
        filter_map = tree_map(is_aberrated, self.apertures, is_leaf=is_aberrated)
        aberrated = filter(self.apertures, filter_map)
        return tree_flatten(aberrated, is_leaf=is_aberrated)[0]


    def __call__(self : ApertureLayer, wavefront : Wavefront) -> Wavefront:
        """
        Apply the aperture to an incoming wavefront.

        Parameters
        ----------
        wavefront: Wavefront
            The incoming wavefront.

        Returns
        -------
        wavefront: Wavefront
            The outgoing wavefront.
        """
        coordinates = wavefront.coordinates
        aper = self._transmission(coordinates)
        opd = self._opd(coordinates)

        # Calcualte and update amplitude and phase
        phase = wavefront.phase + opd * wavefront.wavenumber
        amplitude = wavefront.amplitude * aper
        wavefront = wavefront.set(['amplitude', 'phase'], [amplitude, phase])
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


    def __getattr__(self : ApertureLayer, key : str) -> Any:
        """
        Get the attribute of the aberrated apertures.

        TODO: Needs testing

        Parameters
        ----------
        key: str
            The attribute to get.
        
        Returns
        -------
        attribute: Any
            The attribute of the aberrated apertures.
        """
        if key in self.apertures:
            return self.apertures[key]
        else:
            raise AttributeError(f"{key} not in {self.apertures.keys()}")


class CompoundAperture(CompositeAperture):
    """
    A  class used to combine multiple apertures into a single coherent aperture.
    An example would be an aperture with spiders holding a secondary mirror.
    
    This class is distinct from the MultiAperture class in that the 
    sub-apertures are combined by mulitplying their respective transmissions 
    together, ie the sub-apertures are overlapping.

    This class should not contain a MulitAperture, but MultiApertures can 
    contain CompoundApertures.

    A single aberrated aperture can be placed into the set of apertures.

    Attributes
    ----------
    apertures: dict(str, Aperture)
        The sub-apertures that make up the full aperture.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """


    def __init__(self        : ApertureLayer,
                 apertures   : list,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 normalise   : bool  = False,
                 name        : str   = "CompoundAperture") -> ApertureLayer:
        """
        Constructor for the CompoundAperture class.

        Parameters
        ----------
        apertures: list[Aperture]
            The sub-apertures that make up the full aperture.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'CompoundAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        # Check for more than one aberration
        naberrated = 0
        for aperture in apertures:
            if isinstance(aperture, CompositeAperture):
                raise TypeError("CompositeApertures cannot be nested. To " +\
                    "combine multiple CompositeApertures, use MultiAperture.")
            if isinstance(aperture, AberratedAperture):
                naberrated += 1
        if naberrated > 1:
            raise TypeError("CompoundAperture can only have one " + \
                             "AberratedAperture.")
            
        super().__init__(apertures,
                         centre = centre,
                         shear = shear,
                         compression = compression,
                         rotation = rotation,
                         normalise = normalise,
                         name = name)
        

    def _transmission(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the array representing the aperture on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        aperture : Array 
            The array representing the transmission of the combined 
            sub-apertures. 
        """
        aps = self._stacked_apertures(coordinates)
        return aps.prod(axis=0)
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        summary = ("Applies a Compound Aperture with the following "
                   "sub-apertures: \n")
        for ap in self.apertures.values():
            ap_summary = ap.summary(angular_units, cartesian_units, sigfigs)
            summary += ap_summary + "\n"
        return summary


class MultiAperture(CompositeAperture):
    """
    A  class used to combine multiple apertures into a single coherent aperture.
    An example would be an aperture mask.
    
    This class is distinct from the CompoundAperture class in that the 
    sub-apertures are combined by adding their respective transmissions 
    together, ie the sub-apertures are not overlapping.

    This class can contain multiple CompoundApertures.

    Attributes
    ----------
    apertures: dict(str, Aperture)
        The sub-apertures that make up the full aperture.
    centre: Array, meters
        The (x, y) coordinates of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperutre.
    compression: Array 
        The (x, y) compression of the aperture. 
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """

    
    def __init__(self        : ApertureLayer,
                 apertures   : list,
                 centre      : Array = np.array([0., 0.]), 
                 shear       : Array = np.array([0., 0.]),
                 compression : Array = np.array([1., 1.]),
                 rotation    : Array = np.array(0.),
                 normalise   : bool  = False,
                 name        : str   = "MultiAperture") -> ApertureLayer:
        """
        Constructor for the MultiAperture class.

        Parameters
        ----------
        apertures: list[Aperture]
            The sub-apertures that make up the full aperture.
        centre: Array, meters = np.array([0., 0.])
            The (x, y) coordinates of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperutre.
        compression: Array  = np.array([1., 1.]) 
            The (x, y) compression of the aperture. 
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'MultiAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(apertures,
                         centre = centre,
                         shear = shear,
                         compression = compression,
                         rotation = rotation,
                         normalise = normalise,
                         name = name)


    def _transmission(self : ApertureLayer, coordinates : Array) -> Array:
        """
        Compute the array representing the aperture on the provided coordinates.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        aperture : Array 
            The array representing the transmission of the combined 
            sub-apertures. 
        """
        aps = self._stacked_apertures(coordinates)
        return aps.sum(axis=0)


    def _aberrated_apertures(self : ApertureLayer) -> list:
        """
        Returns the individual apertures with aberrations.
        Note: This method returns CompoundApertures if it contains apertures
        with aberrations in them.

        Returns
        -------
        apertures: list[Union[AberratedAperture, CompoundAperture]]
            The list of apertures with aberrations.
        """
        # Define leaf fn
        def is_aberrated(leaf):
            if isinstance(leaf, AberratedAperture):
                return True
            elif isinstance(leaf, CompoundAperture):
                if len(leaf._aberrated_apertures()) > 0:
                    return True
            return False

        # Get aberrated apertures
        filter_map = tree_map(is_aberrated, self.apertures, 
            is_leaf=is_aberrated)
        aberrated = filter(self.apertures, filter_map)
        return tree_flatten(aberrated, is_leaf=is_aberrated)[0]


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        summary = ("Applies a Multi Aperture with the following "
                   "sub-apertures: \n")
        for ap in self.apertures.values():
            ap_summary = ap.summary(angular_units, cartesian_units, sigfigs)
            summary += ap_summary + "\n"
        return summary


########################
### Static Apertures ###
########################
class AbstractStaticAperture(TransmissiveOptic(), ApertureLayer):
    """
    An abstract class used to represent static apertures. Static apertures 
    pre-calcualte the aperture array on the specified init time cooridantes and
    can not have its parameters optimised. 

    Attributes
    ----------
    aperture: Array
        The aperture represented as an array.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """


    def __init__(self        : ApertureLayer, 
                 aperture    : ApertureLayer, 
                 npixels     : int   = None, 
                 diameter    : float = None,
                 coordinates : Array = None,
                 name        : str   = "AbstractStaticAperture",
                 **kwargs) -> ApertureLayer:
        """
        Constructor for the AbstractStaticAperture class.

        Parameters
        ----------
        aperture: ApertureLayer
            The aperture to be pre-calculated and represented as an array.
        npixels : int = None
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters = None
            The diameter of the aperture in meters. 
        coordinates : Array, meters = None
            The coordinate system to calculate the aperture on.
        name: str = 'AbstractStaticAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        # Input check: Coordinates provided
        if coordinates is not None and \
            (npixels is not None or diameter is not None):
            raise ValueError("If coordinates is specified npixels and " + \
                "diameter can not be provided.")
        # Input check: Coordinates not provided
        elif coordinates is None and \
            (npixels is None or diameter is None):
            raise ValueError("both npixels and diameter must be provided.")
        
        # Generate coordinates if not provided
        if coordinates is None:
            npixels_in = (npixels, npixels)
            pixel_scales = (diameter / npixels, diameter / npixels)
            coordinates = get_pixel_positions(npixels_in, pixel_scales)

        if isinstance(aperture, AberratedAperture):
            normalise = aperture.aperture.normalise
        else:
            normalise = aperture.normalise
        transmission = aperture._transmission(coordinates)
        super().__init__(name = name, normalise = normalise, 
            transmission=transmission, **kwargs)


    @property
    def shape(self : ApertureLayer) -> tuple:
        """
        The shape of the aperture array.

        Returns
        -------
        shape : tuple
            The shape of the aperture transmission array.
        """
        return self.transmission.shape
    

    def _transmission(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the array representing the aperture.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the aperture on.

        Returns
        -------
        aperture : Array 
            The array representing the transmission of the combined 
            sub-apertures. 
        """
        return self.transmission


    def get_transmission(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the array representing the aperture on a set of coordinates 
        with the specified number of pixels and diameter.

        Parameters
        ----------
        npixels : int
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters
            The diameter of the array to calcualte the transmission on.

        Returns
        -------
        transmission : Array 
            The array representing the transmission of the aperture.
        """
        return self._transmission()


class StaticAperture(AbstractStaticAperture):
    """
    A class for static pre-calculated apertures, without aberrations. Static
    apertures with aberrations can be instantiated using the 
    StaticAberratedAberrated class.

    Attributes
    ----------
    aperture: Array
        The aperture represented as an array.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """


    def __init__(self        : ApertureLayer, 
                 aperture    : ApertureLayer, 
                 npixels     : int   = None, 
                 diameter    : float = None,
                 coordinates : Array = None,
                 name        : str   = "StaticAperture",
                 **kwargs) -> ApertureLayer:
        """
        Constructor for the StaticAperture class.

        Parameters
        ----------
        aperture: ApertureLayer
            An instance of DynamicAperture. 
        npixels : int = None
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters = None
            The diameter of the aperture in meters. 
        coordinates : Array, meters = None
            The coordinate system to calculate the aperture on.
        name: str = 'StaticAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        if isinstance(aperture, AbstractStaticAperture):
            raise TypeError("This Aperture is already static, please "
                "provide a dynamic aperture.")
        
        if isinstance(aperture, (CompoundAperture, MultiAperture)) and \
            len(aperture._aberrated_apertures()) > 0 or \
                isinstance(aperture, AberratedAperture):
            raise TypeError("This Aperture contains aberrated apertures, "
                "please use the StaticAberratedAperture class.")
        
        super().__init__(aperture = aperture, 
                         npixels = npixels, 
                         diameter = diameter, 
                         coordinates = coordinates, 
                         name = name,
                         **kwargs)
        

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return "Applies a pre-calculated Static Aperture."


class StaticAberratedAperture(AbstractStaticAperture, 
                              AbstractAberratedAperture):
    """
    A class for static pre-calculated apertures with aberrations. This 
    pre-calcaultes both the aperture and the basis at init time and can not 
    have the aperture properties optimised.

    Attributes
    ----------
    aperture: Array
        The aperture represented as an array.
    basis: Array 
        The basis represented as an array.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    name: str
        The name of the layer, which is used to index the layers dictionary.
    """
    basis : Array


    def __init__(self        : ApertureLayer, 
                 aperture    : ApertureLayer, 
                 npixels     : int   = None, 
                 diameter    : float = None,
                 coordinates : Array = None,
                 name        : str   = "StaticAberratedAperture"
                 ) -> ApertureLayer:
        """
        Constructor for the StaticAberratedAperture class.

        Parameters
        ----------
        aperture: AberratedAperture
            An instance of AberratedAperture. 
        npixels : int = None
            The number of pixels accross one edge of the aperture.  
        diameter : float, meters = None
            The diameter of the aperture in meters. 
        coordinates : Array, meters = None
            The coordinate system to calculate the aperture on.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        name: str = 'StaticAberratedAperture'
            The name of the layer, which is used to index the layers dictionary.
        """
        # Ensure correct aperture types
        if not isinstance(aperture, 
            (AberratedAperture, CompoundAperture, MultiAperture)) or \
                (isinstance(aperture, (CompoundAperture, MultiAperture)) and \
                    len(aperture._aberrated_apertures()) == 0):
            raise TypeError("The provided aperture must have aberrations.")
        
        # Input check: Coordinates provided
        if coordinates is not None and \
            (npixels is not None or diameter is not None):
            raise ValueError("If coordinates is specified npixels and " + \
                "diameter can not be provided.")
        # Input check: Coordinates not provided
        elif coordinates is None and \
            (npixels is None or diameter is None):
            raise ValueError("both npixels and diameter must be provided.")
        
        # Generate coordinates if not provided
        if coordinates is None:
            npixels_in = (npixels, npixels)
        pixel_scales = (diameter / npixels, diameter / npixels)
        coordinates = get_pixel_positions(npixels_in, pixel_scales)

        super().__init__(aperture=aperture, coordinates=coordinates, 
            coefficients=aperture.coefficients, name=name)
        self.basis = aperture._basis(coordinates)


    def __call__(self : ApertureLayer, wavefront : Wavefront) -> Wavefront:
        """
        Apply the aperture to the wavefront.

        Parameters
        ----------
        wavefront: Wavefront
            The wavefront that is passing through the aperture.

        Returns
        -------
        wavefront: Wavefront
            The wavefront after passing through the aperture
        """

        # Calculate and update amplitude and phase
        phase = wavefront.phase + self._opd() * wavefront.wavenumber
        amplitude = wavefront.amplitude * self.transmission
        wavefront = wavefront.set(['amplitude', 'phase'], [amplitude, phase])
        if self.normalise:
            return wavefront.normalise()
        return wavefront


    def _basis(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the basis vectors of the aperture aberrations.

        Parameters
        ----------
        coordinates : Array, meters
            The coordinate system to calculate the basis vectors on.

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        return self.basis


    def get_basis(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the basis vectors of the aperture aberrations.

        Returns
        -------
        basis : Array 
            The array of the basis vectors of the aperture aberrations.
        """
        return self._basis()


    def _opd(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the total optical path difference of the aperture aberrations.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        if self.coefficients.ndim == 1:
            return (self.basis * self.coefficients[:, None, None]).sum(0)
        else:
            return (self.basis * \
                self.coefficients[:, :, None, None]).sum((0, 1))


    def get_opd(self : ApertureLayer, **kwargs) -> Array:
        """
        Compute the total optical path difference of the aperture aberrations.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        return self._opd()
    

    @property
    def opd(self : ApertureLayer) -> Array:
        """
        Return the total optical path difference of the aperture aberrations.

        Returns
        -------
        basis : Array 
            The array of the total opd of the aperture aberrations.
        """
        return self._opd()
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str: # pragma: no cover
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"Applies a pre-calculated Static Aperture with "
                f"{len(self.coefficients)} aberrations.")


###############
### Factory ###
###############
class ApertureFactory():
    """
    This class is not actually ever instatiated, but is rather a class used to 
    give a simple constructor interface that is used to construct the most
    commonly used apertures. It is able to construct hard-edged circular or 
    regular poygonalal apertures. Secondary mirrors obscurations with the same
    aperture shape can be constructed, along with uniformly spaced struts. The
    ratio of the primary aperture opening to the array size is determined by
    the `aperture_ratio` parameter, with secondary mirror obscurations and
    struts being scaled relative to the aperture diameter. 

    Lets look at an example of how to construct a simple circular aperture with
    a secondary mirror obscurtion held by 4 struts. For this example lets take
    a 2m diameter aperutre, with a 20cm secondary mirror held by 3 struts with
    a width of 2cm. In this example the secondary mirror is 10% of the primary
    aperture diameter and the struts are 1% of the primary aperture diameter,
    giving us values of 0.1 and 0.01 for the `secondary_ratio` and
    `strut_ratio` parameters. Let calcualte this for a 512x512 array with the
    aperture spanning the full array.

    ```python
    from dLux import ApertureFactory
    import jax.numpy as np
    import jax.random as jr

    # Construct aperture
    aperture = ApertureFactory(512, secondary_ratio=0.1, nstruts=4, 
        strut_ratio=0.01)
    ```
    
    The resulting aperture class has onc parameters, `.transmission` which
    represents the transmission of the aperture.
    ```

    We can also easily change this to a hexagonal aperture with 3 struts:

    ```python
    # Make aperture
    hexagonal_aperture = ApertureFactory(512, nsides=6, secondary_ratio=0.1,
        nstruts=3, strut_ratio=0.01)
    
    # Examine
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(aperture.transmission)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(hexagonal_aperture.transmission)
    plt.colorbar()
    plt.show()
    ```
    """
    def __new__(cls              : ApertureFactory, 
                npixels          : int, 
                aperture_ratio   : float = 1.0,
                secondary_ratio  : float = 0.,
                nsides           : int   = 0,
                secondary_nsides : int   = 0,
                rotation         : float = 0., 
                nstruts          : int   = 0,
                strut_ratio      : float = 0.,
                strut_rotation   : float = 0.,
                normalise        : bool  = True,
                name             : str   = 'Aperture'):
        """
        Constructs a basic single static aperture.

        Parameters
        ----------
        npixels : int
            Number of pixels used to represent the aperture.
        aperture_ratio : float = 1.
            The ratio of the aperture size to the array size. A value of 1. 
            results in an aperture that fully spans the array, a value of 0.5 
            retuls in an aperure that is half the size of the array, which is 
            equivilent to a padding factor of 2.
        secondary_ratio : float = 0.
            The ratio of the secondary mirror obsuration diameter to the 
            aperture diameter. A value of 0. results in no secondary mirror 
            obsuration.
        nsides : int = 0
            Number of sides of the aperture. A zero input results in a circular
            aperture. All other other values of three and above are supported.
        secondary_nsides : int = 0
            The number of sides of the secondary mirror obsuration. A zero input
            results in a circular aperture. All other other values of three and 
            above are supported.
        rotation : float, radians = 0
            The global rotation of the aperture in radians.
        nstruts : int = 0
            The number of uniformly spaced struts holding the secondary mirror. 
        strut_ratio : float = 0.
            The ratio of the width of the strut to the aperture diameter.
        strut_rotation : float = 0
            The rotation of the struts in radians.
        name : str = 'Aperture'
            The name of the aperture used to index the layers dictionary.
        
        Returns
        -------
        aperture : StaticAperture
            Returns an appropriately constructed StaticAperture.
        """
        # Check vaid inputs
        if nsides < 3 and nsides != 0:
            raise ValueError("nsides must be either 0 or >=3")
        
        if secondary_nsides < 3 and secondary_nsides != 0:
            raise ValueError("secondary_nsides must be either 0 or >=3")
        
        if aperture_ratio <= 0:
            raise ValueError("aperture_ratio must be > 0")
        
        if secondary_ratio < 0:
            raise ValueError("secondary_ratio must be >= 0")
        
        if strut_ratio < 0:
            raise ValueError("strut_ratio must be >= 0")

        # Construct components
        apertures = []

        # Circular Primary
        if nsides == 0:
            apertures.append(CircularAperture(aperture_ratio/2, softening=0))
        # Polygonal Primary
        else: 
            apertures.append(RegularPolygonalAperture(
                nsides, aperture_ratio/2, softening=0, rotation=rotation))

        # Secondary
        if secondary_ratio != 0:
            secondary_rel = aperture_ratio * secondary_ratio

            # Circular
            if secondary_nsides == 0: 
                apertures.append(CircularAperture(
                    secondary_rel/2, softening=0, occulting=True))
            # Polygonal
            else: 
                apertures.append(RegularPolygonalAperture(secondary_nsides, 
                    secondary_rel/2, softening=0, rotation=rotation, 
                        occulting=True))
        
        # Spiders
        if nstruts > 0:
            if strut_ratio == 0:
                raise ValueError("strut_ratio must be > 0 if nstruts > 0")
            strut_rel = aperture_ratio * strut_ratio
            full_rotation = strut_rotation + rotation
            apertures.append(UniformSpider(
                nstruts, strut_rel, rotation=full_rotation, softening=0))

        # Construct CompoundAperture
        full_aperture = CompoundAperture(apertures, normalise=normalise)
        static = StaticAperture(full_aperture, npixels, 1, name=name)

        return static


    def __init__(self             : ApertureFactory, 
                 npixels          : int, 
                 aperture_ratio   : float = 1.0,
                 secondary_ratio  : float = 0.,
                 nsides           : int   = 0,
                 secondary_nsides : int   = 0,
                 rotation         : float = 0., 
                 nstruts          : int   = 0,
                 strut_ratio      : float = 0.,
                 strut_rotation   : float = 0.,
                 normalise        : bool  = True,
                 name             : str   = 'Aperture'):
        """
        Constructs a basic single static aperture.

        Parameters
        ----------
        npixels : int
            Number of pixels used to represent the aperture.
        aperture_ratio : float = 1.
            The ratio of the aperture size to the array size. A value of 1. 
            results in an aperture that fully spans the array, a value of 0.5 
            retuls in an aperure that is half the size of the array, which is 
            equivilent to a padding factor of 2.
        secondary_ratio : float = 0.
            The ratio of the secondary mirror obsuration diameter to the 
            aperture diameter. A value of 0. results in no secondary mirror 
            obsuration.
        nsides : int = 0
            Number of sides of the aperture. A zero input results in a circular
            aperture. All other other values of three and above are supported.
        secondary_nsides : int = 0
            The number of sides of the secondary mirror obsuration. A zero input
            results in a circular aperture. All other other values of three and 
            above are supported.
        rotation : float, radians = 0
            The global rotation of the aperture in radians.
        nstruts : int = 0
            The number of uniformly spaced struts holding the secondary mirror. 
        strut_ratio : float = 0.
            The ratio of the width of the strut to the aperture diameter.
        strut_rotation : float = 0
            The rotation of the struts in radians.
        name : str = 'Aperture'
            The name of the aperture used to index the layers dictionary.
        
        Returns
        -------
        aperture : StaticAperture
            Returns an appropriately constructed StaticAperture.
        """