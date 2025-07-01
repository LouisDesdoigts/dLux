from __future__ import annotations
from abc import abstractmethod
from typing import Any
from equinox import filter
from jax import Array, numpy as np
import jax.tree as jtu
import dLux.utils as dlu

from ..wavefronts import Wavefront
from ..transformations import CoordTransform
from .optical_layers import OpticalLayer, BasisLayer
from .aberrations import ZernikeBasis


__all__ = [
    "CircularAperture",
    "RectangularAperture",
    "RegPolyAperture",
    "Spider",
    "SquareAperture",
    "AberratedAperture",
    "CompoundAperture",
    "MultiAperture",
]


class ApertureLayer(OpticalLayer):
    """
    Base ApertureLayer class, instantiates the normalise attribute.

    Attributes
    ----------
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    normalise: bool

    def __init__(self: OpticalLayer, normalise: bool = False, **kwargs):
        """
        Parameters
        ----------
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        self.normalise = bool(normalise)
        super().__init__(**kwargs)

    @abstractmethod
    def transmission(self, coords, pixel_scale):  # pragma: no cover
        """
        Evaluates the aperture transmission on the given coords, applying the aperture
        transformations to the coords.
        """


class BaseDynamicAperture(ApertureLayer):
    """
    An abstract base class that implements the coordinate transformation attribute.

    Attributes
    ----------
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    transformation: CoordTransform

    def __init__(
        self: ApertureLayer,
        transformation: CoordTransform = None,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        normalise : bool = False
            Should the aperture normalise the wavefront after being applied.
        """
        super().__init__(normalise=normalise)
        if transformation is not None:
            if not isinstance(transformation, CoordTransform):
                raise TypeError(
                    "transformation must be a CoordTransform object. "
                    "Use the CoordTransform class to create one."
                )
        self.transformation = transformation

    def __getattr__(self: ApertureLayer, key: str) -> Any:
        """Raises transformation attributes to the ApertureLayer level."""
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        raise AttributeError(f"{key} not in {self.__class__.__name__}")

    def apply(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        # Apply aperture
        wavefront *= self.transmission(
            wavefront.coordinates, wavefront.pixel_scale
        )
        if self.normalise:
            return wavefront.normalise()
        return wavefront


class DynamicAperture(BaseDynamicAperture):
    """
    An abstract base class that implements the methods required to provide dynamic soft
    edges to the apertures, and the ability to occult the aperture.

    Attributes
    ----------
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softness: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    occulting: bool
    softness: float

    def __init__(
        self: ApertureLayer,
        transformation: CoordTransform = None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: float, pixels = 1.0
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        super().__init__(
            transformation=transformation,
            normalise=normalise,
        )
        self.occulting = bool(occulting)
        self.softness = float(softening)
        if self.softness <= 0:
            raise ValueError("softening must be greater than 0.")

    @abstractmethod  # pragma: no cover
    def extent(self):
        pass


#############################
# Concrete Aperture Classes #
#############################
class CircularAperture(DynamicAperture):
    """
    A dynamically generated circular aperture parameterised by its radius. Both jit and
    grad compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/CircularAperture.png)

    Attributes
    ----------
    radius: float, meters
        The radius of the aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
    softening: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    radius: float

    def __init__(
        self: ApertureLayer,
        radius: float,
        transformation: CoordTransform = None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        radius: Array, meters
            The radius of the aperture.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )
        self.radius = float(radius)

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_circle(coords, self.radius, clip_val, self.occulting)

    @property
    def extent(self: ApertureLayer) -> float:
        """
        Returns the maximum extent of the aperture.

        Returns
        -------
        extent : float
            The maximum extent of the aperture.
        """
        return self.radius

    @property
    def nsides(self: ApertureLayer) -> int:
        """
        Returns the number of sides of the aperture.

        Returns
        -------
        nsides : int
            The number of sides of the aperture.
        """
        return 0


class SquareAperture(DynamicAperture):
    """
    A dynamically generated square aperture parameterised by it side length.
    Both jit and grad compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/SquareAperture.png)

    Attributes
    ----------
    width: float, meters
        The side length of the aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    width: float

    def __init__(
        self: ApertureLayer,
        width: float,
        transformation: CoordTransform = None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        width: Array, meters
            The side length of the aperture.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: float, pixels = 1.0
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_square(coords, self.width, clip_val, self.occulting)

    @property
    def extent(self: ApertureLayer) -> float:
        """
        Returns the maximum extent of the aperture.

        Returns
        -------
        extent : float
            The maximum extent of the aperture.
        """
        return np.sqrt(2) * self.width

    @property
    def nsides(self: ApertureLayer) -> int:
        """
        Returns the number of sides of the aperture.

        Returns
        -------
        nsides : int
            The number of sides of the aperture.
        """
        return 4


class RectangularAperture(DynamicAperture):
    """
    A dynamically generated rectangular aperture parameterised by it width and height.
    Both jit and grad compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/RectangularAperture.png)

    Attributes
    ----------
    height: float, meters
        The length of the aperture in the y-direction.
    width: float, meters
        The length of the aperture in the x-direction.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    height: float
    width: float

    def __init__(
        self: ApertureLayer,
        height: float,
        width: float,
        transformation: CoordTransform = None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        height: Array, meters
            The length of the aperture in the y-direction.
        width: Array, meters
            The length of the aperture in the x-direction.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: float, pixels = 1.0
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        self.height = float(height)
        self.width = float(width)

        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_rectangle(
            coords, self.width, self.height, clip_val, self.occulting
        )

    @property
    def extent(self: ApertureLayer) -> float:
        """
        Returns the maximum extent of the aperture.

        Returns
        -------
        extent : float
            The maximum extent of the aperture.
        """
        return np.hypot(self.height / 2.0, self.width / 2.0)

    @property
    def nsides(self: ApertureLayer) -> int:
        """
        Returns the number of sides of the aperture.

        Returns
        -------
        nsides : int
            The number of sides of the aperture.
        """
        return 4


class RegPolyAperture(DynamicAperture):
    """
    Creates a dynamically generated regular polygon aperture parameterised by its
    number of sides and maximum radius. Both jit and grad compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/RegPolyAperture.png)

    Attributes
    ----------
    nsides: int
        The number of sides of the aperture.
    rmax: float, meters
        The maximum radius to the vertices from its center.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    nsides: int
    rmax: float

    def __init__(
        self: ApertureLayer,
        nsides: int,
        rmax: float,
        transformation: CoordTransform = None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        nsides: int
            The number of sides of the aperture.
        rmax: float, meters
            The maximum radius to the vertices from its center.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: float, pixels = 1.0
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        self.nsides = int(nsides)
        self.rmax = float(rmax)

        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_reg_polygon(
            coords, self.rmax, self.nsides, clip_val, self.occulting
        )

    @property
    def extent(self: ApertureLayer) -> float:
        """
        Returns the maximum extent of the aperture.

        Returns
        -------
        extent : float
            The maximum extent of the aperture.
        """
        return self.rmax


class Spider(DynamicAperture):
    """
    Creates a dynamically generated spider aperture parameterised by its arm width and
    number of arms. Both jit and grad compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/Spider.png)

    Attributes
    ----------
    width: float, meters
        The width of the spider.
    angles: Array, degrees
        The angle of each arm of the spider.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: float, pixels
        The approximate pixel width of the soft boundary applied to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    width: float
    angles: Array

    def __init__(
        self: ApertureLayer,
        width: float,
        angles: Array,
        transformation: CoordTransform = None,
        occulting: bool = True,
        softening: float = 1.0,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        width: float, meters
            The width of the spider.
        angles: Array, degrees
            The angle of each arm of the spider.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        occulting: bool = True
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: float, pixels = 1.0
            The approximate pixel width of the soft boundary applied to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)
        self.angles = np.asarray(angles, dtype=float)

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_spider(
            coords, self.width, self.angles, clip_val, self.occulting
        )

    @property
    def extent(self: ApertureLayer) -> float:
        """
        Returns the maximum extent of the aperture.

        Returns
        -------
        extent : float
            The maximum extent of the aperture.
        """
        raise TypeError("Spiders do not have an extent.")

    @property
    def nsides(self: ApertureLayer) -> int:
        """
        Returns the number of sides of the aperture.

        Returns
        -------
        nsides : int
            The number of sides of the aperture.
        """
        raise TypeError("Spiders do not have a number of sides.")


# TODO: Should this not have transmission as it is held by its sub-aperture?
class AberratedAperture(BasisLayer, ApertureLayer):
    """
    Creates a dynamically generated Aperture with aberrations. Both jit and grad
    compatible.

    ??? abstract "UML"
        ![UML](../../assets/uml/AberratedAperture.png)

    Attributes
    ----------
    aperture: ApertureLayer
        The aperture on which the aberration basis is defined.
    basis: list[Zernike]
        A list of basis functions that generate the basis vectors.
    coefficients: Array
        The amplitude of each basis vector of the aberrations.
    as_phase : bool
        Whether to apply the basis as a phase phase or OPD. If True the basis is
        applied as a phase, else it is applied as an OPD.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    aperture: ApertureLayer

    def __init__(
        self: ApertureLayer,
        aperture: ApertureLayer,
        noll_inds: Array,
        coefficients: Array = None,
        as_phase: bool = False,
    ):
        """
        Parameters
        ----------
        aperture: ApertureLayer
            The aperture on which the aberration basis is defined.
        noll_inds: Array
            The noll indices of the basis functions to use.
        coefficients: Array = None
            The amplitude of each basis vector of the aberrations.
        as_phase : bool = False
            Whether to apply the basis as a phase phase or OPD. If True the basis is
            applied as a phase, else it is applied as an OPD.
        """
        # Ensure aperture is dynamic
        if not isinstance(aperture, DynamicAperture):
            raise TypeError(
                "AberratedApertures can not contain Static, "
                "Compound or Multi Apertures. AberratedApertures can be "
                "placed in Compound or Multi Apertures, which can then be "
                "promoted to Static."
            )

        super().__init__(normalise=aperture.normalise, as_phase=as_phase)

        # Ensure transmissive
        if aperture.occulting:
            raise TypeError("AberratedApertures can not be occulting.")

        if isinstance(aperture, Spider):
            raise TypeError("AberratedApertures can not be spiders.")

        # Set Aperture
        self.aperture = aperture
        self.basis = ZernikeBasis(noll_inds)

        if coefficients is None:
            coefficients = np.zeros(len(noll_inds))
        self.coefficients = np.asarray(coefficients, dtype=float)

    def calculate(self: ApertureLayer):
        """
        Required abstract method. Raises NotImplementedError as it is invalid here.
        """
        raise NotImplementedError(
            "Aberrated Apertures can not use the .calculate() method because "
            "they need coords to be generated on. please use "
            ".eval_basis(coords) method instead."
        )

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        return self.aperture.transmission(coords, pixel_scale)

    def calc_basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Calculates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the basis vectors on.

        Returns
        -------
        basis : Array
            The basis vectors at the given coordinates.
        """
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation.apply(coords)
        coords /= self.aperture.extent
        return self.basis.calculate_basis(coords, self.aperture.nsides)

    def eval_basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Evaluates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to evaluate the basis vectors on.

        Returns
        -------
        aberrations : Array
            The aberrations at the given coordinates.
        """
        basis = self.calc_basis(coords)
        return dlu.eval_basis(basis, self.coefficients)

    def apply(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        # Transmission
        wavefront *= self.transmission(
            wavefront.coordinates, wavefront.pixel_scale
        )
        if self.normalise:
            wavefront = wavefront.normalise()

        # Transform coordinate
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation.apply(wavefront.coordinates)
        else:
            coords = wavefront.coordinates

        # Aberrations
        aberrations = self.eval_basis(coords)
        if self.as_phase:
            wavefront = wavefront.add_phase(aberrations)
        else:
            wavefront += aberrations
        return wavefront


class CompositeAperture(BaseDynamicAperture):
    """
    Base class for dynamically generated Apertures that are a combination of various
    different aperture shapes. Both jit and grad compatible.

    Attributes
    ----------
    apertures: dict
       The sub-apertures that make up the full aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    apertures: dict

    def __init__(
        self: ApertureLayer,
        apertures: list,
        transformation: CoordTransform = None,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        apertures: list[ApertureLayer]
            The sub-apertures that make up the full aperture.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        self.apertures = dlu.list2dictionary(apertures, False, ApertureLayer)
        super().__init__(
            transformation=transformation,
            normalise=normalise,
        )

    def __getattr__(self: ApertureLayer, key: str) -> Any:
        """
        Raises the contained apertures via their dictionary keys.

        Parameters
        ----------
        key: str
            The attribute to get.

        Returns
        -------
        attribute: Any
            The aperture found at the given key.
        """
        if key in list(self.apertures.keys()):
            return self.apertures[key]
        else:
            raise AttributeError(f"{key} not in {self.apertures.keys()}")

    def _aberrated_apertures(self: ApertureLayer) -> list[ApertureLayer]:
        """
        Returns the list of aberrated apertures, from the full set of apertures.

        Returns
        -------
        apertures : list
            The list of aberrated apertures.
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
        # TODO: use partition
        filter_map = jtu.map(
            is_aberrated, self.apertures, is_leaf=is_aberrated
        )
        aberrated = filter(self.apertures, filter_map)
        return jtu.flatten(aberrated, is_leaf=is_aberrated)[0]

    @property
    def coefficients(self: ApertureLayer) -> list[Array]:
        """
        Returns the coefficients of the aberrated apertures.

        Returns
        -------
        coefficients : list[Array]
            The coefficients of the aberrated apertures.
        """
        return [ap.coefficients for ap in self._aberrated_apertures()]

    def calc_basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Calculates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the basis vectors on.

        Returns
        -------
        basis : Array
            The basis vectors at the given coordinates.
        """
        aberrated_apertures = self._aberrated_apertures()
        basis_fn = lambda ap: ap.calc_basis(coords)
        leaf_fn = lambda ap: isinstance(ap, AberratedAperture)
        basii = jtu.map(basis_fn, aberrated_apertures, is_leaf=leaf_fn)
        return jtu.flatten(basii)[0]

    def eval_basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Evaluates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to evaluate the basis vectors on.

        Returns
        -------
        aberrations : Array
            The aberrations at the given coordinates.
        """
        basii = self.calc_basis(coords)
        coeffs = self.coefficients
        eval_fn = lambda basis, coeff: dlu.eval_basis(basis, coeff)
        return np.array(jtu.map(eval_fn, basii, coeffs))

    def transmissions(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        eval_fn = lambda ap: ap.transmission(coords, pixel_scale)
        leaf_fn = lambda ap: isinstance(ap, ApertureLayer)
        transmissions = jtu.map(eval_fn, self.apertures, is_leaf=leaf_fn)
        return np.squeeze(np.array(jtu.flatten(transmissions)[0]))

    def apply(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        # Transmission
        wavefront *= self.transmission(
            wavefront.coordinates, wavefront.pixel_scale
        )
        if self.normalise:
            return wavefront.normalise()

        # Transform coordinate
        if self.transformation is not None:
            coords = self.transformation.apply(wavefront.coordinates)
        else:
            coords = wavefront.coordinates

        # Aberrations
        aberrations = self.eval_basis(coords)
        # TODO: Add add_phase as an option
        # if self.as_phase:
        #     wavefront = wavefront.add_phase(aberrations)
        # else:
        wavefront += aberrations
        return wavefront


class CompoundAperture(CompositeAperture):
    """
    Dynamically generates an Apertures from a series of overlapping sub-apertures. Both
    jit and grad compatible.

    This class combines the aperture via a _multiplication_ of the sub-apertures. An
    example would be a HST-like aperture with an obscuring secondary mirror and spiders.

    This class can only contain a _single_ AberratedAperture.

    If you want to combine apertures via an _addition_ of the sub-apertures such as an
    aperture mask, use the MultiAperture class.

    Note that this class can not contain a MultiAperture, but MultiApertures can
    contain CompoundApertures.

    ??? abstract "UML"
        ![UML](../../assets/uml/CompoundAperture.png)

    Attributes
    ----------
    apertures: dict
        The sub-apertures that make up the full aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    def __init__(
        self: ApertureLayer,
        apertures: list,
        transformation: CoordTransform = None,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        apertures: list[ApertureLayer]
            The sub-apertures that make up the full aperture.
        transformation: CoordTransform
            The object that applies the coordinate transformations to the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the aperture.
        """
        super().__init__(
            apertures=apertures,
            transformation=transformation,
            normalise=normalise,
        )

        # Check for more than one aberrated aperture
        naberrated = 0
        for aperture in self.apertures.values():
            if isinstance(aperture, AberratedAperture):
                naberrated += 1
            if naberrated > 1:
                raise TypeError(
                    "CompoundAperture can only have a single "
                    "AberratedAperture."
                )

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        return self.transmissions(coords, pixel_scale).prod(0)


class MultiAperture(CompositeAperture):
    """
    Dynamically generates an Apertures from a series of separated sub-apertures. Both
    jit and grad compatible.

    This class combines the aperture via an _addition_ of the sub-apertures. An example
    would be a aperture mask with multiple holes.

    This class can only contain a _multiple_ AberratedAperture, or CompoundApertures.

    If you want to combine apertures via a _multiplication_ of the sub-apertures such
    as HST-like aperture, use the CompoundAperture class.

    ??? abstract "UML"
        ![UML](../../assets/uml/MultiAperture.png)

    Attributes
    ----------
    apertures: dict
        The sub-apertures that make up the full aperture.
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    normalise : bool
        Whether to normalise the wavefront after passing through the aperture.
    """

    def eval_basis(self: ApertureLayer, coords: Array) -> Array:
        """
        Evaluates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to evaluate the basis vectors on.

        Returns
        -------
        aberrations : Array
            The aberrations at the given coordinates.
        """
        return super().eval_basis(coords).sum(0)

    def transmission(
        self: ApertureLayer, coords: Array, pixel_scale: float
    ) -> Array:
        """
        Calculates the transmission of the aperture at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the transmission on.
        pixel_scale : float
            The pixel scale of the coordinates.

        Returns
        -------
        transmission : Array
            The transmission of the aperture at the given coordinates.
        """
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        return self.transmissions(coords, pixel_scale).sum(0)
