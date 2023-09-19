from __future__ import annotations
from abc import abstractmethod
from typing import Any
from equinox import filter
from jax import Array, numpy as np
from jax.tree_util import tree_flatten, tree_map
import dLux.utils as dlu

from ..wavefronts import Wavefront
from .transformations import CoordTransform
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

    Attributes
    ----------
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    normalise: bool

    def __init__(self: OpticalLayer, normalise: bool = False, **kwargs):
        """
        Parameters
        ----------
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.normalise = bool(normalise)
        super().__init__(**kwargs)

    @abstractmethod
    def transmission(self, coords, pixel_scale):  # pragma: no cover
        """Evaluates the aperture transmission on the given coords,
        applying the aperture transformations to the coords."""
        pass


class BaseDynamicAperture(ApertureLayer):
    """

    Attributes
    ----------
    transformation: CoordTransform
        The object that applies the coordinate transformations to the aperture.
    """

    transformation: CoordTransform

    def __init__(
        self: ApertureLayer,
        transformation=None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the BaseDynamicAperture class.

        Parameters
        ----------
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

    def __getattr__(self, key):
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
        # Transform coordinate
        if self.transformation is not None:
            coords = self.transformation.apply(wavefront.coordinates)
        else:
            coords = wavefront.coordinates

        # Apply aperture
        wavefront *= self.transmission(coords, wavefront.pixel_scale)
        if self.normalise:
            return wavefront.normalise()
        return wavefront


class DynamicAperture(BaseDynamicAperture):
    """
    An abstract base class that implements the methods required to provide soft
    edges to the apertures and generate either transmissive or occulting
    apertures.

    Attributes
    ----------
    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.

    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    occulting: bool
    softness: float

    def __init__(
        self: ApertureLayer,
        transformation=None,
        occulting: bool = False,
        softening: Array = 1.0,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the DynamicAperture class.

        Parameters
        ----------

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        super().__init__(
            transformation=transformation,
            normalise=normalise,
        )
        self.occulting = bool(occulting)
        self.softness = float(softening)

    @abstractmethod  # pragma: no cover
    def extent(self):
        pass


#############################
# Concrete Aperture Classes #
#############################
class CircularAperture(DynamicAperture):
    """
    A circular aperture parameterised by its radius.

    Attributes
    ----------
    radius: Array, metres
        The radius of the aperture.

    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    radius: Array

    def __init__(
        self: ApertureLayer,
        radius,
        transformation=None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ) -> Array:
        """
        Constructor for the CircularAperture class.

        Parameters
        ----------
        radius: Array, metres
            The radius of the aperture.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.radius = float(radius)

    def transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        print(coords.shape)
        return dlu.soft_circle(coords, self.radius, clip_val, self.occulting)

    @property
    def extent(self):
        return self.radius

    @property
    def nsides(self):
        return 0


class SquareAperture(DynamicAperture):
    """
    A square aperture parameterised by its width.

    Attributes
    ----------
    width: Array, metres
        The length of the aperture in the x-direction.

    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    width: Array

    def __init__(
        self: ApertureLayer,
        width: float,
        transformation=None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ) -> Array:
        """
        Constructor for the SquareAperture class.

        Parameters
        ----------
        width: Array, metres
            The length of the aperture in the x-direction.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)

    def transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_square(coords, self.width, clip_val, self.occulting)

    @property
    def extent(self):
        return np.sqrt(2) * self.width

    @property
    def nsides(self):
        return 4


class RectangularAperture(DynamicAperture):
    """
    A rectangular aperture parameterised by it height and width.

    Attributes
    ----------
    height: Array, metres
        The length of the aperture in the y-direction.
    width: Array, metres
        The length of the aperture in the x-direction.

    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    height: Array
    width: Array

    def __init__(
        self: ApertureLayer,
        height: Array,
        width: Array,
        transformation=None,
        occulting: bool = False,
        softening: float = 1.0,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the RectangularAperture class.

        Parameters
        ----------
        height: Array, metres
            The length of the aperture in the y-direction.
        width: Array, metres
            The length of the aperture in the x-direction.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.height = float(height)
        self.width = float(width)

        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_rectangle(
            coords, self.width, self.height, clip_val, self.occulting
        )

    @property
    def extent(self):
        return np.hypot(self.height / 2.0, self.width / 2.0)

    @property
    def nsides(self):
        return 4


class RegPolyAperture(DynamicAperture):
    """
    A regular polygonal aperture defined by its number of sides and the maximum
    radius to the vertices from its center.

    Attributes
    ----------
    nsides: int
        The number of sides of the aperture.
    rmax: Array, metres
        The maximum radius to the vertices from its center.

    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    nsides: int
    rmax: Array

    def __init__(
        self: ApertureLayer,
        nsides: int,
        rmax: Array,
        transformation=None,
        occulting: bool = False,
        softening: Array = 1.0,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the RegPolyAperture class.

        Parameters
        ----------
        nsides: int
            The number of sides of the aperture.
        rmax: Array, metres
            The maximum radius to the vertices from its center.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.nsides = int(nsides)
        self.rmax = float(rmax)

        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_reg_polygon(
            coords, self.rmax, self.nsides, clip_val, self.occulting
        )

    @property
    def extent(self):
        return self.rmax


class Spider(DynamicAperture):
    """
    A spider aperture defined by its width and number of arms.

    Attributes
    ----------
    width: Array, metres
        The width of the spider.
    angles: int
        The number of arms of the spider.

    occulting: bool
        Is the aperture occulting or transmissive. False results in a
        transmissive aperture, and True results in an occulting aperture.
    softening: Array, pixels
        The approximate pixel width of the soft boundary applied to the
        aperture. Hard edges can be achieved by setting the softening to 0.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    width: Array
    angles: int

    def __init__(
        self: ApertureLayer,
        width: float,
        angles: Array,
        transformation=None,
        occulting: bool = True,
        softening: Array = np.array(1.0),
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the Spiders class.

        Parameters
        ----------
        width: Array, metres
            The width of the spider.
        angles: int
            The number of arms of the spider.

        occulting: bool = False
            Is the aperture occulting or transmissive. False results in a
            transmissive aperture, and True results in an occulting aperture.
        softening: Array, pixels = np.array(1.)
            The approximate pixel width of the soft boundary applied to the
            aperture. Hard edges can be achieved by setting the softening to 0.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        super().__init__(
            transformation=transformation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)
        self.angles = np.asarray(angles, dtype=float)

    def transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_spider(
            coords, self.width, self.angles, clip_val, self.occulting
        )

    @property
    def extent(self):
        raise TypeError("Spiders do not have an extent.")

    @property
    def nsides(self):
        raise TypeError("Spiders do not have a number of sides.")


class AberratedAperture(BasisLayer, ApertureLayer):
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
    normalise: bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    aperture: ApertureLayer

    def __init__(
        self: ApertureLayer,
        aperture: ApertureLayer,
        noll_inds: Array,
        coefficients: Array = None,
        as_phase: bool = False,
    ) -> ApertureLayer:
        """
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

    def calculate(self):
        raise NotImplementedError(
            "Aberrated Apertures can not use the .calculate() method because "
            "they need coords to be generated on. please use "
            ".eval_basis(coords) method instead."
        )

    def transmission(self, coords, pixel_scale):
        return self.aperture.transmission(coords, pixel_scale)

    def calc_basis(self, coords):
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation.apply(coords)
        coords /= self.aperture.extent
        return self.basis.calculate_basis(coords, self.aperture.nsides)

    def eval_basis(self, coords):
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
        # Transform coordinate
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation.apply(wavefront.coordinates)
        else:
            coords = wavefront.coordinates

        # Transmission
        wavefront *= self.transmission(coords, wavefront.pixel_scale)
        if self.normalise:
            return wavefront.normalise()

        # Aberrations
        aberrations = self.eval_basis(coords)
        if self.as_phase:
            wavefront = wavefront.add_phase(aberrations)
        else:
            wavefront += aberrations
        return wavefront


class CompositeAperture(BaseDynamicAperture):
    """
    An abstract class used to combine multiple apertures so that more complex
    apertures can have global transformations applied to them. Two examples
    would be a pupil with spiders holding the secondary mirror or an aperture
    mask.

    Attributes
    ----------
    apertures: dict(str, Aperture)
       The sub-apertures that make up the full aperture.

    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    apertures: dict

    def __init__(
        self: ApertureLayer,
        apertures: list,
        transformation=None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the CompositeAperture class.

        Parameters
        ----------
        apertures: dict(str, Aperture)
            The sub-apertures that make up the full aperture.

        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.apertures = dlu.list2dictionary(apertures, False, ApertureLayer)
        super().__init__(
            transformation=transformation,
            normalise=normalise,
        )

    def __getattr__(self: ApertureLayer, key: str) -> Any:
        """
        Get the attribute of the aberrated apertures.

        Parameters
        ----------
        key: str
            The attribute to get.

        Returns
        -------
        attribute: Any
            The attribute of the aberrated apertures.
        """
        if key in list(self.apertures.keys()):
            return self.apertures[key]
        else:
            raise AttributeError(f"{key} not in {self.apertures.keys()}")

    def _aberrated_apertures(self: ApertureLayer) -> list:
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
        filter_map = tree_map(
            is_aberrated, self.apertures, is_leaf=is_aberrated
        )
        aberrated = filter(self.apertures, filter_map)
        return tree_flatten(aberrated, is_leaf=is_aberrated)[0]

    @property
    def coefficients(self):
        return [ap.coefficients for ap in self._aberrated_apertures()]

    def calc_basis(self: ApertureLayer, coords: Array) -> Array:
        aberrated_apertures = self._aberrated_apertures()
        basis_fn = lambda ap: ap.calc_basis(coords)
        leaf_fn = lambda ap: isinstance(ap, AberratedAperture)
        basii = tree_map(basis_fn, aberrated_apertures, is_leaf=leaf_fn)
        return tree_flatten(basii)[0]

    def eval_basis(self: ApertureLayer, coords: Array) -> Array:
        basii = self.calc_basis(coords)
        coeffs = self.coefficients
        eval_fn = lambda basis, coeff: dlu.eval_basis(basis, coeff)
        return np.array(tree_map(eval_fn, basii, coeffs))

    def transmissions(
        self: ApertureLayer, coords: Array, pixel_scale
    ) -> Array:
        eval_fn = lambda ap: ap.transmission(coords, pixel_scale)
        leaf_fn = lambda ap: isinstance(ap, ApertureLayer)
        transmissions = tree_map(eval_fn, self.apertures, is_leaf=leaf_fn)
        return np.squeeze(np.array(tree_flatten(transmissions)[0]))

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
        # Coordinate transforms
        # Transform coordinate
        if self.transformation is not None:
            coords = self.transformation.apply(wavefront.coordinates)
        else:
            coords = wavefront.coordinates

        # Transmission
        wavefront *= self.transmission(coords, wavefront.pixel_scale)
        if self.normalise:
            return wavefront.normalise()

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
    A  class used to combine multiple apertures into a single coherent
    aperture. An example would be an aperture with spiders holding a secondary
    mirror.

    This class is distinct from the MultiAperture class in that the
    sub-apertures are combined by multiplying their respective transmissions
    together, i.e. the sub-apertures are overlapping.

    This class should not contain a MultiAperture, but MultiApertures can
    contain CompoundApertures.

    A single aberrated aperture can be placed into the set of apertures.

    Attributes
    ----------
    apertures: dict(str, Aperture)
        The sub-apertures that make up the full aperture.

    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    def __init__(
        self: ApertureLayer,
        apertures: list,
        transformation=None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the CompoundAperture class.

        Parameters
        ----------
        apertures: list[Aperture]
            The sub-apertures that make up the full aperture.

        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
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

    def transmission(self: ApertureLayer, coords: Array, pixel_scale) -> Array:
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        return self.transmissions(coords, pixel_scale).prod(0)


class MultiAperture(CompositeAperture):
    """
    A  class used to combine multiple apertures into a single coherent
    aperture. An example would be an aperture mask.

    This class is distinct from the CompoundAperture class in that the
    sub-apertures are combined by adding their respective transmissions
    together, i.e. the sub-apertures are not overlapping.

    This class can contain multiple CompoundApertures.

    Attributes
    ----------
    apertures: dict(str, Aperture)
        The sub-apertures that make up the full aperture.

    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    def eval_basis(self, coords):
        return super().eval_basis(coords).sum(0)

    def transmission(self: ApertureLayer, coords: Array, pixel_scale) -> Array:
        if self.transformation is not None:
            coords = self.transformation.apply(coords)
        return self.transmissions(coords, pixel_scale).sum(0)
