from __future__ import annotations

from abc import abstractmethod
from typing import Any

from equinox import filter
from jax import Array, numpy as np
from jax.tree_util import tree_flatten, tree_map
from zodiax import Base

import dLux
import dLux.utils as dlu


Wavefront = lambda: dLux.wavefronts.Wavefront
Optic = lambda: dLux.optical_layers.Optic
BasisOptic = lambda: dLux.optical_layers.BasisOptic
# OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
BasisLayer = lambda: dLux.optical_layers.BasisLayer
ZernikeBasis = lambda: dLux.aberrations.ZernikeBasis

from . import optical_layers


__all__ = [
    "CircularAperture",
    "RectangularAperture",
    "RegPolyAperture",
    "Spider",
    "SquareAperture",
    # "IrregPolyAperture",
    "AberratedAperture",
    "CompoundAperture",
    "MultiAperture",
    "ApertureFactory",
]


# Class to be held by dynamic apertures
class CoordTransform(Base):
    """
    A simple class to handle the coordinate transformations applied dynamic
    aperture classes. Transformations are applied in the order:
        1. Translation
        2. Rotation
        3. Compression
        4. Shear

    Attributes
    ----------
    translation: Array
        The (x, y) shift applied to the coords.
    rotation: Array
        The clockwise rotation applied to the coords.
    compression: Array
        The (x, y) compression applied to the coords.
    shear: Array
        The (x, y) shear applied to the coords.
    """

    translation: Array
    rotation: float
    compression: Array
    shear: Array

    def __init__(
        self: CoordTransform,
        translation: Array,
        rotation: float,
        compression: Array,
        shear: Array,
    ):
        """
        Parameters
        ----------
        translation: Array
            The (x, y) shift applied to the coords.
        rotation: float, radians
            The clockwise rotation applied to the coords.
        compression: Array
            The (x, y) compression applied to the coords.
        shear: Array
            The (x, y) shear applied to the coords.
        """
        if translation is not None:
            self.translation = np.asarray(translation, dtype=float)
            if self.translation.shape != (2,):
                raise ValueError("center must be have shape (2,).")
        else:
            self.translation = None

        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=float)
            if self.rotation.shape != ():
                raise ValueError("rotation must have shaoe ().")
        else:
            self.rotation = None

        if compression is not None:
            self.compression = np.asarray(compression, dtype=float)
            if self.compression.shape != (2,):
                raise ValueError("compression must have shape (2,).")
        else:
            self.compression = None

        if shear is not None:
            self.shear = np.asarray(shear, dtype=float)
            if self.shear.shape != (2,):
                raise ValueError("shear must be have shape (2,).")
        else:
            self.shear = None

    def calculate(self, npix, diam):
        """Generate the transformed coords from diameter and npix."""
        return self(dlu.pixel_coords(npix, diam))

    def __call__(self, coords):
        if self.translation is not None:
            coords = dlu.translate_coords(coords, self.translation)
        if self.shear is not None:
            coords = dlu.shear_coords(coords, self.shear)
        if self.compression is not None:
            coords = dlu.compress_coords(coords, self.compression)
        if self.rotation is not None:
            coords = dlu.rotate_coords(coords, self.rotation)
        return coords


class ApertureLayer(optical_layers.OpticalLayer):
    """

    Attributes
    ----------
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    normalise: bool

    def __init__(
        self: optical_layers.OpticalLayer, normalise: bool = False, **kwargs
    ):
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
    def _transmission(self, coords, pixel_scale):
        """Evaluates the aperture transmission on the given coords,
        applying the aperture transformations to the coords."""
        pass

    @abstractmethod
    def transmission(self, npix, diam):
        """Evaluates the aperture transmission on paraxial coords"""
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
        centre: Array = None,
        rotation: Array = None,
        compression: Array = None,
        shear: Array = None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the BaseDynamicAperture class.

        Parameters
        ----------
        centre: Array, metres = None
            The (x, y) coords of the centre of the aperture.
        rotation: Array, radians = None
            The clockwise rotation of the aperture.
        compression: Array = None
            The (x, y) compression of the aperture.
        shear: Array = None
            The (x, y) linear shear of the aperture.
        normalise : bool = False
            Should the aperture normalise the wavefront after being applied.
        """
        super().__init__(normalise=normalise)

        inputs = (centre, shear, compression, rotation)
        if all((input is None) for input in inputs):
            self.transformation = None
        else:
            self.transformation = CoordTransform(
                centre, shear, compression, rotation
            )

    def transmission(self, npix, diam):
        return self._transmission(dlu.pixel_coords(npix, diam), diam / npix)

    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
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
            coords = self.transformation(wavefront.coords)
        else:
            coords = wavefront.coords

        # Apply aperture
        wavefront *= self._transmission(coords, wavefront.pixel_scale)
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
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
        rotation: Array = None,
        occulting: bool = False,
        softening: Array = 1.0,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the DynamicAperture class.

        Parameters
        ----------
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
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
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            normalise=normalise,
        )
        self.occulting = bool(occulting)
        self.softness = float(softening)

    @abstractmethod
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
        radius: float,
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
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
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
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
            centre=centre,
            shear=shear,
            compression=compression,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.radius = float(radius)

    def _transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_circle(self.radius, coords, clip_val, self.occulting)

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
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
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
            centre=centre,
            shear=shear,
            compression=compression,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)

    def _transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_square(self.width, coords, clip_val, self.occulting)

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
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
        rotation: Array = None,
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
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def _transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_rectangle(
            self.width, self.height, coords, clip_val, self.occulting
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
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
        rotation: Array = None,
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
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
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
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    def _transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_reg_polygon(
            self.rmax,
            self.nsides,
            coords,
            clip_val,
            self.occulting,
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
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
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
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture
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
            centre=centre,
            shear=shear,
            compression=compression,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

        self.width = float(width)
        self.angles = np.asarray(angles, dtype=float)

    def _transmission(self, coords, pixel_scale):
        if self.transformation is not None:
            coords = self.transformation(coords)
        clip_val = pixel_scale * self.softness / 2
        return dlu.soft_spider(
            self.width, self.angles, coords, clip_val, self.occulting
        )

    @property
    def extent(self):
        raise TypeError(
            "Spiders are always occulting so do not have an extent."
        )


###############
# Aberrations #
###############
class AberratedAperture(ApertureLayer, optical_layers.BasisLayer):
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
        super().__init__(normalise=aperture.normalise, as_phase=as_phase)

        # Ensure aperture is dynamic
        if not isinstance(aperture, DynamicAperture):
            raise TypeError(
                "AberratedApertures can not contain Static, "
                "Compound or Multi Apertures. AberratedApertures can be "
                "placed in Compound or Multi Apertures, which can then be "
                "promoted to Static."
            )

        # Ensure transmissive
        if aperture.occulting:
            raise ValueError("AberratedApertures can not be occulting.")

        if isinstance(aperture, Spider):
            raise ValueError("AberratedApertures can not be spiders.")

        # Set Aperture
        self.aperture = aperture
        self.basis = ZernikeBasis()(noll_inds)

        if coefficients is None:
            coefficients = np.zeros(len(noll_inds))
        self.coefficients = np.asarray(coefficients, dtype=float)

    # Overwrite basis method since it won't work
    def calculate(self):
        raise NotImplementedError(
            "Aberrated Apertures can not use the .calculate() method because "
            "they need coords to be generated on. please use "
            ".eval_basis(coords) method instead."
        )

    def transmission(self, npix, diam):
        return self._transmission(dlu.pixel_coords(npix, diam), diam / npix)

    def _transmission(self, coords, pixel_scale):
        return self.aperture._transmission(coords, pixel_scale)

    def calc_basis(self, coords):
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation(coords)
        coords /= self.aperture.extent
        return self.basis.calculate_basis(coords, self.aperture.nsides)

    def eval_basis(self, coords):
        basis = self.calc_basis(coords)
        return dlu.eval_basis(basis, self.coefficients)

    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
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
            coords = self.transformation(wavefront.coords)
        else:
            coords = wavefront.coords

        # Transmission
        wavefront *= self._transmission(coords, wavefront.pixel_scale)
        if self.normalise:
            return wavefront.normalise()

        # Aberrations
        aberrations = self.eval_basis(coords)
        if self.as_phase:
            wavefront = wavefront.add_phase(aberrations)
        else:
            wavefront += aberrations
        return wavefront


#######################
# Composite Apertures #
#######################
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
    centre: Array, metres
        The (x, y) coords of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperture.
    compression: Array
        The (x, y) compression of the aperture.
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    apertures: dict

    def __init__(
        self: ApertureLayer,
        apertures: list,
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
        rotation: Array = None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the CompositeAperture class.

        Parameters
        ----------
        apertures: dict(str, Aperture)
            The sub-apertures that make up the full aperture.
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.apertures = dlu.list2dictionary(apertures, False, ApertureLayer)
        super().__init__(
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
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

    def _transmissions(
        self: ApertureLayer, coords: Array, pixel_scale
    ) -> Array:
        eval_fn = lambda ap: ap._transmission(coords, pixel_scale)
        leaf_fn = lambda ap: isinstance(ap, ApertureLayer)
        transmissions = tree_map(eval_fn, self.apertures, is_leaf=leaf_fn)
        return np.squeeze(np.array(tree_flatten(transmissions)[0]))

    def __call__(self: ApertureLayer, wavefront: Wavefront) -> Wavefront:
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
            coords = self.transformation(wavefront.coords)
        else:
            coords = wavefront.coords

        # Transmission
        wavefront *= self._transmission(coords, wavefront.pixel_scale)
        if self.normalise:
            return wavefront.normalise()

        # Aberrations
        aberrations = self.eval_basis(coords)
        if self.as_phase:
            wavefront = wavefront.add_phase(aberrations)
        else:
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
    centre: Array, metres
        The (x, y) coords of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperture.
    compression: Array
        The (x, y) compression of the aperture.
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    def __init__(
        self: ApertureLayer,
        apertures: list,
        centre: Array = None,
        shear: Array = None,
        compression: Array = None,
        rotation: Array = None,
        normalise: bool = False,
    ) -> ApertureLayer:
        """
        Constructor for the CompoundAperture class.

        Parameters
        ----------
        apertures: list[Aperture]
            The sub-apertures that make up the full aperture.
        centre: Array, metres = np.array([0., 0.])
            The (x, y) coords of the centre of the aperture.
        shear: Array = np.array([0., 0.])
            The (x, y) linear shear of the aperture.
        compression: Array  = np.array([1., 1.])
            The (x, y) compression of the aperture.
        rotation: Array, radians = np.array(0.)
            The clockwise rotation of the aperture.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        super().__init__(
            apertures=apertures,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
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

    def _transmission(
        self: ApertureLayer, coords: Array, pixel_scale
    ) -> Array:
        if self.transformation is not None:
            coords = self.transformation(coords)
        return self._transmissions(coords, pixel_scale).prod(0)


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
    centre: Array, metres
        The (x, y) coords of the centre of the aperture.
    shear: Array
        The (x, y) linear shear of the aperture.
    compression: Array
        The (x, y) compression of the aperture.
    rotation: Array, radians
        The clockwise rotation of the aperture.
    normalise : bool = False
        Whether to normalise the wavefront after passing through the aperture.
    """

    def eval_basis(self, coords):
        return super().eval_basis(coords).sum(0)

    def _transmission(
        self: ApertureLayer, coords: Array, pixel_scale
    ) -> Array:
        if self.transformation is not None:
            coords = self.transformation(coords)
        return self._transmissions(coords, pixel_scale).sum(0)


###########
# Factory #
###########
def ApertureFactory(
    npixels: int,
    radial_orders: Array = None,
    coefficients: Array = None,
    noll_indices: Array = None,
    aperture_ratio: float = 1.0,
    secondary_ratio: float = 0.0,
    nsides: int = 0,
    secondary_nsides: int = 0,
    rotation: float = 0.0,
    nstruts: int = 0,
    strut_ratio: float = 0.0,
    strut_rotation: float = 0.0,
    normalise: bool = True,
    static: bool = True,
):
    """
    This method is used to
    give a simple constructor interface that is used to construct the most
    commonly used apertures. It is able to construct hard-edged circular or
    regular polygonal apertures. Secondary mirrors obscurations with the same
    aperture shape can be constructed, along with uniformly spaced struts. The
    ratio of the primary aperture opening to the array size is determined by
    the `aperture_ratio` parameter, with secondary mirror obscurations and
    struts being scaled relative to the aperture diameter.

    Let's look at an example of how to construct a simple circular aperture
    with a secondary mirror obscuration held by 4 struts. For this example lets
    take a 2m diameter aperture, with a 20cm secondary mirror held by 3 struts
    with a width of 2cm. In this example the secondary mirror is 10% of the
    primary aperture diameter and the struts are 1% of the primary aperture
    diameter, giving us values of 0.1 and 0.01 for the `secondary_ratio` and
    `strut_ratio` parameters. Let calculate this for a 512x512 array with the
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

    Parameters
    ----------
    npixels : int
        Number of pixels used to represent the aperture.
    radial_orders : Array = None
        The radial orders of the zernike polynomials to be used for the
        aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
        [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
        The order must be increasing but does not have to be consecutive.
        If you want to specify specific zernikes across radial orders the
        noll_indices argument should be used instead.
    coefficients : Array = None
        The zernike coefficients to be applied to the aberrations. Defaults
        to an array of zeros.
    noll_indices : Array = None
        The zernike noll indices to be used for the aberrations. [1, 2, 3]
        would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
        Tilt Y, Defocus.
    aperture_ratio : float = 1.
        The ratio of the aperture size to the array size. A value of 1.
        results in an aperture that fully spans the array, a value of 0.5
        results in an aperture that is half the size of the array, which is
        equivalent to a padding factor of 2.
    secondary_ratio : float = 0.
        The ratio of the secondary mirror obscuration diameter to the
        aperture diameter. A value of 0. results in no secondary mirror
        obscuration.
    nsides : int = 0
        Number of sides of the aperture. A zero input results in a circular
        aperture. All other values of three and above are supported.
    secondary_nsides : int = 0
        The number of sides of the secondary mirror obscuration. A zero input
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
    normalise : bool = True
        Whether to normalise the wavefront after passing through the
        aperture.
    static : bool = True
        Whether to return a static aperture or a dynamic aperture.

    Returns
    -------
    aperture : StaticAperture
        Returns an appropriately constructed StaticAperture.
    """
    # Check valid inputs
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

    # Primary
    if nsides == 0:
        ap = CircularAperture(aperture_ratio / 2, softening=0)
    else:
        ap = RegPolyAperture(
            nsides, aperture_ratio / 2, softening=0, rotation=rotation
        )

    # Aberrations
    if radial_orders is not None:
        radial_orders = np.array(radial_orders)

        if (radial_orders < 0).any():
            raise ValueError("Radial orders must be >= 0")

        noll_indices = []
        for order in radial_orders:
            start = dlu.triangular_number(order)
            stop = dlu.triangular_number(order + 1)
            noll_indices.append(np.arange(start, stop) + 1)
        noll_indices = np.concatenate(noll_indices).astype(int)

    if noll_indices is None:
        apertures.append(ap)
    else:
        ab_ap = AberratedAperture(ap, noll_indices, coefficients=coefficients)
        apertures.append(ab_ap)

    # Secondary
    if secondary_ratio != 0:
        secondary_rel = aperture_ratio * secondary_ratio

        # Circular
        if secondary_nsides == 0:
            apertures.append(
                CircularAperture(
                    secondary_rel / 2, softening=0, occulting=True
                )
            )
        # Polygonal
        else:
            apertures.append(
                RegPolyAperture(
                    secondary_nsides,
                    secondary_rel / 2,
                    softening=0,
                    rotation=rotation,
                    occulting=True,
                )
            )

    # Spiders
    if nstruts > 0:
        if strut_ratio == 0:
            raise ValueError("strut_ratio must be > 0 if nstruts > 0")
        strut_rel = aperture_ratio * strut_ratio
        full_rotation = strut_rotation + rotation
        apertures.append(
            Spider(nstruts, strut_rel, rotation=full_rotation, softening=0)
        )

    # Construct CompoundAperture
    full_aperture = CompoundAperture(apertures, normalise=normalise)
    if static:
        return full_aperture.make_static(npixels, 1)
    return full_aperture
