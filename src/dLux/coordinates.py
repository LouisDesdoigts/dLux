"""Coordinate specifications and transformations."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

__all__ = [
    "Spec",
    "PadSpec",
    "CoordSpec",
    "BaseCoordTransform",
    "CoordTransform",
    "DistortedCoords",
]


class Spec(zdx.Base):
    """
    Abstract base class for coordinate/sampling specifications.

    ??? abstract "UML"
        ![UML](../assets/uml/Spec.png)
    """

    pass


class PadSpec(Spec):
    """
    Coordinate specification defined via integer padding and cropping factors
    relative to an input grid size.

    ??? abstract "UML"
        ![UML](../assets/uml/PadSpec.png)

    Attributes
    ----------
    pad : int
        Factor by which to increase the grid size. The padded grid will have
        ``n * pad`` pixels along each axis.
    crop : int
        Factor by which to reduce the grid size after processing. The cropped
        grid will have ``n_out // crop`` pixels along each axis.
    c : float
        Centre coordinate of the grid, in metres.
    """

    pad: int
    crop: int
    c: float

    def __init__(self, pad=1, crop=1, c=0.0):
        """
        Parameters
        ----------
        pad : int = 1
            Grid size increase factor.
        crop : int = 1
            Grid size reduction factor applied after processing.
        c : float = 0.0
            Centre coordinate of the grid, in metres.
        """
        self.pad = int(pad)
        self.crop = int(crop)
        self.c = np.asarray(c, float)


class CoordSpec(Spec):
    """
    Coordinate specification defined explicitly by number of pixels, pixel
    scale, and centre offset.

    ??? abstract "UML"
        ![UML](../assets/uml/CoordSpec.png)

    Attributes
    ----------
    n : int
        Number of pixels along each axis.
    d : float
        Pixel scale (spacing between adjacent pixels), in metres.
    c : float
        Centre coordinate of the grid, in metres.
    xs : Array, property
        Derived pixel-centre coordinates along one axis, in metres.
    fov : float, property
        Derived total field of view of the grid, in metres.
    extent : tuple[float, float], property
        Derived coordinate range of the grid edges, in metres.
    """

    n: int
    d: float
    c: float

    def __init__(self, n=None, d=None, c=0.0):
        """
        Parameters
        ----------
        n : int = None
            Number of pixels along each axis.
        d : float = None
            Pixel scale in metres.
        c : float = 0.0
            Centre coordinate of the grid, in metres.
        """
        self.n = n
        self.d = None if d is None else np.asarray(d, float)
        self.c = None if c is None else np.asarray(c, float)

    @property
    def xs(self):
        """
        1D array of pixel centre coordinates along one axis.

        Returns
        -------
        xs : Array
            Coordinates of pixel centres, in metres, centred on `c`.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate coordinates.")
        return self.c + (np.arange(self.n) - (self.n - 1) / 2) * self.d

    @property
    def fov(self):
        """
        Total field of view of the grid.

        Returns
        -------
        fov : float
            Field of view in metres, equal to ``n * d``.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate FOV.")
        return self.n * self.d

    @property
    def extent(self):
        """
        Coordinate range (min, max) of the grid edges.

        Returns
        -------
        extent : tuple[float, float]
            ``(lower_edge, upper_edge)`` coordinates in metres.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate extent.")
        return self.c - (self.n / 2) * self.d, self.c + (self.n / 2) * self.d


class BaseCoordTransform(zdx.Base):
    """
    Abstract base class for coordinate transformations.

    Provides a common interface for applying transformations to coordinates,
    including a backwards-compatible `apply` method.

    ??? abstract "UML"
        ![UML](../assets/uml/BaseCoordTransform.png)
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically inherit __call__ docstrings and annotations from parent class.
        """
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])

    def calculate(self: BaseCoordTransform, npix: int, diameter: float) -> Array:
        """
        Generate and apply transformations to coordinates.

        Parameters
        ----------
        npix : int
            The number of pixels in the output array.
        diameter : float
            The diameter of the output array in metres.

        Returns
        -------
        coords : Array
            The transformed coordinates.
        """
        coords = dlu.pixel_coords(npix, diameter)
        return self(coords)

    @abstractmethod
    def __call__(self: BaseCoordTransform, coords: Array) -> Array:  # pragma: no cover
        """
        Apply the transformation to input coordinates.

        Parameters
        ----------
        coords : Array
            The input coordinates to be transformed.

        Returns
        -------
        coords : Array
            The transformed coordinates.
        """

    def apply(self: BaseCoordTransform, coords: Array) -> Array:
        """
        Backwards compatibility alias for `__call__`.

        Parameters
        ----------
        coords : Array
            The input coordinates to be transformed.

        Returns
        -------
        coords : Array
            The transformed coordinates.
        """
        return self(coords)


class CoordTransform(BaseCoordTransform):
    """
    A simple class to handle coordinate transformations applied to dynamic aperture
    classes. Transformations are applied in the order:
        1. Translation
        2. Shear
        3. Compression
        4. Rotation

    ??? abstract "UML"
        ![UML](../assets/uml/CoordTransform.png)

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
    rotation: Array
    compression: Array
    shear: Array

    def __init__(
        self: CoordTransform,
        translation: Array = None,
        rotation: float = None,
        compression: Array = None,
        shear: Array = None,
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
                raise ValueError("translation must have shape (2,).")
        else:
            self.translation = None

        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=float)
            if self.rotation.shape != ():
                raise ValueError("rotation must have shape ().")
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
                raise ValueError("shear must have shape (2,).")
        else:
            self.shear = None

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


class DistortedCoords(BaseCoordTransform):
    """
    A class to handle coordinates distorted by a 2D polynomial distortion.

    ??? abstract "UML"
        ![UML](../assets/uml/DistortedCoords.png)

    Attributes
    ----------
    powers : Array
        Powers of the polynomial distortion.
    distortion : Array
        Distortion coefficients.
    """

    powers: Array
    distortion: Array

    def __init__(
        self: DistortedCoords, order: int = 1, distortion: Array | None = None
    ):
        """
        Parameters
        ----------
        order : int
            Order of polynomial to use.
        distortion : Array | None
            Distortion coefficients, defaulting to 0.
        """
        self.powers = np.array(dlu.gen_powers(order + 1))[:, 1:]

        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = np.asarray(distortion, dtype=float)
        if distortion.shape != self.powers.shape:
            raise ValueError("distortion shape must match powers shape.")
        self.distortion = distortion

    def __call__(self, coords):
        return dlu.distort_coords(coords, self.distortion, self.powers)
