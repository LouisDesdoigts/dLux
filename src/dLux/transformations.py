"""Coordinate transformation utilities for dynamic apertures and distortions."""

from __future__ import annotations
from abc import abstractmethod
import zodiax as zdx
from jax import Array
import jax.numpy as np
import dLux.utils as dlu

__all__ = ["BaseCoordTransform", "CoordTransform", "DistortedCoords"]


class BaseCoordTransform(zdx.Base):
    """
    Abstract base class for coordinate transformations.

    Provides a common interface for applying transformations to coordinates,
    including a backwards-compatible `apply` method.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically inherit __call__ docstrings and annotations from parent class.
        """
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls)

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
    def __call__(self: BaseCoordTransform, coords: Array) -> Array:
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
        raise NotImplementedError("Subclasses must implement __call__")

    def apply(self: BaseCoordTransform, coords: Array) -> Array:
        """
        Backwards compatibility method that invokes __call__.

        Delegates to the __call__ method.
        """
        return self(coords)


# Class to be held by dynamic apertures
class CoordTransform(BaseCoordTransform):
    """
    A simple class to handle coordinate transformations applied to dynamic aperture
    classes. Transformations are applied in the order:
        1. Translation
        2. Shear
        3. Compression
        4. Rotation

    ??? abstract "UML"
        ![UML](../../assets/uml/CoordTransform.png)

    Attributes
    ----------
    translation: Array
        The (x, y) shift applied to the coords.
    rotation: float
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
