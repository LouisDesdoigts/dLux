from __future__ import annotations
from zodiax import Base
from jax import Array
import jax.numpy as np
import dLux.utils as dlu


__all__ = ["CoordTransform"]


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
        return self.apply(dlu.pixel_coords(npix, diam))

    def apply(self, coords):
        if self.translation is not None:
            coords = dlu.translate_coords(coords, self.translation)
        if self.shear is not None:
            coords = dlu.shear_coords(coords, self.shear)
        if self.compression is not None:
            coords = dlu.compress_coords(coords, self.compression)
        if self.rotation is not None:
            coords = dlu.rotate_coords(coords, self.rotation)
        return coords
