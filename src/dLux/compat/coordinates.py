"""Released sampling and coordinate constructors using current specifications."""

import jax.numpy as np
import dLux.utils as dlu
from .warnings import warn_deprecated

Spec = dlu.Spec
BaseCoordTransform = dlu.BaseCoordTransform
__all__ = [
    "Spec",
    "PadSpec",
    "CoordSpec",
    "BaseCoordTransform",
    "CoordTransform",
    "DistortedCoords",
]


class _CoordTransform(dlu.AffineMap):
    """Add the released sampled-grid entry point to an affine map."""

    def calculate(self, npix, diameter):
        """Generate a centred square grid and transform its coordinates."""
        return self(dlu.Grid(npix, diam=diameter).coordinates)


class _DistortedCoords(dlu.Distortion):
    """Add the released sampled-grid entry point to a polynomial distortion."""

    def apply(self, coordinates):
        """Retain the squeezed single-pixel coordinate form of pixel_coords."""
        coordinates = np.asarray(coordinates)
        if coordinates.shape == (2,):
            return self(coordinates[:, None, None])[:, 0, 0]
        return self(coordinates)

    def calculate(self, npix, diameter):
        """Generate a centred square grid and transform its coordinates."""
        return self(dlu.Grid(npix, diam=diameter).coordinates)


def PadSpec(pad=1, crop=1, c=0.0):
    warn_deprecated("PadSpec", "FFTSpec", "FFTSpec(pad=..., crop=..., c=...)")
    return dlu.FFTSpec(pad=pad, crop=crop, c=c)


def CoordSpec(n=None, d=None, c=0.0):
    warn_deprecated("CoordSpec", "Grid", "Grid(n=..., d=..., c=..., unit='m')")
    return dlu.Grid(n=n, d=d, c=c, unit="m")


def CoordTransform(translation=None, rotation=None, compression=None, shear=None):
    warn_deprecated(
        "CoordTransform",
        "AffineMap",
        "AffineMap(matrix, offset); old transformation parameters become matrix "
        "entries",
    )
    # Released shear transposes sampled grid axes, so it is not a pointwise
    # affine map on arbitrary coordinates. Do not silently reinterpret it.
    if shear is not None:
        raise ValueError(
            "Released shear transposes grid samples. Use "
            "compat.utils.shear_coords explicitly, or dlu.Affine(shear=...) "
            "for a pointwise shear."
        )
    from ..utils.compat.coordinates import (
        translate_coords,
        compress_coords,
        rotate_coords,
    )

    points = np.array([[[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])
    for value, function in (
        (translation, translate_coords),
        (compression, compress_coords),
        (rotation, rotate_coords),
    ):
        if value is not None:
            points = function(points, np.asarray(value))
    offset = points[:, 0, 0]
    matrix = points[:, 0, 1:] - offset[:, None]
    return _CoordTransform(matrix, offset)


def DistortedCoords(order=1, distortion=None):
    warn_deprecated(
        "DistortedCoords", "Distortion", "Distortion(order=order, distortion=...)"
    )
    from ..utils.compat.coordinates import gen_powers

    return _DistortedCoords(
        powers=np.asarray(gen_powers(order + 1), dtype=int)[:, 1:],
        distortion=distortion,
    )
