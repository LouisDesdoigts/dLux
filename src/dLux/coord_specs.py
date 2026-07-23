"""Coordinate and sampling specifications."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx

__all__ = ["BaseSpec", "PadSpec", "CoordSpec"]


class BaseSpec(zdx.Base):
    """Base class for coordinate and sampling specifications."""


class PadSpec(BaseSpec):
    """Sampling specification defined by padding and cropping factors."""

    pad: int
    crop: int
    c: float

    def __init__(self, pad=1, crop=1, c=0.0):
        self.pad = int(pad)
        self.crop = int(crop)
        self.c = np.asarray(c, float)


class CoordSpec(BaseSpec):
    """Explicit coordinate specification for a square Cartesian grid."""

    n: int
    d: float
    c: float

    def __init__(self, n=None, d=None, c=0.0):
        self.n = n
        self.d = None if d is None else np.asarray(d, float)
        self.c = None if c is None else np.asarray(c, float)

    @property
    def xs(self):
        """Return one-dimensional pixel-centre coordinates."""
        if self.d is None:
            raise ValueError("d must be specified to calculate coordinates.")
        pixels = np.arange(self.n) - (self.n - 1) / 2
        fn = lambda d, c: c + pixels * d
        return np.vectorize(fn, signature="(),()->(n)")(self.d, self.c)

    @property
    def fov(self):
        """Return the total field of view."""
        if self.d is None:
            raise ValueError("d must be specified to calculate FOV.")
        return self.n * self.d

    @property
    def extent(self):
        """Return the lower and upper grid-edge coordinates."""
        if self.d is None:
            raise ValueError("d must be specified to calculate extent.")
        return self.c - (self.n / 2) * self.d, self.c + (self.n / 2) * self.d
