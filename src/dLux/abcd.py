"""Optical elements represented by ABCD matrices."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from abcdLux import abcd as _abcd

__all__ = [
    "BaseABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDConjugatePlane",
]


class BaseABCDElement(zdx.Base):
    """Base class for elements represented by an ABCD matrix."""


class ABCDFreeSpace(BaseABCDElement):
    """A free-space propagation element represented by an ABCD matrix."""

    distance: float

    def __init__(self, distance):
        self.distance = np.asarray(distance, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for free-space propagation."""
        return _abcd.abcd_free_space(self.distance)


class ABCDLens(BaseABCDElement):
    """A thin lens represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.asarray(focal_length, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the lens."""
        return _abcd.abcd_lens(self.focal_length)


class ABCDMirror(BaseABCDElement):
    """A curved mirror represented by an ABCD matrix."""

    radius: float

    def __init__(self, radius):
        self.radius = np.asarray(radius, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the mirror."""
        return _abcd.abcd_mirror(self.radius)


class ABCDConjugatePlane(BaseABCDElement):
    """A conjugate-plane transform represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.asarray(focal_length, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for conjugate-plane propagation."""
        return _abcd.abcd_fraunhofer(self.focal_length)
