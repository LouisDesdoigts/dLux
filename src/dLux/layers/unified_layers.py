"""Layers that operate on both wavefronts and PSFs."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from ..parametric import BaseParametric
from ..psfs import PSF
from ..wavefronts import Wavefront

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "BaseUnifiedLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Normalise",
    "Resize",
    "Flip",
    "Lambda",
]


class BaseLayer(zdx.Base):
    """Base class for callable transformations of dLux objects."""

    @abstractmethod
    def __call__(self, target: Any) -> Any:  # pragma: no cover
        """Apply this layer to its target."""

    def apply(self, target: Any) -> Any:
        """Backwards-compatible alias for calling the layer."""
        return self(target)

    @staticmethod
    def resolve(value: Any, **kwargs: Any) -> Any:
        """Evaluate a parametric value, or return an ordinary value unchanged."""
        if isinstance(value, BaseParametric):
            return value.evaluate(**kwargs)
        return value

    @staticmethod
    def as_parametric(value: Any, dtype: Any = float) -> Any:
        """Preserve parametric values and convert ordinary values to arrays."""
        if value is None or isinstance(value, BaseParametric):
            return value
        return np.asarray(value, dtype=dtype)

    def __init_subclass__(cls, **kwargs):
        """Inherit callable documentation for concrete layer implementations."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def __call__(self, wavefront: Any) -> Any:  # pragma: no cover
        """Transform a wavefront."""


class BaseUnifiedLayer(BaseLayer):
    """Base class for operations shared by wavefronts and PSFs."""


class TransmissiveLayer(BaseOpticalLayer):
    """Apply a scalar transmission, with optional output normalisation."""

    transmission: Array | BaseParametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = self.as_parametric(transmission)
        self.normalise = bool(normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        transmission = self.resolve(self.transmission, wavefront=wavefront)
        if transmission is not None:
            transmission = wavefront._to_phasor_shape(transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(BaseOpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront."""

    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None

    def __init__(self, opd=None, phase=None):
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        opd = self.resolve(self.opd, wavefront=wavefront)
        phase = self.resolve(self.phase, wavefront=wavefront)
        wavefront = wavefront.add_opd(opd)
        return wavefront.add_phase(phase)


class Normalise(BaseOpticalLayer):
    """Normalise a wavefront to unit power."""

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        return wavefront.normalise()


class Resize(BaseUnifiedLayer):
    """Resize a wavefront or PSF by padding or cropping."""

    npixels: int

    def __init__(self, npixels: int):
        self.npixels = int(npixels)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target.resize(self.npixels)


class Flip(BaseUnifiedLayer):
    """Flip a wavefront or PSF about one or more array axes."""

    axes: tuple[int, ...] | int

    def __init__(self, axes: tuple[int, ...] | int):
        self.axes = axes
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        if not all(isinstance(axis, int) for axis in axes):
            raise ValueError("axes must be an int or tuple of ints.")

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target.flip(self.axes)


class Lambda(BaseUnifiedLayer):
    """Return a wavefront or PSF unchanged."""

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target
