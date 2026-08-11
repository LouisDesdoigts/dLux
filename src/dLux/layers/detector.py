"""Composable deterministic detector transformations."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..fields import Intensity
from ..parametric import Parametric
from .optical import BaseLayer

__all__ = [
    "BaseDetectorLayer",
    "DetectorLayer",
    "Sensitivity",
    "Convolve",
    "Jitter",
    "Bias",
    "Gain",
    "Saturation",
]


class BaseDetectorLayer(BaseLayer):
    """Base class for transformations from one deterministic intensity to another."""

    @abstractmethod
    def apply(self, intensity: Intensity) -> Intensity:
        """Transform a sampled intensity."""


class DetectorLayer(BaseDetectorLayer):
    """Public contract for deterministic detector transformations.

    Implementations receive an ``Intensity`` and must return an ``Intensity`` while
    preserving unrelated leading axes and consistent grid metadata. Resolve
    Parametric leaves against :meth:`context` before applying the transformation.
    """

    @staticmethod
    def context(intensity: Intensity) -> dict[str, Any]:
        """Return the context used to resolve parametric attributes."""
        return {
            "intensity": intensity,
            "data": intensity.data,
            "variables": intensity.data[None],
            "coordinates": intensity.coordinates,
        }


class Sensitivity(DetectorLayer):
    """Multiply an intensity by a scalar, spatial, or parametric response.

    The resolved response follows ordinary JAX broadcasting against
    ``intensity.data``.

    Parameters
    ----------
    response : Array or Parametric
        Multiplicative detector response. Parametric values are evaluated against
        the input intensity.
    """

    response: Array | Parametric

    def __init__(self, response):
        """Initialise the multiplicative detector response."""
        self.response = dlu.to_value(response, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply the resolved response."""
        self = self.resolve(**self.context(intensity))
        return intensity * self.response


class Convolve(DetectorLayer):
    """Convolve an intensity with a two-dimensional kernel.

    ``kernel`` may be a fixed array or Parametric resolving to shape ``(ny, nx)``.
    It is applied independently over the final two intensity axes and is not
    normalised automatically. Leading intensity axes are preserved.

    Parameters
    ----------
    kernel : Array or Parametric
        Two-dimensional convolution kernel in array ``(y, x)`` order.
    """

    kernel: Array | Parametric

    def __init__(self, kernel):
        """Initialise the fixed or parametric convolution kernel."""
        self.kernel = dlu.to_value(kernel, types=Parametric)

        if not isinstance(self.kernel, Parametric) and self.kernel.ndim != 2:
            raise ValueError("kernel must be a 2d array.")

    def apply(self, intensity: Intensity) -> Intensity:
        """Convolve the intensity with the resolved kernel."""
        self = self.resolve(**self.context(intensity))
        if self.kernel.ndim != 2:
            raise ValueError("Resolved kernel must be a 2d array.")
        return intensity.convolve(self.kernel)


class Jitter(DetectorLayer):
    """Convolve an intensity with a normalised Gaussian jitter kernel.

    ``sigma`` is expressed in detector pixels. It may be scalar for circular jitter,
    length two for independent ``(x, y)`` widths, or a ``(2, 2)`` covariance matrix.
    A zero width produces a delta function along that axis and an entirely zero input
    produces the identity kernel.

    ``kernel_size`` and ``oversample`` are scalar or physical ``(x, y)`` integer
    pairs. The returned kernel uses array ``(y, x)`` order, is integrated from the
    sub-pixel grid, and is normalised over its finite support. Covariance matrices
    must be symmetric positive definite; this is not checked during compiled
    evaluation.

    Parameters
    ----------
    sigma : Array or Parametric, pixels
        Scalar circular width, ``(x, y)`` widths, or an ``(x, y)`` covariance matrix.
    kernel_size : int or tuple[int, int]
        Odd detector-pixel kernel dimensions in physical ``(x, y)`` order.
    oversample : int or tuple[int, int]
        Sub-pixel integration factors in physical ``(x, y)`` order.
    """

    sigma: Array | Parametric
    kernel_size: tuple[int, ...]
    oversample: tuple[int, ...]

    def __init__(self, sigma, kernel_size=9, oversample=3):
        """Initialise the jitter distribution and sampling."""
        self.sigma = dlu.to_value(sigma, types=Parametric)
        self.kernel_size = dlu.as_size(kernel_size, 2, "kernel_size")
        self.oversample = dlu.as_size(oversample, 2, "oversample")

        if any(size % 2 == 0 for size in self.kernel_size):
            raise ValueError("kernel_size must contain odd values.")
        if not isinstance(self.sigma, Parametric):
            valid = self.sigma.ndim == 0 or self.sigma.shape in ((2,), (2, 2))
            if not valid:
                raise ValueError("sigma must be scalar or have shape (2,) or (2, 2).")

    def kernel(self, intensity=None) -> Array:
        """Return the resolved, normalised detector-pixel jitter kernel."""
        if isinstance(self.sigma, Parametric) and intensity is None:
            raise ValueError("intensity is required when sigma is parametric.")

        if isinstance(self.sigma, Parametric):
            self = self.resolve(**self.context(intensity))
        sigma = self.sigma

        # Resolve the oversampled kernel dimensions and physical extent
        shape = tuple(n * o for n, o in zip(self.kernel_size, self.oversample))
        spacing = tuple(1 / o for o in self.oversample)
        extent = (np.asarray(shape) - 1) * np.asarray(spacing) / 2

        # Evaluate the independent or correlated Gaussian distribution
        if sigma.ndim < 2:
            widths = np.repeat(sigma, 2) if sigma.ndim == 0 else sigma
            density = dlu.gaussian(0, widths[::-1], shape[::-1], extent[::-1])
        else:
            covariance = sigma[::-1, ::-1]
            scales = np.sqrt(np.diag(covariance))
            limits = extent[::-1] / scales
            density = dlu.mv_gaussian(np.zeros(2), covariance, shape[::-1], limits)

        # Integrate the normalised distribution to detector pixels
        return dlu.downsample(density, self.oversample, mean=False)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply the resolved jitter kernel."""
        return intensity.convolve(self.kernel(intensity))


class Bias(DetectorLayer):
    """Add a scalar, spatial, or parametric bias to an intensity.

    Parameters
    ----------
    bias : Array or Parametric
        Additive detector signal broadcast against the input intensity.
    """

    bias: Array | Parametric

    def __init__(self, bias):
        """Initialise the additive detector signal."""
        self.bias = dlu.to_value(bias, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Add the resolved bias."""
        self = self.resolve(**self.context(intensity))
        return intensity + self.bias


class Gain(DetectorLayer):
    """Multiply an intensity by a linear or parametrically generated gain.

    A Parametric receives the intensity through ``variables`` with shape
    ``(1, ..., ny, nx)`` before being multiplied into the data. This enables nonlinear
    responses such as polynomial gain curves while retaining the same broadcasting
    contract as scalar and spatial gains.

    Parameters
    ----------
    gain : Array or Parametric
        Multiplicative gain or context-dependent nonlinear response.
    """

    gain: Array | Parametric

    def __init__(self, gain):
        """Initialise the linear or parametric gain."""
        self.gain = dlu.to_value(gain, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply the resolved linear or nonlinear gain."""
        self = self.resolve(**self.context(intensity))
        return intensity * self.gain


class Saturation(DetectorLayer):
    """Limit intensity values to a scalar, spatial, or parametric maximum.

    Parameters
    ----------
    limit : Array or Parametric
        Maximum retained intensity, broadcast over the input data.
    """

    limit: Array | Parametric

    def __init__(self, limit):
        """Initialise the upper intensity limit."""
        self.limit = dlu.to_value(limit, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply the resolved upper limit."""
        self = self.resolve(**self.context(intensity))
        return intensity.set(data=np.minimum(intensity.data, self.limit))
