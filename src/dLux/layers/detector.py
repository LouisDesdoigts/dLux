"""Detector-layer abstractions and detector-layer implementations."""

from __future__ import annotations
from abc import abstractmethod

import jax.numpy as np
from jax import Array
import dLux.utils as dlu

from ..fields import PSF
from .optical import BaseLayer

__all__ = [
    "BaseDetectorLayer",
    "DetectorLayer",
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplySaturation",
    "AddConstant",
]


class BaseDetectorLayer(BaseLayer):
    """Base class for layers that transform PSFs."""

    @abstractmethod
    def __call__(self, psf: PSF) -> PSF:
        """Transform a PSF."""


class DetectorLayer(BaseDetectorLayer):
    """Public contract for layers that transform PSFs."""


class ApplyPixelResponse(DetectorLayer):
    """Multiply a PSF by a two-dimensional pixel-response map.

    Parameters
    ----------
    pixel_response : Array
        Response map applied to the final two PSF axes. Its shape must match the
        detector sampling when the layer is evaluated.
    """

    pixel_response: Array

    def __init__(self: ApplyPixelResponse, pixel_response: Array):
        super().__init__()
        self.pixel_response = dlu.to_value(pixel_response)

        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2d array.")

    def __call__(self: ApplyPixelResponse, psf: PSF) -> PSF:
        """Apply the response map to a PSF."""
        return psf * self.pixel_response


class ApplyJitter(DetectorLayer):
    """Convolve a PSF with a radially symmetric Gaussian jitter kernel.

    Parameters
    ----------
    sigma : float, pixels
        Strictly positive standard deviation in detector pixels.
    kernel_size : int
        Positive width of the sampled convolution kernel.
    oversample : int
        Positive sampling factor used before integrating the kernel to detector
        pixels.
    """

    sigma: float
    kernel_size: int
    oversample: int

    def __init__(
        self: ApplyJitter, sigma: float, kernel_size: int = 9, oversample: int = 3
    ):
        super().__init__()
        self.kernel_size = dlu.as_size(kernel_size, 1, "kernel_size")[0]
        self.oversample = dlu.as_size(oversample, 1, "oversample")[0]
        self.sigma = dlu.to_value(sigma, name="sigma")

        if self.sigma.ndim != 0:
            raise ValueError("sigma must be scalar.")
        if self.sigma <= 0:
            raise ValueError("sigma must be greater than zero.")

    @property
    def kernel(self: ApplyJitter) -> Array:
        """Return the normalised, pixel-integrated Gaussian kernel."""
        kernel = dlu.gaussian(
            mean=np.array([0.0, 0.0]),
            std=np.array([self.sigma, self.sigma]),
            npixels=self.kernel_size * self.oversample,
        )
        return dlu.downsample(kernel, self.oversample, mean=False)

    def __call__(self: ApplyJitter, psf: PSF) -> PSF:
        """Convolve a PSF with the jitter kernel."""
        return psf.convolve(self.kernel)


class ApplySaturation(DetectorLayer):
    """Clip PSF values at a fixed saturation threshold.

    Parameters
    ----------
    threshold : float
        Maximum retained detector value.
    """

    threshold: float

    def __init__(self: ApplySaturation, threshold: float):
        super().__init__()
        self.threshold = dlu.to_value(threshold)

    def __call__(self: ApplySaturation, psf: PSF) -> PSF:
        """Apply the saturation threshold to a PSF."""
        return psf.min("data", self.threshold)


class AddConstant(DetectorLayer):
    """Add a spatially constant detector signal to a PSF.

    Parameters
    ----------
    value : float
        Constant value added to every PSF sample.
    """

    value: float

    def __init__(self: AddConstant, value: float):
        super().__init__()
        self.value = dlu.to_value(value)

    def __call__(self: AddConstant, psf: PSF) -> PSF:
        """Add the configured constant to a PSF."""
        return psf + self.value
