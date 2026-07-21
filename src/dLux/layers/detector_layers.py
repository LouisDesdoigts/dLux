"""Detector-layer abstractions and detector-layer implementations."""

from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import Array
import dLux.utils as dlu

from ..psfs import PSF
from .optical_layers import BaseLayer

__all__ = [
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplySaturation",
    "AddConstant",
    "Downsample",
]


class DetectorLayer(BaseLayer):
    """
    A base detector layer class to help with type checking throughout the rest of the
    software.

    ??? abstract "UML"
        ![UML](../assets/uml/DetectorLayer.png)
    """

    def __init__(self: DetectorLayer):
        super().__init__()

    @abstractmethod
    def __call__(self: DetectorLayer, psf: PSF) -> PSF:  # pragma: no cover
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The PSF to operate on.

        Returns
        -------
        psf : PSF
            The transformed PSF.
        """

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Backwards compatibility alias for `__call__`.

        Parameters
        ----------
        psf : PSF
            The PSF to operate on.

        Returns
        -------
        psf : PSF
            The transformed PSF.
        """
        return self(psf)


class ApplyPixelResponse(DetectorLayer):
    """
    Applies a pixel response array to the input PSF via multiplication. This can be
    used to model inter- and intra-pixel sensitivity variations common
    to most detectors.

    ??? abstract "UML"
        ![UML](../assets/uml/ApplyPixelResponse.png)

    Attributes
    ----------
    pixel_response : Array
        The pixel_response to apply to the input PSF.
    """

    pixel_response: Array

    def __init__(self: ApplyPixelResponse, pixel_response: Array):
        """
        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input PSF. Must be a 2d array that
            matches the PSF shape at time of application.
        """
        super().__init__()
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2d array.")

    def __call__(self: ApplyPixelResponse, psf: PSF) -> PSF:
        return psf * self.pixel_response


class ApplyJitter(DetectorLayer):
    """
    Convolves the PSF with a radially symmetric Gaussian kernel parameterised by its
    standard deviation (sigma).

    ??? abstract "UML"
        ![UML](../assets/uml/ApplyJitter.png)

    Attributes
    ----------
    sigma : float, pixels
        The standard deviation of the Gaussian kernel, in units of pixels.
    kernel_size : int
        The size of the convolution kernel to use.
    oversample : int
        The oversampling factor to use when generating the kernel. This is used to
        mitigate aliasing when the kernel is small compared to the pixel size.
    """

    sigma: float
    kernel_size: int
    oversample: int

    def __init__(
        self: ApplyJitter, sigma: float, kernel_size: int = 9, oversample: int = 3
    ):
        """
        Parameters
        ----------
        sigma : float, pixels
            The standard deviation of the Gaussian kernel, in units of pixels.
        kernel_size : int = 9
            The size of the convolution kernel to use.
        oversample : int = 3
            The oversampling factor to use when generating the kernel. This is used to
            mitigate aliasing when the kernel is small compared to the pixel size.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.oversample = int(oversample)

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be greater than 0.")

    @property
    def kernel(self: ApplyJitter) -> Array:
        """
        Generates the normalised Gaussian kernel.

        Returns
        -------
        kernel : Array
            The Gaussian kernel.
        """
        kernel = dlu.gaussian(
            mean=np.array([0.0, 0.0]),
            std=np.array([self.sigma, self.sigma]),
            npixels=self.kernel_size * self.oversample,
        )
        return dlu.downsample(kernel, self.oversample, mean=False)

    def __call__(self: ApplyJitter, psf: PSF) -> PSF:
        return psf.convolve(self.kernel)


class ApplySaturation(DetectorLayer):
    """
    Applies a simple saturation model to the input PSF by clipping any values above
    the threshold value.

    ??? abstract "UML"
        ![UML](../assets/uml/ApplySaturation.png)

    Attributes
    ----------
    threshold : float
        The threshold at which the saturation is applied.
    """

    threshold: float

    def __init__(self: ApplySaturation, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            The threshold at which the saturation is applied.
        """
        super().__init__()
        self.threshold = float(threshold)

    def __call__(self: ApplySaturation, psf: PSF) -> PSF:
        return psf.min("data", self.threshold)


class AddConstant(DetectorLayer):
    """
    Adds a constant to the output PSF. This is typically used to model the mean value of
    the detector noise.

    ??? abstract "UML"
        ![UML](../assets/uml/AddConstant.png)

    Attributes
    ----------
    value : float
        The value to add to the PSF.
    """

    value: float

    def __init__(self: AddConstant, value: float):
        """
        Parameters
        ----------
        value : float
            The value to add to the PSF.
        """
        super().__init__()
        self.value = float(value)

    def __call__(self: AddConstant, psf: PSF) -> PSF:
        return psf + self.value


class Downsample(DetectorLayer):
    """
    Downsamples an input PSF by an integer number of pixels via a sum. Typically used
    to downsample an oversampled PSF to the true pixel size. Note the input PSF size
    must be divisible by kernel_size.

    ??? abstract "UML"
        ![UML](../assets/uml/Downsample.png)

    Attributes
    ----------
    kernel_size : int
        The size of the downsampling kernel.
    """

    kernel_size: int

    def __init__(self: Downsample, kernel_size: int):
        """
        Parameters
        ----------
        kernel_size : int
            The size of the downsampling kernel. Must be greater than 0.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be greater than 0.")

    def __call__(self: Downsample, psf: PSF) -> PSF:
        return psf.downsample(self.kernel_size)
