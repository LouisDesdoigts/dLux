from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from jax.scipy.stats import norm


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
    A base Detector layer class to help with type checking throughout the rest of the
    software.
    """

    def __init__(self: DetectorLayer):
        super().__init__()

    @abstractmethod
    def apply(self: DetectorLayer, psf: PSF) -> PSF:  # pragma: no cover
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """


class ApplyPixelResponse(DetectorLayer):
    """
    Applies a pixel response array to the input psf, via a multiplication. This can be
    used to model variations in the inter and intra-pixel sensitivity variations common
    to most detectors.

    ??? abstract "UML"
        ![UML](../../assets/uml/ApplyPixelResponse.png)

    Attributes
    ----------
    pixel_response : Array
        The pixel_response to apply to the input psf.
    """

    pixel_response: Array

    def __init__(self: DetectorLayer, pixel_response: Array):
        """
        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input psf. Must be a 2-dimensional array
            equal to size of the psf at time of application.
        """
        super().__init__()
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2 dimensional array.")

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """
        return psf * self.pixel_response


class ApplyJitter(DetectorLayer):
    """
    Convolves the psf with a radially symmetric Gaussian kernel parameterised by its
    standard deviation (sigma).

    ??? abstract "UML"
        ![UML](../../assets/uml/ApplyJitter.png)

    Attributes
    ----------
    sigma : float, pixels
        The standard deviation of the Gaussian kernel, in units of pixels.
    kernel_size : int
        The size of the convolution kernel to use.
    """

    kernel_size: int
    sigma: float

    def __init__(self: DetectorLayer, sigma: float, kernel_size: int = 10):
        """
        Parameters
        ----------
        sigma : float, pixels
            The standard deviation of the Gaussian kernel, in units of pixels.
        kernel_size : int = 10
            The size of the convolution kernel to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def generate_kernel(self: DetectorLayer, pixel_scale: float) -> Array:
        """
        Generates the normalised Gaussian kernel.

        Returns
        -------
        kernel : Array
            The Gaussian kernel.
        """
        # TODO: Move to utils?
        # Generate distribution
        sigma = self.sigma * pixel_scale
        x = np.linspace(-10, 10, self.kernel_size) * pixel_scale
        kernel = norm.pdf(x, scale=sigma) * norm.pdf(x[:, None], scale=sigma)
        return kernel / np.sum(kernel)

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """
        kernel = self.generate_kernel(psf.pixel_scale)
        return psf.convolve(kernel)


class ApplySaturation(DetectorLayer):
    """
    Applies a simple saturation model to the input psf, by clipping any values above
    the threshold value.

    ??? abstract "UML"
        ![UML](../../assets/uml/ApplySaturation.png)

    Attributes
    ----------
    threshold : float
        The threshold at which the saturation is applied.
    """

    threshold: float

    def __init__(self: DetectorLayer, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            The threshold at which the saturation is applied.
        """
        super().__init__()
        self.threshold = float(threshold)

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """
        return psf.min("data", self.threshold)


class AddConstant(DetectorLayer):
    """
    Adds a constant to the output psf. This is typically used to model the mean value of
    the detector noise.

    ??? abstract "UML"
        ![UML](../../assets/uml/AddConstant.png)

    Attributes
    ----------
    value : float
        The value to add to the psf.
    """

    value: float

    def __init__(self: DetectorLayer, value: float):
        """
        Parameters
        ----------
        value : float
            The value to add to the psf.
        """
        super().__init__()
        self.value = np.asarray(value, dtype=float)
        if self.value.ndim != 0:
            raise ValueError("value must be a scalar array.")

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """
        return psf + self.value


class Downsample(DetectorLayer):
    """
    Downsamples an input psf by an integer number of pixels via a sum. Typically used
    to downsample an oversampled psf to the true pixel size. Note kernel_size must be
    an integer multiple of the input psf size.

    ??? abstract "UML"
        ![UML](../../assets/uml/Downsample.png)

    Attributes
    ----------
    kernel_size : int
        The size of the downsampling kernel.
    """

    kernel_size: int

    def __init__(self: DetectorLayer, kernel_size: int):
        """
        Parameters
        ----------
        kernel_size : int
            The size of the downsampling kernel.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the PSF.

        Parameters
        ----------
        psf : PSF
            The psf to operate on.

        Returns
        -------
        psf : PSF
            The transformed psf.
        """
        return psf.downsample(self.kernel_size)
