from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from jax.scipy.stats import norm
import dLux

__all__ = [
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplySaturation",
    "AddConstant",
    "IntegerDownsample",
]

PSF = lambda: dLux.psfs.PSF


class DetectorLayer(dLux.base.Base):
    """
    A base Detector layer class to help with type checking throughout the rest
    of the software.
    """

    def __init__(self: DetectorLayer):
        """
        Constructor for the DetectorLayer class.
        """
        super().__init__()

    @abstractmethod
    def __call__(self: DetectorLayer, psf: PSF()) -> PSF:  # pragma: no cover
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
    Applies a pixel response array to the input psf, via a multiplication.

    Attributes
    ----------
    pixel_response : Array
        The pixel_response to apply to the input psf.
    """

    pixel_response: Array

    def __init__(self: DetectorLayer, pixel_response: Array):
        """
        Constructor for the ApplyPixelResponse class.

        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input psf. Must be a
            2-dimensional array equal to size of the psf at time of
            application.
        """
        super().__init__()
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2 dimensional array.")

    def __call__(self: DetectorLayer, psf: PSF()) -> PSF:
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
    Convolves the psf with a Gaussian kernel parameterised by the standard
    deviation (sigma).

    Attributes
    ----------
    sigma : Array, pixels
        The standard deviation of the Gaussian kernel, in units of pixels.
    kernel_size : int
        The size of the convolution kernel to use.
    """

    kernel_size: int
    sigma: Array

    def __init__(self: DetectorLayer, sigma: Array, kernel_size: int = 10):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        sigma : Array, pixels
            The standard deviation of the Gaussian kernel, in units of pixels.
        kernel_size : int = 10
            The size of the convolution kernel to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.ndim != 0:
            raise ValueError("sigma must be a scalar array.")

    def generate_kernel(self: DetectorLayer, pixel_scale: Array) -> Array:
        """
        Generates the normalised Gaussian kernel.

        Returns
        -------
        kernel : Array
            The Gaussian kernel.
        """
        # Generate distribution
        sigma = self.sigma * pixel_scale
        x = np.linspace(-10, 10, self.kernel_size) * pixel_scale
        kernel = norm.pdf(x, scale=sigma) * norm.pdf(x[:, None], scale=sigma)
        return kernel / np.sum(kernel)

    def __call__(self: DetectorLayer, psf: PSF()) -> PSF():
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
    Applies a simple saturation model to the input psf, by clipping any
    values above saturation, to saturation.

    Attributes
    ----------
    saturation : Array
        The value at which the saturation is applied.
    """

    saturation: Array

    def __init__(self: DetectorLayer, saturation: Array) -> DetectorLayer:
        """
        Constructor for the ApplySaturation class.

        Parameters
        ----------
        saturation : Array
            The value at which the saturation is applied.
        """
        super().__init__()
        self.saturation = np.asarray(saturation, dtype=float)
        if self.saturation.ndim != 0:
            raise ValueError("saturation must be a scalar array.")

    def __call__(self: DetectorLayer, psf: PSF()) -> PSF():
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
        return psf.min("data", self.saturation)


class AddConstant(DetectorLayer):
    """
    Add a constant to the output psf. This is typically used to model the
    mean value of the detector noise.

    Attributes
    ----------
    value : Array
        The value to add to the psf.
    """

    value: Array

    def __init__(self: DetectorLayer, value: Array) -> DetectorLayer:
        """
        Constructor for the AddConstant class.

        Parameters
        ----------
        value : Array
            The value to add to the psf.
        """
        super().__init__()
        self.value = np.asarray(value, dtype=float)
        if self.value.ndim != 0:
            raise ValueError("value must be a scalar array.")

    def __call__(self: DetectorLayer, psf: PSF()) -> PSF():
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


class IntegerDownsample(DetectorLayer):
    """
    Downsamples an input psf by an integer number of pixels via a sum.
    The number of pixels in the input psf must be integer divisible by the
    kernel_size.

    Attributes
    ----------
    kernel_size : int
        The size of the downsampling kernel.
    """

    kernel_size: int

    def __init__(self: DetectorLayer, kernel_size: int) -> DetectorLayer:
        """
        Constructor for the IntegerDownsample class.

        Parameters
        ----------
        kernel_size : int
            The size of the downsampling kernel.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)

    def __call__(self, psf):
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
