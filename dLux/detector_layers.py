from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from jax.scipy.stats import norm
from zodiax import Base
import dLux

__all__ = [
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplySaturation",
    "AddConstant",
    "IntegerDownsample",
    "RotateDetector",
]

Image = lambda: dLux.images.Image


class DetectorLayer(Base):
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
    def __call__(
        self: DetectorLayer, image: Image()
    ) -> Image:  # pragma: no cover
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """


class ApplyPixelResponse(DetectorLayer):
    """
    Applies a pixel response array to the input image, via a multiplication.

    Attributes
    ----------
    pixel_response : Array
        The pixel_response to apply to the input image.
    """

    pixel_response: Array

    def __init__(self: DetectorLayer, pixel_response: Array):
        """
        Constructor for the ApplyPixelResponse class.

        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input image. Must be a
            2-dimensional array equal to size of the image at time of
            application.
        """
        super().__init__()
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2 dimensional array.")

    def __call__(self: DetectorLayer, image: Image()) -> Image:
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        return image * self.pixel_response


class ApplyJitter(DetectorLayer):
    """
    Convolves the image with a Gaussian kernel parameterised by the standard
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

    def __call__(self: DetectorLayer, image: Image()) -> Image():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        kernel = self.generate_kernel(image.pixel_scale)
        return image.convolve(kernel)


class ApplySaturation(DetectorLayer):
    """
    Applies a simple saturation model to the input image, by clipping any
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

    def __call__(self: DetectorLayer, image: Image()) -> Image():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        return image.min("image", self.saturation)


class AddConstant(DetectorLayer):
    """
    Add a constant to the output image. This is typically used to model the
    mean value of the detector noise.

    Attributes
    ----------
    value : Array
        The value to add to the image.
    """

    value: Array

    def __init__(self: DetectorLayer, value: Array) -> DetectorLayer:
        """
        Constructor for the AddConstant class.

        Parameters
        ----------
        value : Array
            The value to add to the image.
        """
        super().__init__()
        self.value = np.asarray(value, dtype=float)
        if self.value.ndim != 0:
            raise ValueError("value must be a scalar array.")

    def __call__(self: DetectorLayer, image: Image()) -> Image():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        return image + self.value


class IntegerDownsample(DetectorLayer):
    """
    Downsamples an input image by an integer number of pixels via a sum.
    The number of pixels in the input image must be integer divisible by the
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

    def __call__(self, image):
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        return image.downsample(self.kernel_size)


class RotateDetector(DetectorLayer):
    """
    Applies a rotation to the image using interpolation methods.

    Parameters
    ----------
    angle : Array, radians
        The angle by which to rotate the image in the clockwise direction.
    order : int
        The order of the interpolation.
    """

    angle: Array
    order: int

    def __init__(self: DetectorLayer, angle: Array, order: int = 1):
        """
        Constructor for the RotateDetector class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the image in the clockwise direction.
        """
        super().__init__()
        self.angle = np.asarray(angle, dtype=float)
        self.order = int(order)
        if self.angle.ndim != 0:
            raise ValueError("angle must be a scalar array.")

    def __call__(self: DetectorLayer, image: Image()) -> Image():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        return image.rotate(self.angle, self.order)
