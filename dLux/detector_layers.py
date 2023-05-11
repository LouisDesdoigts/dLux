from __future__ import annotations
import jax.numpy as np
from abc import ABC, abstractmethod
from jax import Array
from jax.scipy.signal import convolve
from jax.scipy.stats import norm
from zodiax import Base
from dLux.utils.interpolation import rotate, fourier_rotate


__all__ = ["ApplyPixelResponse", "ApplyJitter", "ApplySaturation",
           "AddConstant", "IntegerDownsample", "Rotate"]


class DetectorLayer(Base):
    """
    A base Detector layer class to help with type checking throuhgout the rest
    of the software.
    """


    def __init__(self : DetectorLayer) -> DetectorLayer:
        """
        Constructor for the DetectorLayer class.
        """
        super().__init__()


    @abstractmethod
    def __call__(self : DetectorLayer, 
                 image : Array) -> Array: # pragma: no cover
        """
        Abstract method for Detector Layers
        """
        return


class ApplyPixelResponse(DetectorLayer):
    """
    Applies a pixel response array to the the input image, via a multiplication.

    Attributes
    ----------
    pixel_response : Array
        The pixel_response to apply to the input image.
    """
    pixel_response : Array


    def __init__(self           : DetectorLayer,
                 pixel_response : Array) -> DetectorLayer:
        """
        Constructor for the ApplyPixelResponse class.

        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input image. Must be a 2
            dimensional array equal to size of the image at time of application.
        """
        super().__init__()
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2 dimensional array.")


    def __call__(self : DetectorLayer, image) -> Array:
        """
        Applies the pixel response to the input image, via a multiplication.

        Parameters
        ----------
        image : Array
            The image to apply the pixel_response to.

        Returns
        -------
        image : Array
            The image with the pixel_response applied.
        """
        return image * self.pixel_response


class ApplyJitter(DetectorLayer):
    """
    Convolves the image with a gaussian kernel parameterised by the standard
    deviation (sigma).

    Attributes
    ----------
    sigma : Array, pixels
        The standard deviation of the guassian kernel, in units of pixels.
    kernel_size : int
        The size of the convolution kernel to use.
    """
    kernel_size : int
    sigma       : Array


    def __init__(self        : DetectorLayer,
                 sigma       : Array,
                 kernel_size : int = 10) -> DetectorLayer:
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        sigma : Array, pixels
            The standard deviation of the guassian kernel, in units of pixels.
        kernel_size : int = 10
            The size of the convolution kernel to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.ndim != 0:
            raise ValueError("sigma must be a scalar array.")


    def generate_kernel(self : DetectorLayer) -> Array:
        """
        Generates the normalised guassian kernel.

        Returns
        -------
        kernel : Array
            The gaussian kernel.
        """
        # Generate distribution
        x = np.linspace(-10, 10, self.kernel_size)
        kernel = norm.pdf(x,          scale=self.sigma) * \
                 norm.pdf(x[:, None], scale=self.sigma)
        return kernel/np.sum(kernel)


    def __call__(self : DetectorLayer, image : Array) -> Array:
        """
        Convolves the input image with the generate gaussian kernel.

        Parameters
        ----------
        image : Array
            The image to convolve with the gussian kernel.

        Returns
        -------
        image : Array
            The image with the gaussian kernel convolution applied.
        """
        kernel = self.generate_kernel()
        return convolve(image, kernel, mode='same')


class ApplySaturation(DetectorLayer):
    """
    Applies a simple saturation model to the input image, by clipping any
    values above saturation, to saturation.

    Attributes
    ----------
    saturation : Array
        The value at which the saturation is applied.
    """
    saturation : Array


    def __init__(self : DetectorLayer, saturation : Array) -> DetectorLayer:
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


    def __call__(self : DetectorLayer, image : Array) -> Array:
        """
        Applies the satuation effect by reducing all values in the image above
        saturation, to the saturation value.

        Parameters
        ----------
        image : Array
            The image to apply the saturation to.

        Returns
        -------
        image : Array
            The image with the saturation applied.
        """
        return np.minimum(image, self.saturation)


class AddConstant(DetectorLayer):
    """
    Add a constant to the output image. This is typically used to model the
    mean value of the detector noise.

    Attributes
    ----------
    value : Array
        The value to add to the image.
    """
    value : Array


    def __init__(self : DetectorLayer, value : Array) -> DetectorLayer:
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


    def __call__(self : DetectorLayer, image : Array) -> Array:
        """
        Adds the value to the input image.

        Parameters
        ----------
        image : Array
            The image to add the value to.

        Returns
        -------
        image : Array
            The image with the value added.
        """
        return image + self.value


class IntegerDownsample(DetectorLayer):
    """
    Downsamples an input image by an integer number of pixels via a sum.
    The number of pixels in the input image must by integer divisible by the
    kernel_size.

    Attributes
    ----------
    kernel_size : int
        The size of the downsampling kernel.
    """
    kernel_size : int


    def __init__(self : DetectorLayer, kernel_size : int) -> DetectorLayer:
        """
        Constructor for the IntegerDownsample class.

        Parameters
        ----------
        kernel_size : int
            The size of the downsampling kernel.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)


    def downsample(self        : DetectorLayer,
                   array       : Array,
                   kernel_size : int) -> Array:
        """
        Downsamples the input array by kernel_size.

        Parameters
        ----------
        array : Array
            The input array to downsample.

        Returns
        -------
        kernel_size : int
            The size of the downsample kernel.
        """
        size_in = array.shape[0]
        size_out = size_in//kernel_size

        # Downsample first dimension
        array = array.reshape((size_in*size_out, kernel_size)).sum(1)
        array = array.reshape(size_in, size_out).T

        # Downsample second dimension
        array = array.reshape((size_out*size_out, kernel_size)).sum(1)
        array = array.reshape(size_out, size_out).T
        return array


    def __call__(self, image):
        """
        Downsamples the input image by the internally stored kernel_size.

        Parameters
        ----------
        image : Array
            The image to downsample.

        Returns
        -------
        image : Array
            The downsampled image.
        """
        return self.downsample(image, self.kernel_size)


class Rotate(DetectorLayer):
    """
    Applies a rotation to the image using interpolation methods.

    Parameters
    ----------
    angle : Array, radians
        The angle by which to rotate the image in the clockwise direction.
    order : int
        The order of the interpolation.
    """
    angle   : Array


    def __init__(self  : DetectorLayer, 
                 angle : Array, 
                 order : int = 1) -> DetectorLayer:
        """
        Constructor for the Rotate class.

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


    def __call__(self : DetectorLayer, image : Array) -> Array:
        """
        Applies the rotation to an image.

        Parameters
        ----------
        image : Array
            The image to rotate.

        Returns
        -------
        image : Array
            The rotated image.
        """
        return rotate(image, self.angle, self.order)