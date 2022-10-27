from __future__ import annotations
import jax.numpy as np
from abc import ABC, abstractmethod
from jax.scipy.signal import convolve
from jax.scipy.stats import norm
from equinox import tree_at
import dLux



__all__ = ["ApplyPixelResponse", "ApplyJitter", "ApplySaturation",
           "AddConstant", "IntegerDownsample", "Rotate", "FourierRotate"]


Array = np.ndarray


class DetectorLayer(dLux.base.Base, ABC):
    """
    A base Detector layer class to help with type checking throuhgout the rest
    of the software.
    """


    @abstractmethod
    def __call__(self : DetectorLayer, image : Array) -> Array:
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
    pixel_response: np.ndarray


    def __init__(self : DetectorLayer, pixel_response) -> DetectorLayer:
        """
        Constructor for the ApplyPixelResponse class.

        Parameters
        ----------
        pixel_response : Array
            The pixel_response to apply to the input image. Must be a 2
            dimensional array equal to size of the image at time of application.
        """
        self.pixel_response = np.asarray(pixel_response, dtype=float)
        assert self.pixel_response.ndim == 2, \
        ("pixel_response must be a 2 dimensional array.")


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
    kernel_size : int
        The size of the convolution kernel to use.
    sigma : Array, pixles
        The standard deviation of the guassian kernel, in units of pixles.
    """
    kernel_size: int
    sigma: Array


    def __init__(self        : DetectorLayer,
                 sigma       : Array,
                 kernel_size : int = 10) -> DetectorLayer:
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        kernel_size : int = 10
            The size of the convolution kernel to use.
        sigma : Array, pixles
            The standard deviation of the guassian kernel, in units of pixles.
        """
        self.kernel_size = int(kernel_size)
        self.sigma       = np.asarray(sigma, dtype=float)
        assert self.sigma.ndim == 0, ("sigma must be scalar array.")


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
    saturation: Array


    def __init__(self : DetectorLayer, saturation : Array) -> DetectorLayer:
        """
        Constructor for the ApplySaturation class.

        Parameters
        ----------
        saturation : Array
            The value at which the saturation is applied.
        """
        self.saturation = np.asarray(saturation, dtype=float)
        assert self.saturation.ndim == 0, ("saturation must be a scalar array.")


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
    value: Array


    def __init__(self : DetectorLayer, value : Array) -> DetectorLayer:
        """
        Constructor for the AddConstant class.

        Parameters
        ----------
        value : Array
            The value to add to the image.
        """
        self.value = np.asarray(value, dtype=float)
        assert self.value.ndim == 0, ("value must be a scalar array.")


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
    kernel_size: int


    def __init__(self : DetectorLayer, kernel_size : int) -> DetectorLayer:
        """
        Constructor for the IntegerDownsample class.

        Parameters
        ----------
        kernel_size : int
            The size of the downsampling kernel.
        """
        self.kernel_size = int(kernel_size)


    def downsample(self        : DetectorLAyer,
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
        The angle by which to rotate the image in the {}wise direction.
    """
    angle : Array


    def __init__(self           : DetectorLayer,
                 angle          : Array,
                 name           : str = 'Rotate') -> DetectorLayer:
        """
        Constructor for the Rotate class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the image in the {}wise direction.
        name : str = 'Rotate'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.angle = np.asarray(angle, dtype=float)
        assert self.angle.nidm == 0, ("angle must be scalar array.")


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
        return dLux.utils.interpolation.rotate(image, self.angle)


class FourierRotate(DetectorLayer):
    """
    Applies a rotation to the image using fourier methods. This method is
    information conserving and can be repeatedly applied without any loss of
    fidelity, unlike methods that use interpolation.

    Parameters
    ----------
    angle : Array, radians
        The angle by which to rotate the image in the {}wise direction.
    padding : int
        A factor by which to pad the array in the Fourier Space Representation.
    """
    angle          : Array
    padding        : int


    def __init__(self           : DetectorLayer,
                 angle          : Array,
                 padding        : int  = 2,
                 name           : str  = 'FourierRotate') -> DetectorLayer:
        """
        Constructor for the FourierRotation class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the wavefront in the {}wise direction.
        padding : int = 2
            A factor by which to pad the array in the Fourier Space
            Representation.
        name : str = 'FourierRotate'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.angle   = np.asarray(angle, dtype=float)
        self.padding = int(padding)
        assert self.angle.nidm == 0, ("angle must be scalar array.")


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
        return dLux.utils.interpolation.fourier_rotate(image, self.angle,
                                                         self.padding)