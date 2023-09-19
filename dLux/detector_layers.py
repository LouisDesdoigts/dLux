from __future__ import annotations
from abc import abstractmethod
import jax
import jax.numpy as np
from jax import Array
from jax.scipy.stats import norm, multivariate_normal
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
    kernel_size : int
        The size in pixels of the convolution kernel to use.
    r : float, arcseconds
        The magnitude of the jitter.
    shear : float
        The shear of the jitter. A radially symmetric Gaussian kernel would have a shear value of 0.
    phi : float, degrees
        The angle of the jitter.
    """

    kernel_size: int
    r: float = None
    shear: float = None
    phi: float = None

    def __init__(
        self: DetectorLayer,
        r: float,
        shear: float = 0,
        phi: float = 0,
        kernel_size: int = 10,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float, arcseconds
            The magnitude of the jitter.
        shear : float
            The shear of the jitter. A radially symmetric Gaussian kernel would have a shear value of 0.
        phi : float, degrees
            The angle of the jitter.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.r = r
        self.shear = shear
        self.phi = phi

    @property
    def covariance_matrix(self):
        """
        Generates the covariance matrix for the multivariate normal distribution.

        Returns
        -------
        covariance_matrix : Array
            The covariance matrix.
        """
        rot_angle = np.radians(self.phi) - np.pi / 4

        # Construct the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # Construct the skew matrix
        skew_matrix = np.array(
            [[1, self.shear], [self.shear, 1]]
        )  # Ensure skew_matrix is symmetric

        # Compute the covariance matrix
        covariance_matrix = self.r * np.dot(
            np.dot(rotation_matrix, skew_matrix), rotation_matrix.T
        )

        return covariance_matrix

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised multivariate Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float, arcsec/pixel
            The pixel scale of the image.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        # Generate distribution
        extent = pixel_scale * self.kernel_size  # kernel size in arcseconds
        x = np.linspace(0, extent, self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = multivariate_normal.pdf(
            pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix
        )

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
        kernel = self.generate_kernel(
            dLux.utils.rad_to_arcsec(image.pixel_scale)
        )

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
