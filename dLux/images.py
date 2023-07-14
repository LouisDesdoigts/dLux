from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import Array
from zodiax import Base
import dLux.utils as dlu


__all__ = ["Image"]


class Image(Base):
    """
    A class representing some image as it is transformed by the detector. It
    tracks the image via the `image` attribute, and the pixel scale via the
    `pixel_scale` attribute.

    Attributes
    ----------
    image : Array
        The image as it is transformed by the detector.
    pixel_scale : Array
        The pixel scale of the image.
    """

    image: Array
    pixel_scale: Array

    def __init__(self: Image, image: Array, pixel_scale: Array):
        """
        Parameters
        ----------
        image : Array
            The image as it is transformed by the detector.
        pixel_scale : Array
            The pixel scale of the image.
        """
        self.image = np.asarray(image, dtype=float)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)

    @property
    def npixels(self: Image) -> int:
        """
        Returns the side length of the arrays currently representing the
        image.

        Returns
        -------
        pixels : int
            The number of pixels that represent the `Image`.
        """
        return self.image.shape[-1]

    def downsample(self: Image, n: int) -> Image:
        """
        Downsamples the image by a factor of n. This is done by summing the
        image pixels in n x n blocks.

        Parameters
        ----------
        n : int
            The factor by which to downsample the image.

        Returns
        -------
        image : Image
            The downsampled image.
        """
        downsampled = dlu.downsample(self.image, n, "sum")
        return self.set("image", downsampled)

    def convolve(self: Image, other: Array) -> Image:
        """
        Convolves the image with another image. This is done using the
        `jax.scipy.signal.convolve` function.

        Parameters
        ----------
        other : Array
            The image to convolve with.

        Returns
        -------
        image : Image
            The convolved image.
        """
        return self.set("image", convolve(self.image, other, mode="same"))

    def rotate(self: Image, angle: float, order: int = 1) -> Image:
        """
        Rotates the image by a given angle. This is done using interpolation
        methods.

        Parameters
        ----------
        angle : float
            The angle by which to rotate the image.
        order : int
            The order of the interpolation method to use.

        Returns
        -------
        image : Image
            The rotated image.
        """
        return self.set("image", dlu.rotate(self.image, angle, order=order))

    def __mul__(self: Image, other: Array) -> Image:
        """
        Magic method for the multiplication operator. This allows for the
        multiplication of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to multiply the image by.

        Returns
        -------
        image : Image
            The multiplied image.
        """
        return self.multiply("image", other)

    def __imul__(self: Image, other: Array) -> Image:
        """
        Magic method for the inplace multiplication operator. This allows for
        the inplace multiplication of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to multiply the image by.

        Returns
        -------
        image : Image
            The multiplied image.
        """
        return self.__mul__(other)

    def __add__(self: Image, other: Array) -> Image:
        """
        Magic method for the addition operator. This allows for the addition of
        the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to add to the image.

        Returns
        -------
        image : Image
            The added image.
        """
        return self.add("image", other)

    def __iadd__(self: Image, other: Array) -> Image:
        """
        Magic method for the inplace addition operator. This allows for the
        inplace addition of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to add to the image.

        Returns
        -------
        image : Image
            The added image.
        """
        return self.__add__(other)

    def __sub__(self: Image, other: Array) -> Image:
        """
        Magic method for the subtraction operator. This allows for the
        subtraction of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to subtract from the image.

        Returns
        -------
        image : Image
            The subtracted image.
        """
        return self.add("image", -other)

    def __isub__(self: Image, other: Array) -> Image:
        """
        Magic method for the inplace subtraction operator. This allows for the
        inplace subtraction of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to subtract from the image.

        Returns
        -------
        image : Image
            The subtracted image.
        """
        return self.__sub__(other)

    def __truediv__(self: Image, other: Array) -> Image:
        """
        Magic method for the division operator. This allows for the division of
        the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to divide the image by.

        Returns
        -------
        image : Image
            The divided image.
        """
        return self.divide("image", other)

    def __itruediv__(self: Image, other: Array) -> Image:
        """
        Magic method for the inplace division operator. This allows for the
        inplace division of the image by a scalar or another image.

        Parameters
        ----------
        other : Array
            The scalar or image to divide the image by.

        Returns
        -------
        image : Image
            The divided image.
        """
        return self.__truediv__(other)
