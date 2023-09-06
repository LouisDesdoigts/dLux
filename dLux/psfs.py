from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import Array
from zodiax import Base
import dLux.utils as dlu


__all__ = ["PSF"]


class PSF(Base):
    """
    A class representing some psf as it is transformed by the detector. It
    tracks the psf via the `psf` attribute, and the pixel scale via the
    `pixel_scale` attribute.

    Attributes
    ----------
    psf : Array
        The psf as it is transformed by the detector.
    pixel_scale : Array
        The pixel scale of the psf.
    """

    data: Array
    pixel_scale: Array

    def __init__(self: PSF, data: Array, pixel_scale: Array):
        """
        Parameters
        ----------
        psf : Array
            The psf as it is transformed by the detector.
        pixel_scale : Array
            The pixel scale of the psf.
        """
        self.data = np.asarray(data, dtype=float)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)

    @property
    def npixels(self: PSF) -> int:
        """
        Returns the side length of the arrays currently representing the
        psf.

        Returns
        -------
        pixels : int
            The number of pixels that represent the `PSF`.
        """
        return self.data.shape[-1]

    def downsample(self: PSF, n: int) -> PSF:
        """
        Downsamples the psf by a factor of n. This is done by summing the
        psf pixels in n x n blocks.

        Parameters
        ----------
        n : int
            The factor by which to downsample the psf.

        Returns
        -------
        psf : PSF
            The downsampled psf.
        """
        downsampled = dlu.downsample(self.data, n, "sum")
        return self.set("data", downsampled)

    def convolve(self: PSF, other: Array) -> PSF:
        """
        Convolves the psf with another psf. This is done using the
        `jax.scipy.signal.convolve` function.

        Parameters
        ----------
        other : Array
            The psf to convolve with.

        Returns
        -------
        psf : PSF
            The convolved psf.
        """
        return self.set("data", convolve(self.data, other, mode="same"))

    def rotate(self: PSF, angle: float, order: int = 1) -> PSF:
        """
        Rotates the psf by a given angle. This is done using interpolation
        methods.

        Parameters
        ----------
        angle : float
            The angle by which to rotate the psf.
        order : int
            The order of the interpolation method to use.

        Returns
        -------
        psf : PSF
            The rotated psf.
        """
        return self.set("data", dlu.rotate(self.data, angle, order=order))

    def resize(self: PSF, npixels: int) -> PSF:
        """
        Resizes the psf to a given size. This is done using interpolation
        methods.

        Parameters
        ----------
        npixels : int
            The size to resize the psf to.

        Returns
        -------
        psf : PSF
            The resized psf.
        """
        return self.set("data", dlu.resize(self.data, npixels))

    def flip(self: PSF, axis: tuple) -> PSF:
        """
        Flips the psf along the specified axes.

        Parameters
        ----------
        axis : tuple
            The axes along which to flip the PSF.

        Returns
        -------
        psf : PSF
            The new flipped PSF.
        """
        return self.set("data", np.flip(self.data, axis))

    def __mul__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the multiplication operator. This allows for the
        multiplication of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to multiply the psf by.

        Returns
        -------
        psf : PSF
            The multiplied psf.
        """
        return self.multiply("data", other)

    def __imul__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the inplace multiplication operator. This allows for
        the inplace multiplication of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to multiply the psf by.

        Returns
        -------
        psf : PSF
            The multiplied psf.
        """
        return self.__mul__(other)

    def __add__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the addition operator. This allows for the addition of
        the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to add to the psf.

        Returns
        -------
        psf : PSF
            The added psf.
        """
        return self.add("data", other)

    def __iadd__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the inplace addition operator. This allows for the
        inplace addition of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to add to the psf.

        Returns
        -------
        psf : PSF
            The added psf.
        """
        return self.__add__(other)

    def __sub__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the subtraction operator. This allows for the
        subtraction of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to subtract from the psf.

        Returns
        -------
        psf : PSF
            The subtracted psf.
        """
        return self.add("data", -other)

    def __isub__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the inplace subtraction operator. This allows for the
        inplace subtraction of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to subtract from the psf.

        Returns
        -------
        psf : PSF
            The subtracted psf.
        """
        return self.__sub__(other)

    def __truediv__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the division operator. This allows for the division of
        the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to divide the psf by.

        Returns
        -------
        psf : PSF
            The divided psf.
        """
        return self.divide("data", other)

    def __itruediv__(self: PSF, other: Array) -> PSF:
        """
        Magic method for the inplace division operator. This allows for the
        inplace division of the psf by a scalar or another psf.

        Parameters
        ----------
        other : Array
            The scalar or psf to divide the psf by.

        Returns
        -------
        psf : PSF
            The divided psf.
        """
        return self.__truediv__(other)
