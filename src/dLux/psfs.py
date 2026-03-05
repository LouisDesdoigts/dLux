from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import Array
from zodiax import Base
import dLux.utils as dlu

__all__ = ["PSF"]


class PSF(Base):
    """
    A simple class that holds the state of some PSF as it it transformed by detector
    layers.

    Attributes
    ----------
    data : Array
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
        data : Array
            The psf to be transformed by the detector.
        pixel_scale : Array
            The pixel scale of the psf.
        """
        self.data = np.asarray(data, dtype=float)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)

    @property
    def npixels(self: PSF) -> int:
        """
        Returns the side length of the arrays currently representing the psf.

        Returns
        -------
        npixels : int
            The number of pixels that represent the `PSF`.
        """
        return self.data.shape[-1]

    @property
    def ndim(self: PSF) -> int:
        """
        Returns the number of 'dimensions' of the psf. This is used to track the
        vectorised version of the psf returned from vmapping.

        Returns
        -------
        ndim : int
            The 'dimensionality' of dimensions of the psf.
        """
        return self.pixel_scale.ndim

    def downsample(self: PSF, n: int) -> PSF:
        """
        Downsamples the psf by a factor of n. This is done by summing the psf pixels in
        n x n blocks.

        Parameters
        ----------
        n : int
            The factor by which to downsample the psf.

        Returns
        -------
        psf : PSF
            The downsampled psf.
        """
        data = dlu.downsample(self.data, n, mean=False)
        pixel_scale = self.pixel_scale * n
        return self.set(data=data, pixel_scale=pixel_scale)

    def convolve(self: PSF, other: Array, method: str = "auto") -> PSF:
        """
        Convolves the psf with some input array.

        Parameters
        ----------
        other : Array
            The psf to convolve with.
        method : str = "auto"
            The method to use for the convolution. Can be "auto", "direct",
            or "fft". Is "auto" by default, which calls "direct".

        Returns
        -------
        psf : PSF
            The convolved psf.
        """
        return self.set(data=convolve(self.data, other, mode="same", method=method))

    def rotate(self: PSF, angle: float, method: str = "linear") -> PSF:
        """
        Rotates the psf by a given angle via interpolation.

        Parameters
        ----------
        angle : float
            The angle by which to rotate the psf.
        method : str = "linear"
            The interpolation method.

        Returns
        -------
        psf : PSF
            The rotated psf.
        """
        return self.set(data=dlu.rotate(self.data, angle, method=method))

    def resize(self: PSF, npixels: int) -> PSF:
        """
        Resizes the psf via a zero-padding or cropping operation.

        Parameters
        ----------
        npixels : int
            The size to resize the psf to.

        Returns
        -------
        psf : PSF
            The resized psf.
        """
        return self.set(data=dlu.resize(self.data, npixels))

    def flip(self: PSF, axis: tuple) -> PSF:
        """
        Flips the psf along the specified axes. Note we use 'ij' indexing, so axis 0 is
        the y-axis and axis 1 is the x-axis.

        Parameters
        ----------
        axis : tuple
            The axes along which to flip the PSF.

        Returns
        -------
        psf : PSF
            The new flipped PSF.
        """
        return self.set(data=np.flip(self.data, axis))

    def _magic_unified_op(self, other: Array | PSF | None, op: str) -> PSF:
        """
        Internal helper function to unify the logic of the magic methods for addition,
        subtraction, multiplication and division.

        Parameters
        ----------
        other : Array | PSF | None
            The object to operate with. Can be an array, a PSF, or None.
        op : str
            The operation to perform: 'add', 'subtract', 'multiply', or 'divide'.

        Returns
        -------
        psf : PSF
            The resulting PSF after applying the operation.
        """
        # Nones always return unchanged
        if other is None:
            return self

        # Check for supported types
        if not isinstance(other, (PSF, Array, float, int, complex)):
            raise TypeError(
                f"Unsupported type for {op}: {type(other)}. Must be an array, "
                "PSF, or None."
            )

        # Extract data if other is a PSF
        if isinstance(other, PSF):
            other = other.data

        # Apply the operation
        if op == "add":
            return self.add("phasor", other)
        elif op == "subtract":
            return self.add("phasor", -other)
        elif op == "multiply":
            return self.multiply("phasor", other)
        elif op == "divide":
            return self.multiply("phasor", 1 / other)
        else:
            raise ValueError(f"Unsupported operation '{op}'.")

    def __add__(self: PSF, other: PSF | Array | None) -> PSF:
        """
        Allows arrays or PSFs to be added together. Nones are ignored.
        """
        return self._magic_unified_op(other, "add")

    def __sub__(self: PSF, other: PSF | Array | None) -> PSF:
        """
        Allows arrays or PSFs to be subtracted. Nones are ignored.
        """
        return self._magic_unified_op(other, "subtract")

    def __mul__(self: PSF, other: PSF | Array | None) -> PSF:
        """
        Allows arrays or PSFs to be multiplied. Nones are ignored.
        """
        return self._magic_unified_op(other, "multiply")

    def __truediv__(self: PSF, other: PSF | Array | None) -> PSF:
        """
        Allows arrays or PSFs to be divided. Nones are ignored.
        """
        return self._magic_unified_op(other, "divide")

    def __iadd__(self: PSF, other: PSF | Array | None) -> PSF:
        """In-place addition."""
        return self.__add__(other)

    def __isub__(self: PSF, other: PSF | Array | None) -> PSF:
        """In-place subtraction."""
        return self.__sub__(other)

    def __imul__(self: PSF, other: PSF | Array | None) -> PSF:
        """In-place multiplication."""
        return self.__mul__(other)

    def __itruediv__(self: PSF, other: PSF | Array | None) -> PSF:
        """In-place division."""
        return self.__truediv__(other)
