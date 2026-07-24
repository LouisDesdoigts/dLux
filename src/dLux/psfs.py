"""PSF container and PSF-space operations used by detector models."""

from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import Array
from .coordinates import CoordSpec
from .wavefronts import BaseSpatial

__all__ = ["PSF"]


class PSF(BaseSpatial):
    """
    A simple class that holds the state of a PSF as it is transformed by detector
    layers.

    ??? abstract "UML"
        ![UML](../assets/uml/PSF.png)

    Attributes
    ----------
    data : Array
        The PSF as it is transformed by the detector.
    pixel_scale : Array
        The pixel scale of the PSF.
    npixels : int, property
        Derived property from `data`; returns the PSF side length in pixels.
    ndim : int, property
        Derived property from `pixel_scale`; returns PSF vectorisation rank.
    """

    data: Array
    spec: CoordSpec

    @property
    def _field_name(self) -> str:
        return "data"

    def __init__(self: PSF, data: Array, spec: CoordSpec):
        """
        Parameters
        ----------
        data : Array
            The PSF to be transformed by the detector.
        pixel_scale : Array
            The pixel scale of the PSF.
        """
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        if not isinstance(spec, CoordSpec):
            raise TypeError("spec must be a CoordSpec.")
        spec = spec.broadcast(2)
        inferred_n = np.asarray(self.data.shape[-2:][::-1], int)
        if spec.n is None:
            spec = spec.set(n=inferred_n)
        elif tuple(int(value) for value in spec.n) != tuple(inferred_n):
            raise ValueError("data spatial shape must match spec.n.")
        BaseSpatial.__init__(self, spec)

    @classmethod
    def from_wavefront(cls, wavefront) -> PSF:
        """Construct a PSF from a wavefront's intensity and coordinate specification."""
        return cls(wavefront.psf, wavefront.spec)

    @property
    def batch_ndim(self: PSF) -> int:
        """
        Returns the number of dimensions of the PSF. This is used to track the
        vectorised version of the PSF returned from vmapping.

        Returns
        -------
        ndim : int
            The dimensionality of the PSF.
        """
        return self.data.ndim - 2

    def normalise(self: PSF, mode: str = "power", value: float = 1.0) -> PSF:
        """
        Normalise the PSF.

        Parameters
        ----------
        mode : {"power","peak"} = "power"
            - "power": scales so ``sum(data) == value``.
            - "peak": scales so ``max(data) == value``.
        value : float = 1.0
            Target value for the selected mode.

        Returns
        -------
        psf : PSF
            New PSF scaled to achieve the normalisation.
        """
        if mode == "power":
            scale = value / self.data.sum()
        elif mode == "peak":
            scale = value / self.data.max()
        else:
            raise ValueError("mode must be 'power' or 'peak'")
        return self.multiply("data", scale)

    def convolve(self: PSF, other: Array, method: str = "auto") -> PSF:
        """
        Convolves the PSF with an input array.

        Parameters
        ----------
        other : Array
            The array to convolve with the PSF.
        method : str = "auto"
            The method to use for the convolution. Can be "auto", "direct",
            or "fft". Is "auto" by default, which calls "direct".

        Returns
        -------
        psf : PSF
            The convolved PSF.
        """
        return self.set(data=convolve(self.data, other, mode="same", method=method))

    def _magic_unified_op(self: PSF, other: Array | PSF | None, op: str) -> PSF:
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

        return self._apply_field_op(other, op)
