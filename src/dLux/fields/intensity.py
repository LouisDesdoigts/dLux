"""Expected optical or detector intensity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .image import Image

import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from . import wavefront as wavefront_module

__all__ = ["Intensity"]


class Intensity(dlu.Field):
    """A deterministic sampled optical intensity or expected detector signal.

    The stored values may represent a normalised photon distribution, expected
    counts, or another deterministic detector-domain signal. Their interpretation
    is set by the preceding model rather than enforced by this container.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    # Apply a response to expected counts, then produce detector pixels.
    grid = dl.Grid(8, d=0.1, unit="arcsec")
    signal = dl.Intensity(np.ones((8, 8)) * 10, grid)
    signal = dl.SensitivityLayer(response=0.8)(signal)
    image = dl.Detector(downsample=2)(signal)
    print(image)
    ```
    """

    # Inherited fields
    array: Array

    grid: dlu.Grid

    def __init__(self: Intensity, data: Array, grid: dlu.Grid, *, alias=None):
        """Initialise deterministic intensity data on a physical grid.

        Parameters
        ----------
        data : Array
            Real values with shape ``(..., ny, nx)``. Leading axes are retained.
        grid : Grid
            Spatial grid matched against the final ``(ny, nx)`` array axes.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.array = zdx.as_array(data, dtype=float)
        if self.data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        if not isinstance(grid, dlu.Grid):
            raise TypeError("grid must be a Grid.")
        super().__init__(self.array, grid, alias=alias)

    @property
    def intensity(self) -> Array:
        """Return the stored intensity-valued samples without squaring them."""
        return self.array

    @property
    def data(self) -> Array:
        """Return the sampled data stored in array."""
        return self.array

    def normalise(self, mode: str = "power", value: float = 1.0) -> Intensity:
        """Set the total sum (power) or peak to value across all array axes."""
        if mode not in ("power", "peak"):
            raise ValueError("mode must be 'power' or 'peak'")
        reduction = "sum" if mode == "power" else "peak"
        return self.set(array=dlu.normalise(self.array, reduction, scale=value))

    def downsample(self, n: int | tuple[int, int], mean: bool = False) -> Intensity:
        """Bin spatial samples, preserving total counts unless mean=True."""
        return super().downsample(n, mean=mean)

    @classmethod
    def from_wavefront(cls, wavefront, stokes=None) -> Intensity:
        """Construct deterministic intensity from a wavefront.

        ``stokes`` optionally defines the input polarisation with final component
        axis four. The wavefront grid is retained and the intensity has shape
        ``(..., ny, nx)``.
        """
        if not isinstance(wavefront, wavefront_module.Wavefront):
            raise TypeError("wavefront must be a Wavefront.")
        return cls(wavefront.psf_from_stokes(stokes), wavefront.grid)

    def to_image(self) -> "Image":
        """Wrap these samples as an Image without drawing noise."""
        # Image extends Intensity, so import it only when converting.
        from .image import Image

        return Image(self)
