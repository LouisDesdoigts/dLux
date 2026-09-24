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

    Parameters
    ----------
    data : Array
        Real sampled values with shape ``(..., ny, nx)``. Leading axes are preserved
        as vectorisation axes.
    grid : Grid
        Spatial sampling whose physical ``(x, y)`` sizes must match the final array
        axes in reversed ``(nx, ny)`` order.

    Examples
    --------
    Construct an intensity and apply deterministic detector transformations:

    ```python
    import jax.numpy as np

    import dLux as dl
    import dLux.utils as dlu

    # Construct deterministic intensity data on a detector grid
    grid = dl.Grid(n=64, d=10, unit="um")
    data = dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply individual detector layers
    intensity = dl.Jitter(0.5)(intensity)
    intensity = dl.SensitivityLayer(response=0.8)(intensity)
    intensity = dl.BiasLayer(bias=5.0)(intensity)

    # Apply common deterministic field operations
    intensity = intensity.normalise(value=1e5)
    intensity = intensity.rotate(np.deg2rad(5))
    intensity = intensity.downsample(2)
    ```

    Convert a propagated wavefront into an intensity:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct and propagate a wavefront
    pupil_grid = dl.Grid(n=128, diam=1.0, unit="m")
    focal_grid = dl.Grid(n=64, d=10, unit="mas")
    wavefront = dl.Wavefront(wavelength=650e-9, grid=pupil_grid)

    optic = dl.SimpleCircular(diameter=1.0)(pupil_grid)
    propagator = dl.PropagatorLayer(focal_grid)
    wavefront = propagator(optic(wavefront))

    # Convert the wavefront into deterministic intensity
    intensity = wavefront.to_intensity()
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
