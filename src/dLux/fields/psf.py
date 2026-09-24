"""Sampled optical point responses."""

from __future__ import annotations

import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from . import wavefront as wavefront_module, intensity

__all__ = ["PSF"]


class PSF(intensity.Intensity):
    """A continuous point-spread function sampled on a physical grid.

    A PSF is the pre-detector response to a point source. Its final two axes are
    spatial, while leading axes retain unresolved wavelength or vectorisation
    dimensions. Construction and wavefront conversion never integrate those axes;
    an optical system owns spectral integration because only it can determine
    whether the realised spatial grids are compatible.

    Parameters
    ----------
    data : Array
        Real point-response values with shape ``(..., ny, nx)``.
    grid : Grid
        Continuous spatial sampling matched to the final array axes.
    """

    # Inherited fields
    array: Array
    grid: dlu.UnitSpec

    wavelengths: Array | None = None
    positions: Array | None = None

    def __init__(
        self,
        data: Array,
        grid: dlu.Grid | None = None,
        *,
        wavelengths=None,
        positions=None,
        pixel_scale=None,
        alias=None,
    ):
        """Store real point responses and their two-dimensional sampling.

        Parameters
        ----------
        data : Array
            Real values with shape (..., ny, nx). Leading axes are retained.
        grid : Grid
            Spatial sampling matched against the final two array axes.
        wavelengths : Array or None
            Metre wavelengths labelling the axes immediately before (ny, nx).
            Ordinary batches lead; a scalar adds no wavelength axis.
        positions : Array or None
            Radian positions shaped (components, 2), in response-array order.
            Source brightness and spectral weights are not stored on a PSF.
        """
        if not isinstance(grid, dlu.Grid) or pixel_scale is not None:
            from ..compat.sampling import psf_sampling

            grid = psf_sampling(data, grid, pixel_scale)
        super().__init__(data, grid, alias=alias)
        self.wavelengths = zdx.as_array(wavelengths, dtype=float)
        self.positions = zdx.as_array(positions, dtype=float)
        if self.wavelengths is not None:
            axes = self.wavelengths.ndim
            if self.array.shape[-2 - axes : -2] != self.wavelengths.shape:
                raise ValueError("PSF wavelength axes must match wavelengths.shape.")
        if self.positions is not None:
            if self.positions.ndim != 2 or self.positions.shape[-1] != 2:
                raise ValueError("PSF positions must have shape (components, 2).")

    @property
    def ndim(self) -> int:
        """Return the number of point-response axes preceding the spatial axes."""
        return self.batch_ndim

    @classmethod
    def from_wavefront(cls, wavefront, stokes=None) -> PSF:
        """Retain every wavelength's optical response without photometry."""
        if not isinstance(wavefront, wavefront_module.Wavefront):
            raise TypeError("wavefront must be a Wavefront.")
        return cls(
            wavefront.intensity_from_stokes(stokes),
            wavefront.grid,
            wavelengths=wavefront.wavelength,
        )

    def to_intensity(self, flux=1.0) -> intensity.Intensity:
        """Scale this point response into deterministic source intensity.

        Parameters
        ----------
        flux : float or Array
            Source flux broadcast over the PSF's leading axes. Two trailing singleton
            axes are appended so flux never varies across the spatial response.
        """
        flux = zdx.as_array(flux, dtype=float)
        return intensity.Intensity(self.data * flux[..., None, None], self.grid)
