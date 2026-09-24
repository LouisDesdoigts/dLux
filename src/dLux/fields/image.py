"""Detector images and explicit noise realisations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
import jax.random as jr
from jax import Array

import dLux.utils as dlu
from . import intensity as intensity_module

__all__ = ["Image"]


class Image(intensity_module.Intensity):
    """Detector samples and their grid, without uncertainty metadata.

    ``data`` is an array with trailing ``(ny, nx)`` axes or an Intensity whose
    sampling is retained. Arithmetic and resampling use the ordinary Field API.
    Noise methods draw samples; their parameters are explicit inputs and no
    probability distribution is attached to the result.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    import jax.random as jr

    # Keep expected detector samples separate from a noisy realisation.
    grid = dl.Grid(8, d=0.1, unit="arcsec")
    image = dl.Image(np.ones((8, 8)) * 10, grid)
    simulated = image.simulate(jr.key(0), n_frames=2, read_noise=1.0)
    print(simulated)
    ```
    """

    # Inherited fields
    array: Array
    grid: dlu.UnitSpec

    def __init__(
        self,
        data: Array | intensity_module.Intensity,
        grid: dlu.Grid | None = None,
        *,
        alias=None,
    ):
        """Wrap detector samples or retain the sampling of an Intensity.

        Parameters
        ----------
        data : Array or Intensity
            Real (..., ny, nx) detector samples, or an Intensity whose sampling is
            retained.
        grid : Grid or None
            Required with an array; omit when data is an Intensity.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if isinstance(data, intensity_module.Intensity):
            if grid is not None:
                raise ValueError("grid must not be supplied with an Intensity.")
            data, grid = data.data, data.grid

        super().__init__(data, grid, alias=alias)

    @classmethod
    def from_intensity(cls, intensity) -> Image:
        """Construct an image from deterministic sampled intensity."""
        if not isinstance(intensity, intensity_module.Intensity):
            raise TypeError("intensity must be an Intensity.")
        return cls(intensity)

    @property
    def fourier_transform(self) -> Array:
        """Centred Fourier transform over the final two spatial axes."""
        transformed = np.fft.fft2(self.field, axes=(-2, -1))
        return np.fft.fftshift(transformed, axes=(-2, -1))

    @property
    def amplitude_spectrum(self) -> Array:
        """Modulus of the Fourier transform."""
        return np.abs(self.fourier_transform)

    @property
    def power_spectrum(self) -> Array:
        """Squared modulus of the Fourier transform."""
        return self.amplitude_spectrum**2

    def add_poisson_noise(self, key: Array) -> Image:
        """Draw Poisson counts using the stored non-negative expectation."""
        data = jr.poisson(key, self.field).astype(self.field.dtype)
        return self.set(field=data)

    def add_read_noise(self, key: Array, sigma: float | Array) -> Image:
        """Add Gaussian noise; sigma broadcasts against the image samples."""
        sigma = np.asarray(sigma, dtype=self.field.dtype)
        noise = jr.normal(key, self.field.shape, self.field.dtype) * sigma
        return self.set(field=self.field + noise)

    def simulate(
        self, key: Array, n_frames: int = 1, *, read_noise: float | Array | None = None
    ) -> Image:
        """Average independent Poisson and optional Gaussian noise realisations.

        ``n_frames`` is a fixed Python integer. ``read_noise`` gives the Gaussian
        standard deviation per frame, in the same units as the image samples.
        Full detector simulations, including gain and response layers, belong to
        Imager and Detector.
        """
        if not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("n_frames must be a positive integer.")

        def simulate_frame(key):
            photon_key, read_key = jr.split(key)
            image = self.add_poisson_noise(photon_key)
            if read_noise is not None:
                image = image.add_read_noise(read_key, read_noise)
            return image.field

        samples = eqx.filter_vmap(simulate_frame)(jr.split(key, n_frames))
        return self.set(field=samples.mean(0))
