"""Spatial source models composed explicitly with spectra and optical systems."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as np
import jax.scipy as jsp
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from .psfs import PSF
from .spectra import BaseSpectrum

if TYPE_CHECKING:
    from .systems import OpticalSystem

__all__ = [
    "BaseSource",
    "Source",
    "PointSource",
    "PointSources",
    "BinarySource",
    "ResolvedSource",
    "PointResolvedSource",
]


class BaseSource(zdx.Base):
    """Base contract for spatial source models."""

    @abstractmethod
    def model(
        self,
        optics: OpticalSystem,
        spectrum: BaseSpectrum,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Any:  # pragma: no cover
        """Model this spatial source with a spectrum through an optical system."""


class Source(BaseSource):
    """Base class for concrete spatial source models."""


def _validate_return_mode(return_wf: bool, return_psf: bool) -> None:
    if return_wf and return_psf:
        raise ValueError("Cannot return both Wavefront and PSF objects.")


def _validate_spectrum(spectrum: BaseSpectrum) -> BaseSpectrum:
    if not isinstance(spectrum, BaseSpectrum):
        raise TypeError("spectrum must be a BaseSpectrum.")
    return spectrum


def _as_position(position) -> Array:
    position = np.asarray(position, dtype=float)
    if position.shape != (2,):
        raise ValueError("position must have shape (2,).")
    return position


class PointSource(Source):
    """A point source with angular position and flux."""

    position: Array
    flux: Array

    def __init__(self, position=np.zeros(2), flux=1.0):
        self.position = _as_position(position)
        self.flux = np.asarray(flux, dtype=float)

    def model(
        self,
        optics,
        spectrum,
        return_wf=False,
        return_psf=False,
    ):
        _validate_return_mode(return_wf, return_psf)
        spectrum = _validate_spectrum(spectrum)
        wavelengths, weights = spectrum.params()
        return optics.propagate(
            wavelengths,
            self.position,
            weights * self.flux,
            return_wf,
            return_psf,
        )


class PointSources(Source):
    """A vectorised collection of point-source positions and fluxes."""

    position: Array
    flux: Array

    def __init__(self, position=np.zeros((1, 2)), flux=None):
        self.position = np.asarray(position, dtype=float)
        if self.position.ndim != 2 or self.position.shape[-1] != 2:
            raise ValueError("position must have shape (n, 2).")
        self.flux = (
            np.ones(len(self.position))
            if flux is None
            else np.asarray(flux, dtype=float)
        )
        if self.flux.shape != self.position.shape[:1]:
            raise ValueError("flux must have shape (n,).")

    def model(
        self,
        optics,
        spectrum,
        return_wf=False,
        return_psf=False,
    ):
        _validate_return_mode(return_wf, return_psf)
        spectrum = _validate_spectrum(spectrum)
        wavelengths, weights = spectrum.params()

        def propagate(position, flux):
            return optics.propagate(
                wavelengths,
                position,
                weights * flux,
                return_wf=True,
            )

        wavefronts = eqx.filter_vmap(propagate)(self.position, self.flux)
        if return_wf:
            return wavefronts
        data = wavefronts.psf.sum((0, 1))
        spec = wavefronts.spec
        d = spec.d[0, 0] if spec.d is not None and spec.d.ndim > 1 else spec.d
        c = spec.c[0, 0] if spec.c is not None and spec.c.ndim > 1 else spec.c
        psf = PSF(data, spec.set(d=d, c=c))
        return psf if return_psf else psf.data


class ResolvedSource(PointSource):
    """A point-centred source convolved with a supplied spatial distribution."""

    distribution: Array

    def __init__(
        self,
        position=np.zeros(2),
        flux=1.0,
        distribution=np.ones((3, 3)),
    ):
        self.distribution = np.asarray(distribution, dtype=float)
        if self.distribution.ndim != 2:
            raise ValueError("distribution must be a 2d array.")
        super().__init__(position, flux)

    def model(
        self,
        optics,
        spectrum,
        return_wf=False,
        return_psf=False,
    ):
        _validate_return_mode(return_wf, return_psf)
        if return_wf:
            raise NotImplementedError(
                "Wavefront information cannot be preserved through convolution."
            )
        spectrum = _validate_spectrum(spectrum)
        wavelengths, weights = spectrum.params()
        psf = optics.propagate(
            wavelengths,
            self.position,
            weights * self.flux,
            return_psf=True,
        )
        data = jsp.signal.convolve(psf.data, self.distribution, mode="same")
        output = PSF(data, psf.spec)
        return output if return_psf else output.data


class BinarySource(Source):
    """A two-point source parameterised by separation and contrast."""

    position: Array
    mean_flux: Array
    separation: Array
    position_angle: Array
    contrast: Array

    def __init__(
        self,
        position=np.zeros(2),
        mean_flux=1.0,
        separation=0.0,
        position_angle=np.pi / 2,
        contrast=1.0,
    ):
        self.position = _as_position(position)
        self.mean_flux = np.asarray(mean_flux, dtype=float)
        self.separation = np.asarray(separation, dtype=float)
        self.position_angle = np.asarray(position_angle, dtype=float)
        self.contrast = np.asarray(contrast, dtype=float)

    def model(
        self,
        optics,
        spectrum,
        return_wf=False,
        return_psf=False,
    ):
        positions = dlu.positions_from_sep(
            self.position,
            self.separation,
            self.position_angle,
        )
        flux = dlu.fluxes_from_contrast(self.mean_flux, self.contrast)
        return PointSources(positions, flux).model(
            optics,
            spectrum,
            return_wf,
            return_psf,
        )


class PointResolvedSource(ResolvedSource):
    """A point source plus a co-located resolved component."""

    contrast: Array

    def __init__(
        self,
        position=np.zeros(2),
        flux=1.0,
        distribution=np.ones((3, 3)),
        contrast=1.0,
    ):
        self.contrast = np.asarray(contrast, dtype=float)
        super().__init__(position, flux, distribution)

    def model(
        self,
        optics,
        spectrum,
        return_wf=False,
        return_psf=False,
    ):
        _validate_return_mode(return_wf, return_psf)
        if return_wf:
            raise NotImplementedError(
                "Wavefront information cannot be preserved through convolution."
            )
        spectrum = _validate_spectrum(spectrum)
        wavelengths, weights = spectrum.params()
        flux = dlu.fluxes_from_contrast(self.flux, self.contrast)
        psf = optics.propagate(
            wavelengths,
            self.position,
            weights,
            return_psf=True,
        )
        point = flux[0] * psf.data
        resolved = jsp.signal.convolve(
            flux[1] * psf.data,
            self.distribution,
            mode="same",
        )
        output = PSF(point + resolved, psf.spec)
        return output if return_psf else output.data
