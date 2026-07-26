"""Spectral and spatial source models."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.scipy as jsp
from jax import Array

import dLux.utils as dlu
from .parametric import Parametric, ParametricHolder, resolve

__all__ = ["BaseSource", "Spectrum", "Source", "BinarySource"]

_DEFAULT_UNITS = {
    "wavelengths": "m",
    "position": "rad",
    "flux": "photon",
    "distribution": "linear",
}


def _as_parameter(value):
    if value is None or isinstance(value, Parametric):
        return value
    return np.asarray(value, dtype=float)


def _merge_units(units=None):
    units = {} if units is None else dict(units)
    unknown = set(units) - set(_DEFAULT_UNITS)
    if unknown:
        raise ValueError(f"Unknown source unit keys: {sorted(unknown)}.")
    return {**_DEFAULT_UNITS, **units}


def _convert_flux(flux, unit):
    unit = str(unit).strip()
    if unit.startswith("log_"):
        return np.exp(flux) * dlu.unit_factor(unit[4:])
    return flux * dlu.unit_factor(unit)


def _convert_distribution(distribution, unit):
    unit = str(unit).strip()
    if unit == "linear":
        return distribution
    if unit == "log":
        return np.exp(distribution)
    return _convert_flux(distribution, unit)


class BaseSource(ParametricHolder):
    """Source brightness and optional resolved distribution."""

    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(self, flux=None, distribution=None, units=None):
        self.flux = _as_parameter(flux)
        self.distribution = _as_parameter(distribution)
        self.units = _merge_units(units)

    def source_params(self, nsource=None, **context):
        """Resolve flux and distribution in canonical source units."""
        flux = resolve(self.flux, float, source=self, **context)
        flux = np.asarray(1.0 if flux is None else flux, dtype=float)
        if nsource is None:
            if flux.ndim != 0:
                raise ValueError("Single-source flux must be scalar.")
        else:
            if flux.ndim == 0:
                flux = np.broadcast_to(flux, (nsource,))
            if flux.shape != (nsource,):
                raise ValueError("Vectorised flux must have shape (nsource,).")
        flux = _convert_flux(flux, self.units["flux"])
        distribution = self.distribution_params(nsource, **context)
        return flux, distribution

    def distribution_params(self, nsource, **context):
        """Resolve and validate optional per-source spatial distributions."""
        distribution = resolve(self.distribution, float, source=self, **context)
        if distribution is None:
            return None
        distribution = np.asarray(distribution, dtype=float)
        valid = distribution.ndim == 2 or (
            nsource is not None
            and distribution.ndim == 3
            and distribution.shape[0] == nsource
        )
        if not valid:
            raise ValueError("distribution must have shape (y, x) or (nsource, y, x).")
        return _convert_distribution(distribution, self.units["distribution"])

    @staticmethod
    def _convolve(data, distribution):
        if data.ndim == 2:
            return jsp.signal.convolve(data, distribution, mode="same")
        leading = data.shape[:-2]
        if distribution.ndim == 2:
            distribution = np.broadcast_to(distribution, leading + distribution.shape)
        else:
            extra = len(leading) - 1
            distribution = distribution.reshape(
                (distribution.shape[0],) + (1,) * extra + distribution.shape[-2:]
            )
            distribution = np.broadcast_to(
                distribution, leading + distribution.shape[-2:]
            )
        shape = data.shape
        convolved = eqx.filter_vmap(
            lambda image, kernel: jsp.signal.convolve(image, kernel, mode="same")
        )(
            data.reshape((-1,) + shape[-2:]),
            distribution.reshape((-1,) + distribution.shape[-2:]),
        )
        return convolved.reshape(shape)

    def _propagate(self, optics, params):
        """Propagate one or more spatial source components."""
        wavelengths = params["wavelengths"]
        weights = params["weights"]
        position = params["position"]
        flux = params["flux"]
        if position.ndim == 1:
            if weights.ndim != 1:
                raise ValueError(
                    "Single-component source weights must be one-dimensional."
                )
            return optics.propagate(
                wavelengths, position, weights * flux, return_all=True
            )

        if weights.ndim == 1:
            weights = np.broadcast_to(weights, position.shape[:-1] + weights.shape)
        elif weights.shape[:-1] != position.shape[:-1]:
            raise ValueError(
                "Vectorised weights leading shape must match source positions."
            )

        def propagate(component_position, component_flux, component_weights):
            return optics.propagate(
                wavelengths,
                component_position,
                component_weights * component_flux,
                return_all=True,
            )

        return eqx.filter_vmap(propagate)(position, flux, weights)

    def model(self, optics, return_all=False):
        """Model the source through an optical system."""
        params = self.params()
        result = self._propagate(optics, params)
        psf = result["PSF"]
        distribution = params["distribution"]
        if distribution is not None:
            psf = psf.set(data=self._convolve(psf.data, distribution))
        if params["position"].ndim > 1:
            spec = psf.spec
            spec = spec.set(d=spec.d[0], c=None if spec.c is None else spec.c[0])
            psf = psf.set(data=psf.data.sum(0), spec=spec)
        result = {**result, "PSF": psf, "psf": psf.data}
        if return_all:
            return result
        return psf


class Spectrum(ParametricHolder):
    """Wavelength samples and their corresponding spectral weights."""

    wavelengths: Array | Parametric
    weights: Array | Parametric
    units: dict

    def __init__(self, wavelengths, weights=None, units=None):
        self.wavelengths = _as_parameter(wavelengths)
        if weights is None:
            if isinstance(self.wavelengths, Parametric):
                raise ValueError(
                    "weights are required when wavelengths are parametric."
                )
            weights = np.ones_like(self.wavelengths)
        self.weights = _as_parameter(weights)
        self.units = _merge_units(units)

    def spectrum_params(self, **context: Any) -> tuple[Array, Array]:
        """Resolve wavelengths and weights in canonical wavelength units."""
        wavelengths = resolve(self.wavelengths, float, spectrum=self, **context)
        weights = resolve(
            self.weights,
            float,
            spectrum=self,
            wavelengths=wavelengths,
            variables=wavelengths,
            **context,
        )
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1d array.")
        if weights.ndim not in (1, 2):
            raise ValueError("weights must be a 1d or 2d array.")
        if weights.shape[-1] != wavelengths.shape[0]:
            raise ValueError("weights trailing axis must match the wavelength axis.")
        wavelengths = wavelengths * dlu.unit_factor(self.units["wavelengths"])
        return wavelengths, weights

    def model(self, optics, return_all=False):
        """Model this spectrum as an on-axis, unit-flux point source."""
        return Source(self.wavelengths, weights=self.weights, units=self.units).model(
            optics, return_all
        )


class Source(BaseSource, Spectrum):
    """A point source combining spatial and spectral source properties."""

    wavelengths: Array | Parametric
    weights: Array | Parametric
    position: Array | Parametric | None
    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(
        self,
        wavelengths,
        position=None,
        flux=None,
        weights=None,
        distribution=None,
        units=None,
    ):
        self.position = _as_parameter(position)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve all point-source parameters in canonical units."""
        wavelengths, weights = self.spectrum_params()
        position = resolve(self.position, float, source=self, wavelengths=wavelengths)
        position = (
            np.zeros(2) if position is None else np.asarray(position, dtype=float)
        )
        if position.shape != (2,):
            raise ValueError("position must have shape (2,).")
        position = position * dlu.unit_factor(self.units["position"])
        flux, distribution = self.source_params(wavelengths=wavelengths)
        return {
            "wavelengths": wavelengths,
            "weights": weights,
            "position": position,
            "flux": flux,
            "distribution": distribution,
        }


class BinarySource(BaseSource, Spectrum):
    """A binary source parameterised by centre, separation, and contrast."""

    wavelengths: Array | Parametric
    weights: Array | Parametric
    centre: Array | Parametric | None
    separation: Array | Parametric
    position_angle: Array | Parametric
    contrast: Array | Parametric
    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(
        self,
        wavelengths,
        centre=None,
        separation=0.0,
        position_angle=np.pi / 2,
        contrast=1.0,
        flux=None,
        weights=None,
        distribution=None,
        units=None,
    ):
        self.centre = _as_parameter(centre)
        self.separation = _as_parameter(separation)
        self.position_angle = _as_parameter(position_angle)
        self.contrast = _as_parameter(contrast)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve all binary parameters in canonical units."""
        wavelengths, weights = self.spectrum_params()
        centre = resolve(self.centre, float, source=self)
        centre = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
        if centre.shape != (2,):
            raise ValueError("centre must have shape (2,).")
        separation = resolve(self.separation, float, source=self)
        position_angle = resolve(self.position_angle, float, source=self)
        contrast = resolve(self.contrast, float, source=self)
        factor = dlu.unit_factor(self.units["position"])
        position = dlu.positions_from_sep(
            centre * factor, separation * factor, position_angle
        )
        mean_flux, _ = self.source_params(wavelengths=wavelengths)
        distribution = self.distribution_params(2, wavelengths=wavelengths)
        flux = dlu.fluxes_from_contrast(mean_flux, contrast)
        return {
            "wavelengths": wavelengths,
            "weights": weights,
            "position": position,
            "flux": flux,
            "distribution": distribution,
        }
