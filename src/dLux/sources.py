"""Spectral and spatial source models."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.scipy as jsp
from jax import Array

import dLux.utils as dlu
from .parametric import Parametric, ParametricHolder, resolve_parametric
from .fields import PSF

__all__ = [
    "BaseSource",
    "Spectrum",
    "Source",
    "BinarySource",
]

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

    def source_params(self, nsource=None, **context):
        """Resolve flux and distribution in canonical source units."""
        flux = resolve_parametric(self.flux, source=self, **context)
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
        distribution = resolve_parametric(
            self.distribution,
            source=self,
            **context,
        )
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
            distribution = np.broadcast_to(
                distribution,
                leading + distribution.shape,
            )
        else:
            extra = len(leading) - 1
            distribution = distribution.reshape(
                (distribution.shape[0],) + (1,) * extra + distribution.shape[-2:]
            )
            distribution = np.broadcast_to(
                distribution,
                leading + distribution.shape[-2:],
            )
        shape = data.shape
        convolved = eqx.filter_vmap(
            lambda image, kernel: jsp.signal.convolve(
                image,
                kernel,
                mode="same",
            )
        )(
            data.reshape((-1,) + shape[-2:]),
            distribution.reshape((-1,) + distribution.shape[-2:]),
        )
        return convolved.reshape(shape)

    def _model_components(
        self,
        optics,
        wavelengths,
        weights,
        position,
        flux,
        distribution,
        return_all,
    ):
        if position.ndim == 1:
            if weights.ndim != 1:
                raise ValueError(
                    "Single-component source weights must be one-dimensional."
                )
            result = optics.propagate(
                wavelengths, position, weights * flux, return_all=True
            )
            if distribution is None:
                return result if return_all else result["PSF"]
            psf = PSF(
                self._convolve(result["PSF"].data, distribution), result["PSF"].spec
            )
            if return_all:
                return {"Wavefront": result["Wavefront"], "PSF": psf}
            return psf

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

        results = eqx.filter_vmap(propagate)(position, flux, weights)
        wavefronts = results["Wavefront"]
        psf = results["PSF"]
        if distribution is not None:
            psf = psf.set(data=self._convolve(psf.data, distribution))
        if return_all:
            return {"Wavefront": wavefronts, "PSF": psf}
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
        if not isinstance(self.wavelengths, Parametric) and not isinstance(
            self.weights, Parametric
        ):
            self.spectrum_params()

    def spectrum_params(self, **context: Any) -> tuple[Array, Array]:
        """Resolve wavelengths and weights in canonical wavelength units."""
        wavelengths = np.asarray(
            resolve_parametric(self.wavelengths, spectrum=self, **context),
            dtype=float,
        )
        weights = np.asarray(
            resolve_parametric(
                self.weights,
                spectrum=self,
                wavelengths=wavelengths,
                variables=wavelengths,
                **context,
            ),
            dtype=float,
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

    position: Array | Parametric | None

    def __init__(
        self,
        wavelengths,
        position=None,
        flux=None,
        weights=None,
        distribution=None,
        units=None,
    ):
        self.wavelengths = _as_parameter(wavelengths)
        if weights is None:
            if isinstance(self.wavelengths, Parametric):
                raise ValueError(
                    "weights are required when wavelengths are parametric."
                )
            weights = np.ones_like(self.wavelengths)
        self.weights = _as_parameter(weights)
        self.position = _as_parameter(position)
        self.flux = _as_parameter(flux)
        self.distribution = _as_parameter(distribution)
        self.units = _merge_units(units)
        if not any(
            isinstance(value, Parametric)
            for value in (
                self.wavelengths,
                self.weights,
                self.position,
                self.flux,
                self.distribution,
            )
        ):
            self.params()

    def params(self):
        """Resolve all point-source parameters in canonical units."""
        wavelengths, weights = self.spectrum_params()
        position = resolve_parametric(
            self.position,
            source=self,
            wavelengths=wavelengths,
        )
        position = (
            np.zeros(2) if position is None else np.asarray(position, dtype=float)
        )
        if position.shape != (2,):
            raise ValueError("position must have shape (2,).")
        position = position * dlu.unit_factor(self.units["position"])
        flux, distribution = self.source_params(wavelengths=wavelengths)
        return wavelengths, weights, position, flux, distribution

    def model(self, optics, return_all=False):
        """Model this point source through an optical system."""
        return self._model_components(optics, *self.params(), return_all)


class BinarySource(BaseSource, Spectrum):
    """A binary source parameterised by centre, separation, and contrast."""

    centre: Array | Parametric | None
    separation: Array | Parametric
    position_angle: Array | Parametric
    contrast: Array | Parametric

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
        self.wavelengths = _as_parameter(wavelengths)
        if weights is None:
            if isinstance(self.wavelengths, Parametric):
                raise ValueError(
                    "weights are required when wavelengths are parametric."
                )
            weights = np.ones_like(self.wavelengths)
        self.weights = _as_parameter(weights)
        self.centre = _as_parameter(centre)
        self.separation = _as_parameter(separation)
        self.position_angle = _as_parameter(position_angle)
        self.contrast = _as_parameter(contrast)
        self.flux = _as_parameter(flux)
        self.distribution = _as_parameter(distribution)
        self.units = _merge_units(units)
        if not any(
            isinstance(value, Parametric)
            for value in (
                self.wavelengths,
                self.weights,
                self.centre,
                self.separation,
                self.position_angle,
                self.contrast,
                self.flux,
                self.distribution,
            )
        ):
            self.params()

    def params(self):
        """Resolve all binary parameters in canonical units."""
        wavelengths, weights = self.spectrum_params()
        centre = resolve_parametric(self.centre, source=self)
        centre = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
        if centre.shape != (2,):
            raise ValueError("centre must have shape (2,).")
        separation = np.asarray(
            resolve_parametric(self.separation, source=self),
            dtype=float,
        )
        position_angle = np.asarray(
            resolve_parametric(self.position_angle, source=self),
            dtype=float,
        )
        contrast = np.asarray(
            resolve_parametric(self.contrast, source=self),
            dtype=float,
        )
        factor = dlu.unit_factor(self.units["position"])
        position = dlu.positions_from_sep(
            centre * factor,
            separation * factor,
            position_angle,
        )
        mean_flux, _ = self.source_params(wavelengths=wavelengths)
        distribution = self.distribution_params(2, wavelengths=wavelengths)
        flux = dlu.fluxes_from_contrast(mean_flux, contrast)
        return wavelengths, weights, position, flux, distribution

    def model(self, optics, return_all=False):
        """Model both binary components through an optical system."""
        return self._model_components(optics, *self.params(), return_all)
