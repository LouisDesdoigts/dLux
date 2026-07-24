"""Spectral wavelength and weight parameterisation."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

from .parametric import BaseParametric, resolve_parametric

__all__ = ["BaseSpectrum", "Spectrum"]


class BaseSpectrum(zdx.Base):
    """Base contract for spectral parameterisations."""


class Spectrum(BaseSpectrum):
    """Wavelength samples and their corresponding spectral weights."""

    wavelengths: Array | BaseParametric
    weights: Array | BaseParametric

    def __init__(self, wavelengths, weights=None):
        self.wavelengths = self._as_parameter(wavelengths)
        if weights is None:
            if isinstance(self.wavelengths, BaseParametric):
                raise ValueError(
                    "weights are required when wavelengths are parametric."
                )
            weights = np.ones_like(self.wavelengths)
        self.weights = self._as_parameter(weights)
        if not isinstance(self.wavelengths, BaseParametric) and not isinstance(
            self.weights, BaseParametric
        ):
            self._validate(*self.params())

    @staticmethod
    def _as_parameter(value):
        if isinstance(value, BaseParametric):
            return value
        return np.asarray(value, dtype=float)

    @staticmethod
    def _validate(wavelengths, weights):
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1d array.")
        if weights.ndim not in (1, 2):
            raise ValueError("weights must be a 1d or 2d array.")
        if weights.shape[-1] != wavelengths.shape[0]:
            raise ValueError("weights trailing axis must match the wavelength axis.")

    def params(self, **context: Any) -> tuple[Array, Array]:
        """Resolve and return wavelength samples and spectral weights."""
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
        self._validate(wavelengths, weights)
        return wavelengths, weights
