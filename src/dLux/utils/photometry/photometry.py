"""Independent spectral response and integration calculations."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

from .bandpass import BandPass, _wavelengths
from .spectral import BaseSpectrum, Spectrum, Filter
from .integration import integrate_spectrum
from .weights import Weights

__all__ = ["Photometry"]


class Photometry(zdx.Module):
    """Combine incident density with filter transmission and detector QE.

    Sampling is supplied when calculating an integral. Spectra define density;
    BandPass defines integration widths; responses multiply without normalisation.
    Spectrum owns incident brightness and normalisation; filter and detector losses
    are applied afterwards without changing that scale.
    """

    spectrum: BaseSpectrum | Weights | None = None
    filter: Filter | None = None
    qe: Filter | None = None

    def __init__(self, spectrum=None, *, filter=None, qe=None, alias=None):
        """Store definitions; raw values and Expressions are promoted to curves.

        ``spectrum=None`` uses a flat shape containing one photon, or permits
        response-only use through response or apply. QE has the same curve contract
        as a filter; it needs no separate class.
        """
        if spectrum is None or isinstance(spectrum, (BaseSpectrum, Weights)):
            self.spectrum = spectrum
        else:
            self.spectrum = Spectrum(spectrum)
        self.filter = (
            filter if filter is None or isinstance(filter, Filter) else Filter(filter)
        )
        self.qe = qe if qe is None or isinstance(qe, Filter) else Filter(qe)
        self.alias = alias

    def sample(self, wavelengths=None, **context) -> Array:
        """Evaluate density times filter/QE at points, without applying flux.

        A BandPass may supply its centres, but sample never integrates. Integrated
        Weights are not densities and cannot be sampled as a continuous curve.
        """
        self = self.populate()
        spectrum = Spectrum() if self.spectrum is None else self.spectrum
        if isinstance(spectrum, Weights):
            raise TypeError("Integrated Weights have no density to sample.")
        sampling = spectrum.bandpass if wavelengths is None else wavelengths
        values = spectrum.sample(sampling, **context)
        return values * self.response(sampling, **context)

    def __call__(self, bandpass=None, **context) -> Array | Weights:
        """Sample points, integrate a BandPass, or apply losses to Weights."""
        self, bandpass = zdx.populate((self, bandpass))
        spectrum = Spectrum() if self.spectrum is None else self.spectrum

        # Integrated inputs already own their sampling and must not be reintegrated.
        if isinstance(spectrum, Weights):
            if bandpass is not None:
                raise ValueError("Integrated weights already define their sampling.")
            return spectrum * self.response(spectrum.bandpass, **context)

        # Choose stored/explicit sampling before dispatch, so both use one route.
        sampling = spectrum.bandpass if bandpass is None else bandpass
        if isinstance(sampling, BandPass):
            return integrate_spectrum(
                spectrum,
                sampling,
                filter=self.filter,
                qe=self.qe,
                **context,
            )
        return self.sample(sampling, **context)

    def integrate(self, bandpass=None, **context) -> Array:
        """Sum integrated counts, retaining leading component axes."""
        result = self(bandpass, **context)
        if not isinstance(result, Weights):
            raise TypeError("Integration requires a BandPass defining bin boundaries.")
        return result.integrate()

    def response(self, wavelengths, **context) -> Array:
        """Sample combined filter transmission and QE without evaluating the spectrum.

        Wavelengths are in metres. The result ends in the wavelength sample axis;
        any leading response axes follow ordinary broadcasting. Missing responses
        have unit transmission. No integration or normalisation is performed.
        """
        self = self.populate()
        # Keep the sampling object when calling curves: equal BandPass centres
        # can use stored samples directly instead of constructing an interpolator.
        values = np.ones_like(_wavelengths(wavelengths, **context))
        for curve in (self.filter, self.qe):
            if curve is not None:
                values = values * curve.sample(wavelengths, **context)
        return values

    def apply(self, wavelengths, weights, **context) -> Array:
        """Multiply responses into integrated masses without changing their measure."""
        weights = np.asarray(weights)
        samples = _wavelengths(wavelengths, **context)
        if weights.ndim == 0 or weights.shape[-1] != samples.size:
            raise ValueError("Weights must end in the wavelength sample axis.")
        if self.filter is None and self.qe is None:
            return weights
        return weights * self.response(wavelengths, **context)
