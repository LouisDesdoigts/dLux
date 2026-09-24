"""Integrated spectral contributions, distinct from continuous spectral curves."""

from math import isclose

import jax.numpy as np
import zodiax as zdx
from jax import Array

from .bandpass import BandPass, _wavelengths

__all__ = ["Weights"]


class Weights(zdx.Module):
    """Integrated counts on one wavelength sampling, with wavelength last.

    Spectrum creates incident photon counts. Filters and QE multiply them without
    renormalising or applying integration widths again. After QE, the values are
    expected detected counts. Raw metre vectors are accepted for numerical results
    such as irregular integrated bins; ordinary imaging uses a BandPass.
    """

    value: Array
    bandpass: BandPass | Array

    def __init__(self, value, bandpass, *, alias=None):
        self.value = zdx.as_array(value, dtype=float)
        self.bandpass = (
            bandpass if isinstance(bandpass, BandPass) else _wavelengths(bandpass)
        )
        n = bandpass.n if isinstance(bandpass, BandPass) else self.bandpass.size
        if self.value.ndim == 0 or self.value.shape[-1] != n:
            raise ValueError("Weights must end in the bandpass wavelength axis.")
        self.alias = alias

    @property
    def wavelengths(self):
        """Wavelengths in metres."""
        return _wavelengths(self.bandpass)

    def weights(self, **context):
        """Return metre wavelengths and integrated values, without reintegration."""
        return _wavelengths(self.bandpass, **context), self.value

    def integrate(self):
        """Sum contributions over wavelength, retaining any leading batches."""
        return self.value.sum(-1)

    def __mul__(self, response):
        """Apply scalar or sampled dimensionless responses."""
        if isinstance(response, Weights):
            raise TypeError(
                "Multiply Weights by a response, not another Weights object."
            )
        return self.set(value=self.value * np.asarray(response))

    def __rmul__(self, response):
        return self * response

    def __add__(self, other):
        """Add independent contributions on the same wavelength samples.

        BandPass bins are checked through static metadata, including under JIT.
        Raw wavelength vectors receive shape checks only; matching coordinates
        are the caller's responsibility. No runtime comparisons are installed.
        """
        if not isinstance(other, Weights):
            return NotImplemented
        first, second = self.wavelengths, other.wavelengths
        if first.shape != second.shape:
            raise ValueError("Adding Weights requires matching wavelength samples.")
        if isinstance(self.bandpass, BandPass) and isinstance(other.bandpass, BandPass):
            # Compare SI bounds with allowance for unit-conversion roundoff.
            bounds = zip(self.bandpass.bounds, other.bandpass.bounds)
            matches = all(
                isclose(first, second, rel_tol=1e-6) for first, second in bounds
            )
            if not matches:
                raise ValueError("Adding Weights requires matching wavelength bins.")
        return self.set(value=self.value + other.value)
