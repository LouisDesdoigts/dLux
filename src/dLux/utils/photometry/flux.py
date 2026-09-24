"""Brightness allocations independent of positions and optical propagation."""

from __future__ import annotations

import jax
import jax.numpy as np
import zodiax as zdx
from jax import Array

__all__ = ["Flux", "ContrastFlux", "flux2mag", "mag2flux"]


def flux2mag(flux, reference=1.0):
    """Convert positive integrated flux to magnitude relative to reference flux.

    Both fluxes use the same units and band. reference=1 gives relative magnitudes;
    this function does not infer an AB, ST or Vega calibration.
    """
    return -2.5 * np.log10(np.asarray(flux) / np.asarray(reference))


def mag2flux(magnitude, reference=1.0):
    """Convert magnitude to integrated flux in the reference flux's units."""
    return np.asarray(reference) * 10 ** (-0.4 * np.asarray(magnitude))


class Flux(zdx.Module):
    """Incident photons over the current BandPass, before filter and QE losses.

    Only unit="photons" is supported. The value is an integrated photon count;
    normalising a spectral shape distributes this count across wavelengths.
    """

    value: Array | zdx.Expression
    unit: str = "photons"

    def __init__(self, value=1.0, unit="photons", *, alias=None):
        if unit != "photons":
            raise ValueError(
                f"Only flux unit 'photons' is supported; received {unit!r}."
            )
        self.value = zdx.as_array(value, dtype=float)
        self.unit = unit
        self.alias = alias

    def photons(self, wavelengths, weights, **context):
        """Distribute total photons over a spectral shape, with wavelength last.

        Numerical rescaling keeps normalisation stable for large spectral values.
        It cancels from the result and does not change the supplied total photons.
        """
        self = self.resolve(**dict(context, wavelengths=wavelengths))
        weights = np.asarray(weights)

        scale = jax.lax.stop_gradient(np.max(np.abs(weights), axis=-1, keepdims=True))
        fractions = weights / np.where(scale > 0, scale, 1)
        total = fractions.sum(-1, keepdims=True)
        fractions = fractions / np.where(total > 0, total, 1)
        return np.asarray(self.value)[..., None] * fractions


class ContrastFlux(zdx.Expression):
    """Primary and companion fluxes from a primary flux and relative contrasts.

    By default contrast is log10(companion/primary); log=False uses a linear ratio.
    Scalar contrast produces two fluxes; a vector supplies one ratio per companion.
    Omitted contrast gives equal brightness, in either convention. The resulting
    arrays follow the same primary-first order as astrometry.Binary/Companions.

    The numerical result may represent relative brightness. A normalised Source
    interprets it as incident photons in the calculation's BandPass, before
    filter and QE losses.
    """

    primary: Array | zdx.Expression
    contrast: Array | zdx.Expression
    log: bool = True

    def __init__(self, primary=1.0, contrast=None, *, log=True, alias=None):
        self.log = bool(log)
        if contrast is None:
            contrast = 0.0 if self.log else 1.0
        self.primary = zdx.as_array(primary, dtype=float)
        self.contrast = zdx.as_array(contrast, dtype=float)
        if not isinstance(self.primary, zdx.Expression) and self.primary.ndim != 0:
            raise ValueError("primary must be a scalar.")
        if not isinstance(self.contrast, zdx.Expression) and self.contrast.ndim > 1:
            raise ValueError("contrast must be scalar or a companion vector.")
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Return one primary flux followed by all companion fluxes."""
        primary, contrast = zdx.resolve((self.primary, self.contrast), **context)
        ratio = 10.0**contrast if self.log else contrast
        return np.concatenate((primary[None], primary * np.atleast_1d(ratio)))
