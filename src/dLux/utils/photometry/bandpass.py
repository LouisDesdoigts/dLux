"""Wavelength sampling, independent of spectra and optical propagation."""

from __future__ import annotations

from numbers import Integral, Real
from math import isfinite

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

from .. import grids
from ..numerical.units import canonical_unit, unit_factor

__all__ = ["BandPass"]


def _wavelengths(value, **context):
    """Read a wavelength vector in metres from an array or sampling object."""
    if isinstance(value, grids.UnitSpec):
        value = value.resolve(**context)
        if value.ndim != 1:
            raise ValueError("Wavelength sampling must have one dimension.")
        if value.unit is not None:
            canonical_unit(value.unit, dimension="length")
        value = value.axes[0]
    else:
        value = zdx.resolve(value, **context)
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if value.ndim != 1 or value.size == 0:
        raise ValueError("Wavelengths must be a non-empty vector.")
    return value


def _fixed_float(value):
    """Accept Python sampling numbers, never arrays or deferred expressions."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "BandPass limits must be fixed Python numbers, not arrays or expressions."
        )
    return float(value)


def _fixed_range(value):
    """Store exactly two Python endpoints as immutable sampling metadata."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("range must contain two fixed Python endpoints.")
    return tuple(_fixed_float(endpoint) for endpoint in value)


class BandPass(grids.UnitSpec):
    """Fixed regular wavelength sampling, independent of spectra and responses.

    range=(min, max) gives the outer edges of n equal-width wavelength bins.
    Wavelengths are bin centres; both endpoints belong to the full interval.
    min/max/n and wl/delta_wl/n are alternative constructor spellings. Explicit
    wavelengths are centres, with outer edges inferred from their regular spacing.
    Constructors store one static range tuple and a sample count.
    Build a new BandPass to change sampling; fit spectral parameters on that grid.
    All returned coordinates and integration widths are in metres.

    In an Imager, n is the number of optical propagations. Fine spectral sampling
    belongs to the Spectrum, Filter and QE definitions used to calculate weights.

    Examples
    --------
    ```python
    import dLux as dl

    # Endpoints bound three equal bins; wavelengths are their centres.
    band = dl.BandPass(range=(500, 800), n=3, unit="nm")
    print(band)
    print(band.wavelengths)
    print(band.edges)
    ```
    """

    range: tuple[float, float] = eqx.field(static=True, converter=_fixed_range)
    n: int = eqx.field(static=True, default=1)
    unit: str = eqx.field(static=True, default="m")

    def __init__(
        self,
        wl=None,
        delta_wl=None,
        n=1,
        *,
        min=None,
        max=None,
        range=None,
        wavelengths=None,
        unit="m",
        alias=None,
    ):
        """Define fixed, equal-width wavelength bins or one monochromatic point.

        Parameters
        ----------
        wl : float or None
            Fixed Python central wavelength in unit. With no delta_wl and n=1,
            represents a monochromatic point.
        delta_wl : float or None
            Fixed Python full interval width in unit; needed for multiple samples
            with wl.
        n : int
            Positive number of equal-width bins, default one. In Imager this is the
            number of optical wavelength samples.
        min : float or None
            Fixed Python lower outer edge in unit. Supply together with max,
            separately from other representations.
        max : float or None
            Fixed Python upper outer edge in unit, paired with min.
        range : tuple of float or None
            Fixed Python (lower, upper) outer edges in unit. Mutually exclusive with
            wl/delta_wl, min/max and wavelengths.
        wavelengths : ArrayLike or UnitSpec or None
            Regular wavelength centres used at construction to infer static
            endpoints and count. Traced arrays cannot define this metadata.
        unit : str
            Wavelength unit, default m. Stored endpoints use this unit; returned
            coordinates and widths use metres.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.unit = self.validate_unit(unit, dimension="length")
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer.")

        # Canonical endpoints make independently constructed samplings comparable.
        if range is not None:
            if any(
                value is not None for value in (min, max, wl, delta_wl, wavelengths)
            ):
                raise ValueError(
                    "Supply range, min/max, wl/delta_wl, or wavelengths separately."
                )
            low, high = _fixed_range(range)
        elif min is not None or max is not None:
            if min is None or max is None:
                raise ValueError("Supply both min and max.")
            if any(value is not None for value in (wl, delta_wl, wavelengths)):
                raise ValueError(
                    "Supply min/max, wl/delta_wl, or wavelengths separately."
                )
            low, high = _fixed_float(min), _fixed_float(max)
        elif wavelengths is None:
            if wl is None:
                raise ValueError("Supply min/max, wl, or explicit wavelengths.")
            if n > 1 and delta_wl is None:
                raise ValueError("Multiple samples require delta_wl.")
            centre = _fixed_float(wl)
            width = 0.0 if delta_wl is None else _fixed_float(delta_wl)
            low, high = centre - width / 2, centre + width / 2
        else:
            if wl is not None or delta_wl is not None or n != 1:
                raise ValueError("Supply wavelengths or wl, delta_wl and n.")
            if isinstance(wavelengths, grids.UnitSpec):
                if wavelengths.ndim != 1 or self.unit != "m":
                    raise ValueError("A one-dimensional UnitSpec owns its own unit.")
                wavelengths = wavelengths.axes[0]
            # Explicit arrays are a setup convenience. Only Python endpoints are
            # retained; traced wavelength arrays cannot define static sampling.
            samples = np.atleast_1d(np.asarray(wavelengths, dtype=float))
            if samples.ndim != 1 or samples.size == 0:
                raise ValueError("wavelengths must be a non-empty vector.")
            grids.Axis(samples, unit=self.unit)
            low, high = float(samples[0]), float(samples[-1])
            n = samples.size
            if n > 1:
                spacing = (high - low) / (n - 1)
                low, high = low - spacing / 2, high + spacing / 2

        if not isfinite(low) or not isfinite(high) or low <= 0 or high < low:
            raise ValueError("Wavelengths and widths must be finite and positive.")
        if n > 1 and high == low:
            raise ValueError("Multiple samples require a positive wavelength interval.")
        self.range, self.n = (low, high), int(n)
        self.alias = alias

    @property
    def ndim(self) -> int:
        return 1

    @property
    def min(self) -> float:
        """Lower interval edge in the declared unit."""
        return self.range[0]

    @property
    def max(self) -> float:
        """Upper interval edge in the declared unit."""
        return self.range[1]

    @property
    def c(self) -> float:
        """Interval centre in the declared unit."""
        return (self.min + self.max) / 2

    @property
    def d(self) -> float | None:
        """Bin width in the declared unit, or None for a monochromatic point."""
        return None if self.min == self.max else (self.max - self.min) / self.n

    @property
    def bounds(self) -> tuple[float, float]:
        """Full integration interval in metres, including for a single sample."""
        factor = unit_factor(self.unit, dimension="length")
        return self.min * factor, self.max * factor

    @property
    def nodes(self) -> tuple[float, ...]:
        """Static metre coordinates for comparing and combining sampling grids."""
        factor = unit_factor(self.unit, dimension="length")
        if self.n == 1:
            return (self.c * factor,)
        return tuple((self.min + (i + 0.5) * self.d) * factor for i in range(self.n))

    @property
    def axes(self) -> tuple[Array]:
        return (np.asarray(self.nodes),)

    @property
    def wavelengths(self) -> Array:
        return self.axes[0]

    def same_sampling(self, other) -> bool:
        """Compare wavelength centres without comparing array values."""
        return isinstance(other, BandPass) and self.nodes == other.nodes

    @property
    def edges(self) -> Array:
        """n + 1 regularly spaced bin edges spanning the full range, in metres."""
        low, high = self.bounds
        return np.linspace(low, high, self.n + 1)

    @property
    def widths(self) -> Array | None:
        return self.sample()[1]

    def sample(self, **context) -> tuple[Array, Array | None]:
        """Return bin centres and equal integration widths, both in metres."""
        wavelengths = self.wavelengths
        if self.d is None:
            return wavelengths, None
        return wavelengths, np.full((self.n,), self.d * unit_factor(self.unit))

    def resample(self, n, d) -> BandPass:
        """Construct fixed sampling with new bin count/width and the same centre."""
        n = grids.as_size(n, 1, "n")[0]
        return BandPass(self.c, _fixed_float(d) * n, n, unit=self.unit)
