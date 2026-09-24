"""Spectral shapes, their brightness, and multiplicative response curves."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array, core

from .. import grids
from ..grids.specs import UnitSpec
from ..numerical.interpolation import Interp
from ..numerical.polynomials import Polynomial
from ..numerical.units import canonical_unit, unit_factor
from .bandpass import BandPass, _wavelengths
from .weights import Weights
from .flux import Flux
from .integration import integrate_spectrum

__all__ = [
    "BaseSpectrum",
    "Spectrum",
    "PolySpectrum",
    "Filter",
    "Blackbody",
    "blackbody",
    "sample_curve",
]


def _curve_values(value, wavelengths, **context):
    """Evaluate an array or generic expression using spectral coordinates only."""
    local = dict(context, wavelengths=wavelengths, coordinates=wavelengths)
    local["variables"] = wavelengths
    local.pop("grid", None)
    values = np.asarray(zdx.resolve(value, **local), dtype=float)
    if values.ndim > 0 and values.shape[-1] not in (1, wavelengths.size):
        raise ValueError("A spectral value must end in its wavelength sample axis.")
    return np.broadcast_to(values, (*values.shape[:-1], wavelengths.size))


def sample_curve(value, knots=None, query=None, method="cubic2", **context) -> Array:
    """Evaluate a spectral value on native knots, then sample at query coordinates.

    Arrays and fixed Basis expressions use knots to locate their samples. Without
    knots, a scalar or analytic expression evaluates directly at the query points.
    Knots may be a BandPass, one-dimensional UnitSpec, or metre vector. Omitted
    query points use those native knots. No integration or normalisation is applied.

    BandPass tables extend their interpolant to the outer bin edges and are zero
    outside that interval. Raw calibration vectors end at their first/last knot.
    """
    query = knots if query is None else query
    if query is None:
        raise ValueError("Supply wavelengths or stored spectral sampling.")
    same_sampling = knots is query
    if isinstance(knots, BandPass):
        same_sampling = knots.same_sampling(query)

    # Resolve a fixed basis on its native sampling; analytic values without knots
    # use the query directly. Both Spectrum and Filter follow this same route.
    query = _wavelengths(query, **context)
    native = query if knots is None else _wavelengths(knots, **context)
    values = _curve_values(value, native, **context)
    if knots is None or same_sampling:
        return values

    bounds = knots.bounds if isinstance(knots, BandPass) else None
    if native.size == 1:
        low, high = (native[0], native[0]) if bounds is None else bounds
        return np.where((query >= low) & (query <= high), values, 0)
    fill = 0.0 if bounds is None else None
    values = Interp(values, native, ndim=1, method=method, fill=fill).sample(query)
    if bounds is not None:
        low, high = bounds
        values = np.where((query >= low) & (query <= high), values, 0)
    return values


class BaseSpectrum(zdx.Module):
    """A photon spectral shape with optional sampling and total brightness.

    Subclasses implement sample(wavelengths, **context), returning a density shape.
    Calling with a BandPass integrates the shape and applies flux before losses.
    A numerical flux means photons in this calculation's BandPass. Flux objects
    explicitly represent photon counts. None retains the absolute photon-density scale.
    """

    bandpass: grids.UnitSpec | Array | zdx.Expression | None = None
    flux: Array | zdx.Expression | Flux | None = None

    def __init__(self, bandpass=None, flux=1.0, *, alias=None):
        if isinstance(bandpass, grids.UnitSpec):
            if bandpass.ndim != 1:
                raise ValueError("Spectral sampling must be one-dimensional.")
            if bandpass.unit is not None:
                canonical_unit(bandpass.unit, dimension="length")
            self.bandpass = bandpass
        else:
            self.bandpass = zdx.as_array(bandpass, dtype=float)
        self.flux = flux if isinstance(flux, Flux) else zdx.as_array(flux, dtype=float)
        self.alias = alias

    @abstractmethod
    def sample(self, wavelengths=None, **context) -> Array:
        """Return the unnormalised photon-density shape at metre wavelengths."""

    def __call__(self, bandpass=None, **context) -> Array | Weights:
        """Sample point values for wavelengths, or integrate counts for a BandPass.

        Point sampling returns the unnormalised density shape. Only integration
        applies flux, using the calculation BandPass as its reference interval.
        Omitted sampling uses the stored bandpass.
        """
        self, bandpass = zdx.populate((self, bandpass))
        sampling = self.bandpass if bandpass is None else bandpass
        if not isinstance(sampling, BandPass):
            return self.sample(sampling, **context)
        return integrate_spectrum(self, sampling, **context)

    def integrate(self, bandpass=None, **context) -> Array:
        """Return total incident counts over a stored or explicit BandPass."""
        result = self(bandpass, **context)
        if not isinstance(result, Weights):
            raise TypeError("Integration requires a BandPass defining bin boundaries.")
        return result.integrate()


class Spectrum(BaseSpectrum):
    """An array, Basis, Polynomial or other expression defining a spectral shape.

    Optional bandpass locates stored samples; an Axis supplies its own units and
    raw wavelength arrays are metres. Stored samples are interpolated to a runtime
    bandpass. Without stored sampling, the value evaluates directly at the query.

    Cubic2 interpolation is the default for sampled shapes.
    Calling with wavelengths returns point values; a BandPass produces Weights.
    ``from_weights`` constructs Weights directly for already-integrated counts.
    """

    # Inherited fields
    bandpass: UnitSpec | Array | zdx.Expression | None = None
    flux: Array | zdx.Expression | Flux | None = None

    value: Array | zdx.Expression
    method: str

    def __init__(
        self,
        value=1.0,
        bandpass=None,
        flux=1.0,
        *,
        method="cubic2",
        wavelengths=None,
        unit="m",
        alias=None,
    ):
        if wavelengths is not None:
            if bandpass is not None:
                raise ValueError("Supply bandpass or wavelengths, not both.")
            bandpass = BandPass(wavelengths=wavelengths, unit=unit)
        elif unit != "m":
            raise ValueError("unit applies to wavelengths; a bandpass owns its unit.")
        super().__init__(bandpass, flux, alias=alias)
        self.value = zdx.as_array(value, dtype=float)
        self.method = str(method)

    @classmethod
    def from_weights(cls, bandpass, weights, *, alias=None):
        """Construct Weights for already-integrated counts; no density is implied."""
        return Weights(weights, bandpass, alias=alias)

    def sample(self, wavelengths=None, **context):
        """Evaluate the stored shape without integration or brightness scaling."""
        self = self.populate()
        return sample_curve(
            self.value, self.bandpass, wavelengths, self.method, **context
        )


class PolySpectrum(Spectrum):
    """A shape 1 + c[0] u + c[1] u² + ... on a fixed reference BandPass.

    The reference interval maps to u in [-1, 1], independently of the query range.
    Zero coefficients give a flat spectrum. Flux is the integrated brightness;
    coefficients change shape only. Values outside the reference interval are zero.
    """

    def __init__(
        self, bandpass, coeffs=None, flux=1.0, *, degree=1, method="cubic2", alias=None
    ):
        if not isinstance(bandpass, BandPass) or bandpass.min == bandpass.max:
            raise ValueError("PolySpectrum requires a finite reference BandPass.")
        if coeffs is not None and not isinstance(coeffs, zdx.Expression):
            coeffs = zdx.as_array(coeffs, dtype=float)
            if coeffs.ndim == 0:
                raise ValueError("coeffs must end in a polynomial coefficient axis.")
            degree = coeffs.shape[-1]
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer.")
        polynomial = Polynomial(degrees=tuple(range(1, degree + 1)), coeffs=coeffs)
        super().__init__(polynomial, bandpass, flux, method=method, alias=alias)

    def sample(self, wavelengths=None, **context):
        """Evaluate the polynomial directly, keeping its reference interval fixed."""
        self = self.populate()
        wavelengths = self.bandpass if wavelengths is None else wavelengths
        wavelengths = _wavelengths(wavelengths, **context)
        low, high = self.bandpass.bounds
        coordinates = 2 * (wavelengths - low) / (high - low) - 1
        values = 1 + _curve_values(self.value, coordinates, **context)
        return np.where((wavelengths >= low) & (wavelengths <= high), values, 0)


class Filter(zdx.Module):
    """A multiplicative response with optional native wavelength sampling.

    Arrays and fixed bases use ``bandpass`` to locate their native samples.
    Analytic expressions and scalars may omit it. Calling a Filter on Weights
    samples the response on that object's bandpass, retaining absolute losses.
    QE uses this same class. Wavelength units belong to the native sampling.
    Cubic2 interpolation is the default; use method="linear" for straight segments.
    """

    value: Array | zdx.Expression
    bandpass: UnitSpec | Array | zdx.Expression | None = None
    method: str

    def __init__(
        self,
        value=1.0,
        bandpass=None,
        *,
        wavelengths=None,
        method="cubic2",
        knots=None,
        unit="m",
        alias=None,
    ):
        # Translate the older knots spelling once; only native sampling is stored.
        if knots is not None:
            if bandpass is not None or wavelengths is not None:
                raise ValueError("Supply bandpass, wavelengths or knots separately.")
            if isinstance(knots, UnitSpec):
                if unit != "m":
                    raise ValueError("Sampling objects own their wavelength unit.")
                bandpass = knots
            else:
                bandpass = zdx.as_array(knots, dtype=float) * unit_factor(
                    unit, dimension="length"
                )
        elif wavelengths is not None:
            if bandpass is not None:
                raise ValueError("Supply bandpass or wavelengths, not both.")
            bandpass = BandPass(wavelengths=wavelengths, unit=unit)
        elif unit != "m":
            raise ValueError("unit applies to wavelengths; a bandpass owns its unit.")
        if isinstance(bandpass, UnitSpec):
            if bandpass.ndim != 1:
                raise ValueError("Spectral sampling must be one-dimensional.")
            canonical_unit(bandpass.unit, dimension="length")
        else:
            bandpass = zdx.as_array(bandpass, dtype=float)
        self.value = zdx.as_array(value, dtype=float)
        self.bandpass = bandpass
        self.method = str(method)
        self.alias = alias

    def sample(self, wavelengths=None, **context):
        """Sample the dimensionless response, using native knots when omitted."""
        self = self.populate()
        return sample_curve(
            self.value, self.bandpass, wavelengths, self.method, **context
        )

    def __call__(self, weights: Weights, **context) -> Weights:
        """Apply losses to integrated counts without changing their sampling."""
        if not isinstance(weights, Weights):
            raise TypeError("Filter expects Weights; use sample for response values.")
        self, weights = zdx.populate((self, weights))
        response = self.sample(weights.bandpass, **context)
        return weights * response

    def integrate(self, bandpass=None, **context) -> Array:
        """Integrate the response over a finite BandPass, returning metres.

        This is an effective bandwidth, not a photon count or mean transmission.
        Photometry integrates the response together with the actual source density.
        Omitted sampling uses the response's native BandPass.
        """
        self, bandpass = zdx.populate((self, bandpass))
        sampling = self.bandpass if bandpass is None else bandpass

        # A unit density and no flux normalisation leave only the response integral.
        reference = Spectrum(flux=None)
        result = integrate_spectrum(reference, sampling, filter=self, **context)
        return result.integrate()

    def prepare(self, bandpass: BandPass, **context) -> Filter:
        """Sample a fixed response once, returning a Filter on the chosen bandpass.

        This is an explicit snapshot. Keep the original definition when fitting its
        parameters, or prepare again after changing them.
        """
        self, bandpass = zdx.populate((self, bandpass))
        bandpass = bandpass.resolve(**context)
        return Filter(self.sample(bandpass, **context), bandpass, method=self.method)


def blackbody(wavelengths, temperature):
    """Planck photon radiance per second, m², steradian and metre of wavelength."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    temperature = np.asarray(temperature, dtype=float)
    temperature = temperature.reshape(temperature.shape + (1,) * wavelengths.ndim)
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    exponent = (h * c / k) / (wavelengths * temperature)
    log_denominator = exponent + np.log(-np.expm1(-exponent))
    return np.exp(np.log(2 * c) - 4 * np.log(wavelengths) - log_denominator)


class Blackbody(BaseSpectrum):
    """A Planck spectrum with temperature in kelvin and the ordinary flux contract.

    By default the spectrum contains one photon over the chosen BandPass.
    sample returns photon radiance; flux=None retains that absolute scale, which
    requires the source's apparent solid angle and observing factors separately.
    """

    # Inherited fields
    bandpass: UnitSpec | Array | zdx.Expression | None = None
    flux: Array | zdx.Expression | Flux | None = None

    temperature: Array | zdx.Expression

    def __init__(self, temperature, bandpass=None, flux=1.0, *, alias=None):
        super().__init__(bandpass, flux, alias=alias)
        self.temperature = zdx.as_array(temperature, dtype=float)
        if not isinstance(self.temperature, zdx.Expression):
            invalid = np.any(~np.isfinite(self.temperature) | (self.temperature <= 0))
            if not isinstance(invalid, core.Tracer) and bool(invalid):
                raise ValueError("temperature must be finite and positive.")

    def sample(self, wavelengths=None, **context):
        self = self.populate()
        sampling = self.bandpass if wavelengths is None else wavelengths
        if sampling is None:
            raise ValueError("Supply wavelengths or stored spectral sampling.")
        wavelengths = _wavelengths(sampling, **context)
        temperature = zdx.resolve(
            self.temperature, **dict(context, wavelengths=wavelengths)
        )
        return blackbody(wavelengths, temperature)
