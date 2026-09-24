"""Shared integration of spectral densities into wavelength bins.

Spectrum and Photometry use the same numerical route. Curves supply sample and
bandpass; integration returns Weights and never constructs a spectral definition.
"""

from math import sqrt

import interpax
import jax
import jax.numpy as np
import zodiax as zdx
from jax import Array

from .bandpass import BandPass, _wavelengths
from .flux import Flux
from .weights import Weights

__all__ = ["integrate_curve", "integrate_spectrum", "bin_spectrum"]


def integrate_curve(values, knots, edges, method="cubic2") -> Array:
    """Integrate a sampled density into metre bins, with zero outside support.

    Linear interpolation gives trapezoidal integrals. Cubic methods reuse
    Interpax's knot derivatives and integrate the resulting Hermite polynomial.
    This integrates the represented curve, not unresolved spectral features.
    The wavelength axis is last; earlier value axes are retained.
    """
    knots, edges = np.asarray(knots), np.asarray(edges)
    values = np.asarray(values)
    if knots.size < 2 or values.shape[-1] != knots.size:
        raise ValueError("Integration needs at least two matching wavelength samples.")

    # Work on a dimensionless interval to avoid large inverse-metre spline terms.
    width = knots[-1] - knots[0]
    x = (knots - knots[0]) / width
    bounds = np.clip((edges - knots[0]) / width, 0, 1)
    values = np.moveaxis(values, -1, 0)
    if method == "linear":
        spacing = np.diff(x).reshape((-1,) + (1,) * (values.ndim - 1))
        slopes = np.diff(values, axis=0) / spacing
        curve = interpax.PPoly(np.stack((slopes, values[:-1])), x, check=False)
    else:
        interpolator = interpax.Interpolator1D(x, values, method=method)
        curve = interpax.CubicHermiteSpline(
            x, values, interpolator.derivs["fx"], check=False
        )
    integrals = jax.vmap(curve.integrate)(bounds[:-1], bounds[1:]) * width
    return np.moveaxis(integrals, 0, -1)


def integrate_spectrum(
    spectrum, bandpass, *, filter=None, qe=None, **context
) -> Weights:
    """Integrate spectrum times filter/QE onto a fixed calculation BandPass.

    Native nodes and support boundaries provide the fine integration mesh.
    Unbound array tables refer to the calculation centres. Products are integrated
    within each interval, preserving each curve's interpolation and support.
    Analytic shapes require sampling fine enough to resolve their features.
    """
    if not isinstance(bandpass, BandPass):
        raise TypeError("Integration requires a BandPass defining bin boundaries.")
    if isinstance(spectrum, Weights):
        raise TypeError("Integrated Weights cannot be reinterpreted as a density.")

    # Fixed definitions (for example Map or Basis) become tables before binding
    # sampling. Wavelength-dependent definitions remain deferred for the fine mesh.
    spectrum, filter, qe = zdx.resolve((spectrum, filter, qe), **context)

    # A monochromatic point has no density integral. Its explicit flux assigns
    # counts to that single wavelength before filter and QE losses are applied.
    if bandpass.d is None:
        if spectrum.flux is None:
            raise ValueError("Absolute density integration requires a bandwidth.")
        wavelengths = bandpass.wavelengths
        values = spectrum.sample(bandpass, **context)
        flux = spectrum.flux
        if not isinstance(flux, Flux):
            flux = Flux(flux)
        counts = flux.photons(wavelengths, values, **context)
        for response in (filter, qe):
            if response is not None:
                counts = counts * response.sample(bandpass, **context)
        return Weights(counts, bandpass)

    # Bare tables refer to the calculation nodes. Bind them before sampling on
    # the finer mesh supplied by the other curves. Analytic curves need no binding.
    curves = []
    for curve in (spectrum, filter, qe):
        if curve is not None and curve.bandpass is None:
            value = getattr(curve, "value", None)
            if isinstance(value, Array) and value.ndim > 0:
                curve = curve.set(bandpass=bandpass)
        curves.append(curve)
    spectrum, filter, qe = curves

    # Curve interpolation owns the shape. Integration only selects enough interior
    # samples to integrate products of those linear or cubic pieces accurately.
    method = "linear"
    for curve in curves:
        if curve is not None and getattr(curve, "method", "linear") != "linear":
            method = "cubic2"
    result = bin_spectrum(
        spectrum,
        bandpass.edges,
        bandpass.edges,
        filter=filter,
        qe=qe,
        method=method,
        **context,
    )
    return Weights(result.value, bandpass)


def bin_spectrum(
    spectrum,
    edges,
    bandpass=None,
    *,
    method="cubic2",
    filter=None,
    qe=None,
    **context,
) -> Weights:
    """Integrate density times responses into explicit metre bins.

    The supplied mesh bounds the integration domain. Native knots, support edges
    and output edges split it into intervals; each curve retains its interpolation
    method. Interior quadrature samples never join a supported endpoint to a zero
    outside support. Two-point Gauss integration (linear) is exact for products of
    up to three linear curves; five-point integration (cubic/cubic2) handles three
    cubic curves. Analytic spectra are approximated at the same sample points.
    Incident flux is assigned over the requested bins before filter/QE losses.
    """
    if isinstance(spectrum, Weights):
        raise TypeError("Binning requires a density, not integrated samples.")
    sampling = spectrum.bandpass if bandpass is None else bandpass
    wavelengths = _wavelengths(sampling, **context)
    edges = _wavelengths(edges, **context)
    if wavelengths.size < 2 or edges.size < 2:
        raise ValueError("Integration needs at least two mesh points and bin edges.")
    if method not in ("linear", "cubic", "cubic2"):
        raise ValueError("Integration method must be linear, cubic or cubic2.")

    # Fixed unbound tables refer to the supplied mesh. Every native knot and support
    # edge must be an interval boundary, even when it lies between optical samples.
    curves = zdx.resolve((spectrum, filter, qe), **context)
    prepared, points = [], [wavelengths, edges]
    for curve in curves:
        if curve is not None:
            if curve.bandpass is None:
                value = getattr(curve, "value", None)
                if isinstance(value, Array) and value.ndim > 0:
                    curve = curve.set(bandpass=wavelengths)
            if curve.bandpass is not None:
                points.append(_wavelengths(curve.bandpass, **context))
                if isinstance(curve.bandpass, BandPass):
                    points.append(np.asarray(curve.bandpass.bounds))
        prepared.append(curve)
    spectrum, filter, qe = prepared
    low, high = wavelengths[0], wavelengths[-1]
    knots = np.sort(np.clip(np.concatenate(points), low, high))

    # Standard Gauss-Legendre nodes and weights on [-1, 1]. Repeated knots are
    # harmless zero-width intervals, so mesh construction has a fixed shape in JIT.
    if method == "linear":
        nodes = np.asarray([-1 / sqrt(3), 1 / sqrt(3)])
        weights = np.ones(2)
    else:
        inner = sqrt(5 - 2 * sqrt(10 / 7)) / 3
        outer = sqrt(5 + 2 * sqrt(10 / 7)) / 3
        a, b = (322 - 13 * sqrt(70)) / 900, (322 + 13 * sqrt(70)) / 900
        nodes = np.asarray([-outer, -inner, 0, inner, outer])
        weights = np.asarray([a, b, 128 / 225, b, a])
    half_width = np.diff(knots) / 2
    centres = (knots[:-1] + knots[1:]) / 2
    query = (centres[:, None] + half_width[:, None] * nodes).reshape(-1)

    # Evaluate the original interpolants, not a new spline through their product.
    # Rescaling avoids overflow for large densities and cancels in flux assignment.
    density = spectrum.sample(query, **context)
    if spectrum.flux is not None:
        scale = jax.lax.stop_gradient(np.max(np.abs(density), axis=-1, keepdims=True))
        density = density / np.where(scale > 0, scale, 1)
    transmitted = density
    for response in (filter, qe):
        if response is not None:
            transmitted = transmitted * response.sample(query, **context)

    # Accumulate interval masses and read cumulative values at the bin boundaries.
    # Boundary insertion above makes this indexing exact, without interpolation.
    indices = np.searchsorted(knots, np.clip(edges, low, high))

    def integrate(values):
        values = values.reshape(values.shape[:-1] + (half_width.size, nodes.size))
        masses = (values * weights).sum(-1) * half_width
        cumulative = np.concatenate(
            (np.zeros_like(masses[..., :1]), masses.cumsum(-1)), -1
        )
        return np.diff(np.take(cumulative, indices, axis=-1), axis=-1)

    centres = (edges[:-1] + edges[1:]) / 2
    detected = integrate(transmitted)
    if spectrum.flux is None:
        return Weights(detected, centres)

    incident = integrate(density)
    flux = spectrum.flux
    if not isinstance(flux, Flux):
        flux = Flux(flux)
    counts = flux.photons(centres, incident, **context)
    if filter is None and qe is None:
        return Weights(counts, centres)

    total = incident.sum(-1, keepdims=True)
    scale = counts.sum(-1, keepdims=True) / np.where(total != 0, total, 1)
    return Weights(detected * scale, centres)
