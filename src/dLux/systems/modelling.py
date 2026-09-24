"""Shared source, spectral-weight and intensity calculations for optical systems.

Systems own their execution stages. These shared helpers prepare physical inputs,
traverse source collections and reduce point responses with photon counts.
"""

import jax.numpy as np
from jax import Array
import zodiax as zdx

import dLux.utils as dlu
from ..fields import Intensity
from .sources import Source, ResolvedSource, Scene
from ..utils.photometry import Weights


def source_profile(source, grid, optic=None, **context):
    """Sample one resolved profile on centred angular displacement coordinates."""
    if not isinstance(source, ResolvedSource):
        return None
    spacing = grid.d_si
    if dlu.unit_dimension(grid.unit) == "length":
        focal_length = None if optic is None else getattr(optic, "focal_length", None)
        if focal_length is None:
            raise ValueError(
                "A length-valued image needs focal_length to sample a resolved source."
            )
        spacing = spacing / zdx.resolve(focal_length, **context)
    # Odd support puts zero displacement at the centre of the convolution kernel,
    # independently of an even-sized or off-axis image grid.
    shape = tuple(n + (n % 2 == 0) for n in grid.n)
    angular_grid = dlu.Grid(n=shape, d=spacing, unit="rad")
    profile = source.profile.sample(angular_grid, **context)
    if profile.ndim not in (2, 3):
        raise ValueError(
            "A resolved profile must have shape (y, x) or (wavelength, y, x)."
        )
    return profile


def render_sources(
    system,
    stage,
    source,
    offset=None,
    stokes=None,
    **context,
):
    """Retain optical source structure, then add its incoherent intensities."""
    offset = np.zeros(2) if offset is None else offset
    if not isinstance(source, (Source, Scene, list, tuple)) and hasattr(
        source, "_as_source"
    ):
        source = source._as_source(**context)
    if isinstance(source, (Scene, list, tuple)):
        named = isinstance(source, Scene)
        if named:
            if stage != "photometry":
                position = np.asarray(zdx.resolve(source.position, **context))
                if position.shape != (2,):
                    raise ValueError("Scene position must resolve to shape (2,).")
                offset = offset + position
            children = source.sources
        else:
            children = dict(enumerate(source))
        values = {
            name: render_sources(system, stage, child, offset, stokes, **context)
            for name, child in children.items()
        }
        parts = list(values.values())
        if not parts:
            raise ValueError("A source collection cannot be empty.")
        if stage in ("wavefront", "psf", "photometry"):
            return values if named else parts
        if stage == "all":
            fields = {name: value["Wavefront"] for name, value in values.items()}
            intensities = [value["Intensity"] for value in parts]
            return {
                "Wavefront": fields if named else list(fields.values()),
                "Intensity": sum(intensities[1:], intensities[0]),
            }
        return sum(parts[1:], parts[0])

    return system._render_source(stage, source, offset, stokes, **context)


def spectral_weights(source, bandpass, filter=None, qe=None, **context):
    """Calculate component-by-wavelength masses independently of propagation."""
    sampling = source.spectrum.bandpass if bandpass is None else bandpass
    if not isinstance(sampling, dlu.BandPass):
        raise TypeError(
            "Photometry requires a BandPass with bin boundaries. "
            "Use wavelength arrays only for optical propagation."
        )
    positions = source.resolve_positions(
        **dict(context, wavelengths=sampling.wavelengths)
    )
    local = dict(context, positions=positions)
    model = dlu.Photometry(source.spectrum, filter=filter, qe=qe)

    # Check fixed curve shapes before integration. Photometry owns sampling and
    # binding; these checks enforce only the Source's one-shared-curve contract.
    model = model.resolve(**local)
    for response in (model.filter, model.qe):
        if response is not None and isinstance(response.value, Array):
            if response.value.ndim > 1:
                raise ValueError(
                    "Filter and detector QE must provide one shared wavelength curve."
                )
    if isinstance(model.spectrum, dlu.Spectrum):
        value = model.spectrum.value
        if isinstance(value, Array) and value.ndim > 1:
            raise ValueError(
                "A Source requires one shared spectrum; "
                "use a Scene for separate spectra."
            )

    result = model(sampling, **local)
    weights = result.value
    if weights.ndim > 2 or (
        weights.ndim == 2 and weights.shape[0] != positions.shape[0]
    ):
        raise ValueError("Source flux must resolve to a scalar or shape (C,).")
    weights = np.broadcast_to(weights, (positions.shape[0], sampling.n))
    return Weights(weights, sampling)


def integrate_psf(psf, weights):
    """Sum component-by-wavelength PSFs with their integrated photon counts."""
    wavelengths, values = weights.wavelengths, weights.value
    expected = psf.data.shape[:-2]
    if values.shape != expected or wavelengths.shape != psf.wavelengths.shape:
        raise ValueError(
            "Photometric weights must match the PSF source and wavelength axes."
        )
    data = (psf.data * values[..., None, None]).sum((0, 1))
    return Intensity(data, psf.grid)


def source_samples(source, bandpass, offset=None, **context):
    """Return wavelength points and radian positions without evaluating brightness."""
    sampling = source.spectrum.bandpass if bandpass is None else bandpass
    if sampling is None:
        raise ValueError("Supply bandpass or explicit spectral sampling.")
    if isinstance(sampling, dlu.UnitSpec):
        wavelengths = sampling.resolve(**context).axes[0]
    else:
        wavelengths = np.atleast_1d(np.asarray(zdx.resolve(sampling, **context)))
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("bandpass must provide a non-empty wavelength vector.")
    positions = source.resolve_positions(**dict(context, wavelengths=wavelengths))
    if offset is not None:
        positions = positions + offset
    return wavelengths, positions
