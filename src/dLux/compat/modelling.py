"""Small released return-contract adapters around current fields and spectra."""

import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu
from ..utils.photometry import BaseSpectrum
from ..fields import PSF
from ..systems.sources import Scene, ResolvedSource, PointSource


def result(field, weights=None, *, return_wf=False, return_psf=False):
    if return_wf and return_psf:
        raise ValueError("Select return_wf or return_psf, not both.")
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != field.wavelength.shape:
            raise ValueError("weights must have the same shape as wavelengths.")
    if return_wf:
        if weights is not None:
            field = field.set(phasor=field.phasor * np.sqrt(weights)[..., None, None])
        return field
    data = field.intensity
    if weights is not None:
        data = np.sum(data * weights[..., None, None], axis=0)
    if return_psf:
        return PSF(data, field.grid)
    return data


def model_source(system, source, return_wf=False, return_psf=False, offset=None):
    if return_wf and return_psf:
        raise ValueError("Select return_wf or return_psf, not both.")
    # Current Imager owns the complete source/profile/photometric pipeline.
    from ..systems.imagers import Imager

    if isinstance(system, Imager):
        if return_wf:
            if isinstance(source, (Scene, ResolvedSource)):
                raise TypeError(
                    "Request component wavefronts directly from Imager for Scenes or "
                    "resolved profiles."
                )
            field = system.wavefront(source)
            _, weights = system.photometry(source).weights()
            field = field.set(phasor=field.phasor * np.sqrt(weights)[..., None, None])
            # Imager keeps a source axis. A released PointSource returned only
            # wavelength axes; remove just that known singleton, not arbitrary batches.
            if isinstance(source, PointSource):
                field = field.set(phasor=field.phasor[0])
            return field
        image = system.intensity(source)
        return PSF(image.data, image.grid) if return_psf else image.data

    # Sampled OpticalSystem now owns the same canonical source-rendering route.
    if offset is not None:
        source = Scene({"source": source}, position=offset)
    return system.model(source, return_wf=return_wf, return_psf=return_psf)


def _sample_source(source, bandpass=None, offset=None, **context):
    """Return wavelengths, positions and incident photon weights shaped (C, L).

    A Source shares one spectrum across its point positions. Flux scales that
    spectrum once here; subsequent filters and QE only remove photons.
    """
    # Sample the shared spectral shape before applying source brightness.
    spectrum = source.spectrum
    if not isinstance(spectrum, BaseSpectrum):
        raise TypeError("Source.spectrum must be a Spectrum or Blackbody.")
    spectrum = spectrum.populate()
    sampling = spectrum.bandpass if bandpass is None else bandpass
    sampling = zdx.resolve(sampling, **context)
    if sampling is None:
        raise ValueError("Supply wavelength sampling for the source.")
    if isinstance(sampling, dlu.UnitSpec):
        sampling = sampling.axes[0]
    wavelengths = np.atleast_1d(np.asarray(sampling, dtype=float))
    from .spectra import _IntegratedSpectrum

    if isinstance(spectrum, _IntegratedSpectrum):
        if bandpass is not None:
            raise ValueError("Integrated samples already define their sampling.")
        weights = spectrum.value
    else:
        weights = spectrum.sample(wavelengths, **context)
        if spectrum.flux is None:
            raise ValueError("Absolute spectral densities require a BandPass.")
    if weights.ndim != 1:
        raise ValueError(
            "A Source requires one shared spectrum; use a Scene for separate spectra."
        )

    # Resolve positions and convert scalar or per-component flux into photon weights.
    local = dict(context, wavelengths=wavelengths)
    positions = np.atleast_2d(np.asarray(zdx.resolve(source.positions, **local)))
    if spectrum.flux is not None:
        flux = (
            spectrum.flux
            if isinstance(spectrum.flux, dlu.Flux)
            else dlu.Flux(spectrum.flux)
        )
        weights = flux.photons(
            wavelengths, weights, **dict(context, positions=positions)
        )

    # Broadcast the resulting weights to an explicit component-by-wavelength array.
    if weights.ndim > 2 or (
        weights.ndim == 2 and weights.shape[0] != positions.shape[0]
    ):
        raise ValueError("Source flux must resolve to a scalar or shape (C,).")
    weights = np.broadcast_to(weights, (positions.shape[0], wavelengths.size))
    if offset is not None:
        positions = positions + offset
    return wavelengths, positions, weights


def spectral_weights(source, bandpass=None, filter=None, qe=None, **context):
    """Apply released relative point weights at compatibility system boundaries."""
    wavelengths, _, weights = _sample_source(source, bandpass, **context)
    result = dlu.Weights(weights, wavelengths)
    for response in (filter, qe):
        if response is not None:
            response = (
                response if isinstance(response, dlu.Filter) else dlu.Filter(response)
            )
            result = response(result, **context)
    return result
