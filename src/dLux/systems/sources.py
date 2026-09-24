"""Declarative source definitions consumed by optical systems."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

__all__ = ["Source", "PointSource", "BinarySource", "ResolvedSource", "Scene"]


class Source(zdx.Module):
    """Incoherent point sources sharing one spectral shape.

    Positions have shape ``(C, 2)`` in radians. Flux is scalar or ``(C,)``;
    a scalar supplies the same incident photon count to every position.
    Brightness lives on spectrum.flux and refers to the current BandPass before
    filter and QE losses. The flux constructor argument sets that leaf.
    Spectrum(flux=None) retains absolute photon density.

    Sources store definitions. Imager owns spectral integration, propagation,
    resolved-profile convolution and detection.
    """

    spectrum: dlu.BaseSpectrum
    positions: Array | zdx.Expression

    def __init__(
        self,
        spectrum=1.0,
        positions=None,
        flux=None,
        *,
        wavelengths=None,
        weights=None,
        alias=None,
    ):
        """Store a spectrum, radian positions and optional incident photon counts.

        Raw spectral arrays or Expressions become Spectrum definitions. Omitting
        positions gives one on-axis source; ``(2,)`` promotes to ``(1, 2)``.
        A supplied Spectrum or Blackbody retains its flux unless overridden.
        Raw spectral shapes default to one photon per position. Positions and
        flux may be Expressions.
        Absolute density is declared by Spectrum(flux=None); Source stores no
        separate normalisation flag or brightness leaf.
        """
        if wavelengths is not None or weights is not None:
            from ..compat.sources import source_spectrum

            spectrum = source_spectrum(spectrum, wavelengths, weights)

        # Wrap raw spectral shapes and apply an explicit flux override to that spectrum.
        if isinstance(spectrum, dlu.Weights):
            raise TypeError(
                "Source requires a spectral density, not integrated Weights. "
                "Weights are produced by photometry."
            )
        spectrum = (
            spectrum
            if isinstance(spectrum, dlu.BaseSpectrum)
            else dlu.Spectrum(spectrum)
        )
        if flux is not None:
            value = (
                flux if isinstance(flux, dlu.Flux) else zdx.as_array(flux, dtype=float)
            )
            spectrum = spectrum.set(flux=value)
        self.spectrum = spectrum

        # Store positions as (components, 2), defaulting to one on-axis point.
        positions = np.zeros((1, 2)) if positions is None else positions
        positions = zdx.as_array(positions, dtype=float)
        if not isinstance(positions, zdx.Expression):
            if positions.shape == (2,):
                positions = positions[None, :]
            if positions.ndim != 2 or positions.shape[-1] != 2:
                raise ValueError("positions must have shape (C, 2).")
            if positions.shape[0] == 0:
                raise ValueError("A Source needs at least one position.")
        self.positions = positions

        # Known brightness arrays must be shared scalars or match the position count.
        flux = (
            spectrum.flux.value
            if isinstance(spectrum.flux, dlu.Flux)
            else spectrum.flux
        )
        if flux is not None and not isinstance(flux, zdx.Expression):
            if flux.ndim > 1:
                raise ValueError("flux must be scalar or have shape (C,).")
            if flux.ndim == 1 and not isinstance(positions, zdx.Expression):
                if flux.shape != positions.shape[:1]:
                    raise ValueError("flux must have one value per source position.")
        self.alias = alias

    def resolve_positions(self, **context) -> Array:
        """Resolve only astrometry, returning non-empty ``(C, 2)`` radians."""
        positions = np.atleast_2d(zdx.resolve(self.positions, **context))
        if positions.ndim != 2 or positions.shape[-1] != 2 or positions.shape[0] == 0:
            raise ValueError("Source positions must have non-empty shape (C, 2).")
        return positions

    def model(self, optics, return_wf=False, return_psf=False):
        """Deprecated forwarding method; optical systems own source modelling."""
        from ..compat.modelling import model_source
        from ..compat.warnings import warn_deprecated

        warn_deprecated(
            "source.model(optics)",
            "imager.model(source)",
            "Call the system's modelling method",
        )
        return model_source(optics, self, return_wf, return_psf)


class PointSource(Source):
    """One point source with a spectrum, position and incident photon flux."""

    # Inherited fields
    spectrum: dlu.BaseSpectrum
    positions: Array | zdx.Expression

    def __init__(
        self,
        *args,
        spectrum=None,
        position=None,
        flux=None,
        unit="arcsec",
        wavelengths=None,
        weights=None,
        alias=None,
    ):
        """Store one position in unit (arcseconds by default), or the optical axis.

        Flux follows the spectrum's current-BandPass convention. A spectrum with
        flux=None supplies its own absolute photon density.
        """
        if args or wavelengths is not None or weights is not None:
            from ..compat.sources import point_spectrum

            spectrum = point_spectrum(args, spectrum, wavelengths, weights)
        spectrum = 1.0 if spectrum is None else spectrum
        super().__init__(spectrum, dlu.Point(position, unit=unit), flux, alias=alias)


class BinarySource(Source):
    """Two point sources sharing a spectrum, parameterised about the primary.

    The primary receives ``primary_flux`` incident photons and the companion
    receives ``primary_flux * 10**contrast`` by default (log=False uses a ratio).
    Positions and fluxes are ordinary Expressions, so Imager handles this class
    exactly like Source.
    """

    # Inherited fields
    spectrum: dlu.BaseSpectrum
    positions: Array | zdx.Expression

    def __init__(
        self,
        spectrum=1.0,
        position=None,
        separation=0.0,
        pa=0.0,
        primary_flux=None,
        contrast=None,
        *,
        unit="arcsec",
        log=True,
        wavelengths=None,
        weights=None,
        mean_flux=None,
        position_angle=None,
        alias=None,
    ):
        """Store binary astrometry and brightness with one normalised spectrum.

        ``position`` is the primary's ``(x, y)`` position, defaulting to on-axis.
        Position and separation use unit (arcseconds by default). PA is always in
        degrees, from ``+y`` towards ``+x``.
        Omitted primary_flux uses the supplied spectrum's flux (one by default).
        Flux is primary incident photons over the current BandPass and contrast is
        log10(companion/primary); log=False accepts a linear ratio.
        Omitted contrast gives equal brightness. Numerical inputs may be Expressions.
        """
        if any(
            value is not None
            for value in (wavelengths, weights, mean_flux, position_angle)
        ):
            from ..compat.sources import binary_source_inputs

            spectrum, positions, flux = binary_source_inputs(
                spectrum,
                position,
                separation,
                1.0 if contrast is None else contrast,
                wavelengths,
                weights,
                mean_flux,
                position_angle,
            )
            super().__init__(spectrum, positions, flux, alias=alias)
            return

        positions = dlu.Binary(position, separation, pa, unit=unit)
        super().__init__(spectrum, positions, primary_flux, alias=alias)

        # The supplied spectrum owns primary brightness unless explicitly replaced.
        primary = self.spectrum.flux
        if primary is None:
            raise ValueError(
                "BinarySource requires primary_flux for an absolute-density spectrum. "
                "Use a Scene for independently defined absolute spectra."
            )
        if isinstance(primary, dlu.Flux):
            flux = dlu.Flux(
                dlu.ContrastFlux(primary.value, contrast, log=log), primary.unit
            )
        else:
            flux = dlu.ContrastFlux(primary, contrast, log=log)
        self.spectrum = self.spectrum.set(flux=flux)


class ResolvedSource(Source):
    """A source whose local morphology is described by a sampled Profile.

    The profile owns its angular sampling and normalisation. Spectrum.flux supplies
    brightness, and positions locate the profile centres. Imager samples this
    definition and renders it through the optical model.
    """

    # Inherited fields
    spectrum: dlu.BaseSpectrum
    positions: Array | zdx.Expression

    profile: dlu.Profile

    def __init__(
        self,
        spectrum=1.0,
        positions=None,
        flux=None,
        *,
        profile=None,
        profile_grid=None,
        wavelengths=None,
        weights=None,
        distribution=None,
        position=None,
        alias=None,
    ):
        """Store a Profile, or construct one from a definition and angular grid.

        A coordinate-dependent Expression can wait for sampling from Imager.
        Arrays require a grid; a Field or Profile already owns its sampling.
        Local profile coordinates are relative to each source position.
        """
        if distribution is not None or position is not None:
            from ..compat.sources import resolved_inputs

            positions, profile = resolved_inputs(
                positions, profile, profile_grid, distribution, position
            )
        if profile is None:
            raise ValueError("Supply a sampled profile.")
        super().__init__(
            spectrum,
            positions,
            flux,
            wavelengths=wavelengths,
            weights=weights,
            alias=alias,
        )
        if isinstance(profile, dlu.Profile):
            if profile_grid is not None:
                raise ValueError("A Profile already owns its sampling grid.")
        else:
            profile = dlu.Profile(profile, grid=profile_grid)

        grid = profile.grid
        if grid is not None:
            if grid.unit is not None:
                dlu.canonical_unit(grid.unit, dimension="angle")
        self.profile = profile


class Scene(zdx.Module):
    """Named independent sources with a shared positional offset.

    Nested Scenes can group a companion and its disk, for example. Imager adds
    source intensities and applies detector noise once to the combined image.
    Each child retains its own spectrum, brightness and optional profile.
    """

    sources: dict[str, Source | Scene]
    position: Array | zdx.Expression

    def __init__(
        self, sources: Mapping[str, Source | Scene], position=None, *, alias=None
    ):
        """Store a non-empty mapping and a shared ``(x, y)`` radian offset.

        Values are Source or Scene objects. Position defaults to zero and may be
        an Expression. Names provide ordinary immutable parameter paths.
        """
        # Check named sources before storing the scene and its shared offset.
        if isinstance(sources, (list, tuple)):
            from ..compat.warnings import warn_deprecated

            warn_deprecated(
                "Scene(sequence)", "Scene(mapping)", "Scene({'name': source})"
            )
            sources = dlu.list2dictionary(
                sources, ordered=False, allowed_types=(Source, Scene)
            )
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError("sources must be a non-empty named mapping.")
        for name, source in sources.items():
            if not isinstance(name, str):
                raise TypeError("Scene source names must be strings.")
            if not isinstance(source, (Source, Scene)):
                raise TypeError("Scene entries must be Source or Scene objects.")
        self.sources = dict(sources)
        position = np.zeros(2) if position is None else position
        self.position = zdx.as_array(position, dtype=float)
        if not isinstance(self.position, zdx.Expression):
            if self.position.shape != (2,):
                raise ValueError("Scene position must have shape (2,).")
        self.alias = alias

    def model(self, optics, return_wf=False, return_psf=False):
        """Forward released Scene modelling to the system's current operations."""
        from ..compat.modelling import model_source

        return model_source(optics, self, return_wf, return_psf)
