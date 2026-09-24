"""Released source constructors returning current declarative sources."""

import jax.numpy as np
import dLux.utils as dlu
from ..systems import sources as current
from .spectra import Spectrum
from .warnings import warn_deprecated

BaseSource = current.Source
Scene = current.Scene
__all__ = [
    "BaseSource",
    "Source",
    "PointSource",
    "PointSources",
    "BinarySource",
    "ResolvedSource",
    "PointResolvedSource",
    "Scene",
]


def spectral_input(wavelengths, weights, spectrum):
    if wavelengths is not None:
        if spectrum is not None:
            raise ValueError("Supply wavelengths/weights or spectrum, not both.")
        if weights is not None and np.asarray(weights).ndim > 1:
            raise ValueError(
                "Per-component spectra require a Scene of PointSource objects, each "
                "with its own spectrum."
            )
        return Spectrum(wavelengths, weights)
    if weights is not None:
        raise ValueError("weights requires wavelengths.")
    if spectrum is None:
        raise ValueError(
            "Released source construction requires wavelengths or spectrum."
        )
    if not isinstance(spectrum, dlu.BaseSpectrum):
        raise TypeError("spectrum must be a current BaseSpectrum.")
    return spectrum


def source_spectrum(spectrum, wavelengths, weights):
    """Translate old Source keyword inputs before canonical construction."""
    if not isinstance(spectrum, (int, float)) or spectrum != 1.0:
        raise ValueError("Supply spectrum or wavelengths/weights, not both.")
    return spectral_input(wavelengths, weights, None)


def point_spectrum(args, spectrum, wavelengths, weights):
    """Disambiguate released positional wavelengths and current spectrum objects."""
    if args:
        if (
            len(args) != 1
            or not isinstance(args[0], dlu.BaseSpectrum)
            or spectrum is not None
        ):
            raise TypeError(
                "Use wavelengths= for released samples or spectrum= for a "
                "spectral definition; positional arrays are ambiguous."
            )
        spectrum = args[0]
    if wavelengths is not None or weights is not None:
        spectrum = spectral_input(wavelengths, weights, spectrum)
    return spectrum


def binary_source_inputs(
    spectrum,
    position,
    separation,
    contrast,
    wavelengths,
    weights,
    mean_flux,
    position_angle,
):
    """Translate released midpoint/mean-flux arguments at the constructor boundary."""
    warn_deprecated(
        "released BinarySource arguments",
        "BinarySource",
        "Use primary_flux and pa; old midpoint/mean-flux inputs are converted once",
    )
    if isinstance(spectrum, (int, float)) and spectrum == 1.0:
        spectrum = None
    return binary_inputs(
        wavelengths,
        position,
        1.0 if mean_flux is None else mean_flux,
        separation,
        np.pi / 2 if position_angle is None else position_angle,
        contrast,
        spectrum,
        weights,
    )


def resolved_inputs(positions, profile, profile_grid, distribution, position):
    """Translate released profile/position names without inventing sampling."""
    if distribution is not None or position is not None:
        warn_deprecated(
            "ResolvedSource distribution/position",
            "ResolvedSource",
            "ResolvedSource(profile=..., profile_grid=..., positions=...)",
        )
    if distribution is not None:
        if profile is not None or profile_grid is None:
            raise ValueError(
                "Released distribution requires profile_grid and no separate profile."
            )
        profile = distribution
    if position is not None:
        if positions is not None:
            raise ValueError("Supply position or positions, not both.")
        positions = position
    return positions, profile


def Source(wavelengths=None, weights=None, spectrum=None):
    warn_deprecated("compat.Source", "Source", "Source(spectrum=...)")
    return current.Source(spectrum=spectral_input(wavelengths, weights, spectrum))


def PointSource(wavelengths=None, position=None, flux=1.0, weights=None, spectrum=None):
    warn_deprecated(
        "compat.PointSource",
        "PointSource",
        "PointSource(spectrum=..., unit='rad')",
    )
    return current.PointSource(
        spectrum=spectral_input(wavelengths, weights, spectrum),
        position=position,
        unit="rad",
        flux=flux,
    )


def PointSources(
    wavelengths=None, position=None, flux=None, weights=None, spectrum=None
):
    warn_deprecated("PointSources", "Source", "Source(positions=..., spectrum=...)")
    return current.Source(
        spectrum=spectral_input(wavelengths, weights, spectrum),
        positions=position,
        flux=flux,
    )


def binary_inputs(
    wavelengths,
    position,
    mean_flux,
    separation,
    position_angle,
    contrast,
    spectrum,
    weights,
):
    from ..utils.compat.source import positions_from_sep, fluxes_from_contrast

    position = np.zeros(2) if position is None else np.asarray(position)
    positions = positions_from_sep(position, separation, position_angle)
    fluxes = fluxes_from_contrast(mean_flux, contrast)
    return spectral_input(wavelengths, weights, spectrum), positions, fluxes


def BinarySource(
    wavelengths=None,
    position=None,
    mean_flux=1.0,
    separation=0.0,
    position_angle=np.pi / 2,
    contrast=1.0,
    spectrum=None,
    weights=None,
):
    warn_deprecated(
        "released BinarySource",
        "Source",
        "Source(positions=..., flux=...); midpoint/mean-flux values are converted once",
    )
    spectrum, positions, flux = binary_inputs(
        wavelengths,
        position,
        mean_flux,
        separation,
        position_angle,
        contrast,
        spectrum,
        weights,
    )
    return current.Source(spectrum=spectrum, positions=positions, flux=flux)


def ResolvedSource(
    wavelengths=None,
    position=None,
    flux=1.0,
    distribution=None,
    weights=None,
    spectrum=None,
    *,
    profile_grid=None,
):
    warn_deprecated(
        "compat.ResolvedSource",
        "ResolvedSource",
        "ResolvedSource(profile=Profile(...))",
    )
    if profile_grid is None:
        raise ValueError(
            "distribution has no angular sampling. Supply profile_grid, or construct "
            "ResolvedSource(profile=Profile(...))."
        )
    return current.ResolvedSource(
        spectrum=spectral_input(wavelengths, weights, spectrum),
        positions=position,
        flux=flux,
        profile=distribution,
        profile_grid=profile_grid,
    )


def PointResolvedSource(*args, **kwargs):
    raise TypeError(
        "Use a Scene containing PointSource and ResolvedSource with explicit profile "
        "sampling and component fluxes."
    )
