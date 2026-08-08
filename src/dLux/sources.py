"""Spectral and spatial source models."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.scipy as jsp
from jax import Array, vmap

import dLux.utils as dlu

from .fields import Wavefront
from .parametric import Parametric, ParametricHolder, resolve

__all__ = ["BaseSource", "Spectrum", "Source", "BinarySource"]

_DEFAULT_UNITS = {
    "wavelengths": "m",
    "position": "rad",
    "flux": "photon",
    "distribution": "linear",
}


def __getattr__(name):
    """Resolve source names retained by the compatibility layer."""
    legacy = {
        "PointResolvedSource",
        "PointSource",
        "PointSources",
        "ResolvedSource",
        "Scene",
    }
    if name in legacy:
        from . import compatibility

        return getattr(compatibility, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _merge_units(units=None):
    """Merge source unit overrides with canonical defaults."""
    units = {} if units is None else dict(units)
    unknown = set(units) - set(_DEFAULT_UNITS)
    if unknown:
        raise ValueError(f"Unknown source unit keys: {sorted(unknown)}.")
    return {**_DEFAULT_UNITS, **units}


def _convert_flux(flux, unit):
    """Convert linear or logarithmic flux into canonical units."""
    unit = str(unit).strip()
    if unit == "log":
        return 10**flux
    if unit == "ln":
        return np.exp(flux)
    try:
        factor = dlu.unit_factor(unit)
    except ValueError as error:
        raise ValueError(
            "Flux unit must be 'photon', a supported prefixed photon unit, "
            "'log', or 'ln'. See TODO: add units documentation link."
        ) from error
    return flux * factor


class BaseSource(ParametricHolder):
    """Source brightness and optional resolved distribution."""

    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(self, flux=None, distribution=None, units=None):
        self.flux = dlu.to_value(flux, optional=True, types=Parametric)
        self.distribution = dlu.to_value(distribution, optional=True, types=Parametric)
        self.units = _merge_units(units)

    def source_params(self, nsource=None, **context):
        """Resolve flux and distribution in canonical source units."""
        flux = self.flux_params(nsource, **context)
        distribution = self.distribution_params(nsource, **context)
        return flux, distribution

    def flux_params(self, nsource=None, **context):
        """Resolve and validate flux in canonical source units."""
        flux = resolve(self.flux, float, source=self, **context)
        flux = np.asarray(1.0 if flux is None else flux, dtype=float)
        if nsource is None:
            if flux.ndim != 0:
                raise ValueError("Single-source flux must be scalar.")
        else:
            if flux.ndim == 0:
                flux = np.broadcast_to(flux, (nsource,))
            if flux.shape != (nsource,):
                raise ValueError("Vectorised flux must have shape (nsource,).")
        return _convert_flux(flux, self.units["flux"])

    def distribution_params(self, nsource, **context):
        """Resolve and validate optional per-source spatial distributions."""
        distribution = resolve(self.distribution, float, source=self, **context)
        if distribution is None:
            return None
        distribution = np.asarray(distribution, dtype=float)
        valid = distribution.ndim == 2 or (
            nsource is not None
            and distribution.ndim == 3
            and distribution.shape[0] == nsource
        )
        if not valid:
            raise ValueError("distribution must have shape (y, x) or (nsource, y, x).")

        # Convert the resolved distribution into canonical linear units
        unit = str(self.units["distribution"]).strip()
        if unit == "linear":
            return distribution
        if unit == "log":
            return 10**distribution
        if unit == "ln":
            return np.exp(distribution)
        return _convert_flux(distribution, unit)

    @staticmethod
    def _convolve(data, distribution):
        """Convolve image data with shared or component distributions."""
        # Convolve a single image directly
        if data.ndim == 2:
            return jsp.signal.convolve(data, distribution, mode="same")

        # Broadcast distributions over leading image dimensions
        leading = data.shape[:-2]
        if distribution.ndim == 2:
            distribution = np.broadcast_to(distribution, leading + distribution.shape)
        else:
            extra = len(leading) - 1
            nsource = distribution.shape[0]
            shape = (nsource,) + (1,) * extra + distribution.shape[-2:]
            distribution = distribution.reshape(shape)
            distribution = np.broadcast_to(distribution, leading + shape[-2:])

        # Flatten and convolve every image-kernel pair
        shape = data.shape
        images = data.reshape((-1,) + shape[-2:])
        kernels = distribution.reshape((-1,) + distribution.shape[-2:])

        @vmap
        def convolve(image, kernel):
            return jsp.signal.convolve(image, kernel, mode="same")

        convolved = convolve(images, kernels)

        # Restore the original image shape
        return convolved.reshape(shape)

    def _propagate(self, optics, params):
        """Propagate one or more spatial source components."""
        # Unpack the resolved source parameters
        wavelengths = params["wavelengths"]
        weights = params["weights"]
        position = params["position"]
        flux = params["flux"]

        # Propagate a single spatial source component
        if position.ndim == 1:
            if weights.ndim != 1:
                raise ValueError(
                    "Single-component source weights must be one-dimensional."
                )
            return optics.propagate(
                wavelengths, position, weights * flux, return_all=True
            )

        # Align shared or component-dependent spectral weights
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, position.shape[:-1] + weights.shape)
        elif weights.shape[:-1] != position.shape[:-1]:
            raise ValueError(
                "Vectorised weights leading shape must match source positions."
            )

        # Vectorise propagation over spatial source components
        def propagate(component_position, component_flux, component_weights):
            return optics.propagate(
                wavelengths,
                component_position,
                component_weights * component_flux,
                return_all=True,
            )

        propagate = eqx.filter_vmap(propagate)
        return propagate(position, flux, weights)

    def wavefront(self, grid):
        """Create flux-weighted point-source wavefronts on an input grid.

        Resolved distributions remain an image-plane operation in ``model``. A
        vectorised source such as ``BinarySource`` returns one wavefront per component.
        """
        # Resolve the source parameters
        params = self.params()
        wavelengths = params["wavelengths"]
        position = params["position"]
        flux = params["flux"]
        weights = params["weights"]

        # Define initialisation of one weighted source component
        def initialise(pos, component_flux, component_weights):
            wavefront = Wavefront(wavelengths, grid).normalise().tilt(pos)
            weight = np.sqrt(component_flux * component_weights)
            scale = wavefront._to_phasor_shape(weight)
            return wavefront.set(phasor=wavefront.phasor * scale)

        # Initialize a single source component directly
        if position.ndim == 1:
            if weights.ndim != 1:
                raise ValueError(
                    "Single-component source weights must be one-dimensional."
                )
            return initialise(position, flux, weights)

        # Align weights and vectorise over source components
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, position.shape[:-1] + weights.shape)
        elif weights.shape[:-1] != position.shape[:-1]:
            raise ValueError(
                "Vectorised weights leading shape must match source positions."
            )
        initialise = eqx.filter_vmap(initialise)
        return initialise(position, flux, weights)

    def model(self, optics, return_all=False):
        """Model the source through an optical system."""
        # Resolve and propagate the source parameters
        params = self.params()
        result = self._propagate(optics, params)
        psf = result["PSF"]
        distribution = params["distribution"]

        # Convolve any resolved source distributions
        if distribution is not None:
            psf = psf.set(data=self._convolve(psf.data, distribution))

        # Collapse vectorised spatial source components
        if params["position"].ndim > 1:
            grid = psf.grid
            grid = grid.set(d=grid.d[0], c=None if grid.c is None else grid.c[0])
            psf = psf.set(data=psf.data.sum(0), grid=grid)

        # Package the modeled source outputs
        result = {**result, "PSF": psf, "psf": psf.data}
        if return_all:
            return result
        return psf


class Spectrum(ParametricHolder):
    """Wavelength samples and their corresponding spectral weights.

    Explicit weights are consumed exactly as supplied. Spectral parametrics may
    optionally normalise each spectrum to unit sum along its trailing wavelength
    axis. Realised weights must be positive with a finite, non-zero sum.

    Parameters
    ----------
    wavelengths : Array or Parametric
        Scalar or one-dimensional wavelength samples.
    weights : Array, Parametric, or None
        Spectral weights with a trailing axis matching ``wavelengths``.
    units : dict or None
        Unit overrides. Wavelengths accept supported length units such as ``"m"``,
        ``"um"``, ``"nm"``, or ``"angstrom"``.
    """

    wavelengths: Array | Parametric
    weights: Array | Parametric
    units: dict

    def __init__(self, wavelengths, weights=None, units=None):
        self.wavelengths = dlu.to_value(wavelengths, types=Parametric)
        if weights is None:
            if isinstance(self.wavelengths, Parametric):
                raise ValueError(
                    "weights are required when wavelengths are parametric."
                )
            weights = np.ones_like(self.wavelengths)
        self.weights = dlu.to_value(weights, types=Parametric)
        self.units = _merge_units(units)

    def spectrum_params(self, **context: Any) -> tuple[Array, Array]:
        """Resolve wavelengths and weights in canonical wavelength units.

        Scalar monochromatic inputs are promoted to a length-one spectral axis.
        """
        # Resolve wavelengths in canonical physical units
        wavelengths = resolve(self.wavelengths, float, spectrum=self, **context)
        wavelengths = np.atleast_1d(wavelengths)
        unit = self.units["wavelengths"]
        try:
            wavelengths = wavelengths * dlu.unit_factor(unit)
        except ValueError as error:
            raise ValueError(
                f"Unknown wavelength unit {unit!r}. See TODO: add units "
                "documentation link."
            ) from error

        # Resolve spectral weights on the wavelength samples
        weights = resolve(
            self.weights,
            float,
            spectrum=self,
            wavelengths=wavelengths,
            variables=wavelengths,
            **context,
        )
        weights = np.atleast_1d(weights)

        # Validate the shared trailing spectral-axis contract
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1d array.")
        if weights.ndim not in (1, 2):
            raise ValueError("weights must be a 1d or 2d array.")
        if weights.shape[-1] != wavelengths.shape[0]:
            raise ValueError("weights trailing axis must match the wavelength axis.")
        return wavelengths, weights

    def model(self, optics, return_all=False):
        """Model this spectrum as an on-axis, unit-flux point source."""
        return Source(self.wavelengths, weights=self.weights, units=self.units).model(
            optics, return_all
        )


class Source(BaseSource, Spectrum):
    """Represent one or more sources with spatial and spectral parameters.

    Parameters
    ----------
    wavelengths : Array or Parametric
        Scalar or one-dimensional wavelength samples.
    position : Array, Parametric, or None
        One on-sky position with shape ``(2,)``, or multiple positions with shape
        ``(nsource, 2)``, in the configured position unit.
    flux : Array, Parametric, or None
        Scalar or per-source flux in the configured flux unit.
    weights : Array, Parametric, or None
        Shared weights with shape ``(nwavelength,)`` or per-source weights with
        shape ``(nsource, nwavelength)``.
    distribution : Array, Parametric, or None
        Optional shared distribution with shape ``(y, x)`` or per-source
        distributions with shape ``(nsource, y, x)``.
    units : dict or None
        Unit overrides for ``wavelengths``, ``position``, ``flux``, and
        ``distribution``. Wavelengths accept supported length units; positions accept
        angular units such as ``"rad"``, ``"deg"``, ``"arcsec"``, or ``"mas"``.
        Flux defaults to ``"photon"`` and also accepts prefixed photon units,
        ``"log"`` for base-10 photon flux, or ``"ln"`` for natural-log photon flux.
        Distributions accept ``"linear"``, ``"log"``, ``"ln"``, or photon units.
    """

    wavelengths: Array | Parametric
    weights: Array | Parametric
    position: Array | Parametric | None
    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(
        self,
        wavelengths,
        position=None,
        flux=None,
        weights=None,
        distribution=None,
        units=None,
    ):
        self.position = dlu.to_value(position, optional=True, types=Parametric)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve all point-source parameters in canonical units."""
        # Resolve spectral and position parameters
        wavelengths, weights = self.spectrum_params()
        position = resolve(self.position, float, source=self, wavelengths=wavelengths)
        position = (
            np.zeros(2) if position is None else np.asarray(position, dtype=float)
        )
        valid = position.ndim in (1, 2) and position.shape[-1] == 2
        if not valid:
            raise ValueError("position must have shape (2,) or (nsource, 2).")
        unit = self.units["position"]
        try:
            position = position * dlu.unit_factor_to_rad(unit)
        except ValueError as error:
            raise ValueError(
                f"Unknown position unit {unit!r}. See TODO: add units "
                "documentation link."
            ) from error

        # Resolve brightness and optional spatial distribution
        nsource = None if position.ndim == 1 else position.shape[0]
        flux, distribution = self.source_params(nsource, wavelengths=wavelengths)
        return {
            "wavelengths": wavelengths,
            "weights": weights,
            "position": position,
            "flux": flux,
            "distribution": distribution,
        }


class BinarySource(BaseSource, Spectrum):
    """Represent a binary source by centre, separation, and contrast.

    Parameters
    ----------
    wavelengths : Array or Parametric
        Scalar or one-dimensional wavelength samples.
    centre : Array, Parametric, or None
        Mean on-sky ``(x, y)`` position.
    separation : Array or Parametric
        Angular component separation.
    position_angle : Array or Parametric
        Position angle in radians.
    contrast : Array or Parametric
        Ratio between the component fluxes.
    flux : Array, Parametric, or None
        Mean total flux of the binary.
    weights : Array, Parametric, or None
        Shared or component-dependent spectral weights.
    distribution : Array, Parametric, or None
        Shared or per-component resolved distributions.
    units : dict or None
        Unit overrides for wavelength, angular position, photon flux, and resolved
        distributions, following the same conventions as :class:`Source`.
    """

    wavelengths: Array | Parametric
    weights: Array | Parametric
    centre: Array | Parametric | None
    separation: Array | Parametric
    position_angle: Array | Parametric
    contrast: Array | Parametric
    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(
        self,
        wavelengths,
        centre=None,
        separation=0.0,
        position_angle=np.pi / 2,
        contrast=1.0,
        flux=None,
        weights=None,
        distribution=None,
        units=None,
    ):
        self.centre = dlu.to_value(centre, optional=True, types=Parametric)
        self.separation = dlu.to_value(separation, types=Parametric)
        self.position_angle = dlu.to_value(position_angle, types=Parametric)
        self.contrast = dlu.to_value(contrast, types=Parametric)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve all binary parameters in canonical units."""
        # Resolve the shared spectrum and binary geometry
        wavelengths, weights = self.spectrum_params()
        centre = resolve(self.centre, float, source=self)
        centre = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
        if centre.shape != (2,):
            raise ValueError("centre must have shape (2,).")
        separation = resolve(self.separation, float, source=self)
        position_angle = resolve(self.position_angle, float, source=self)
        contrast = resolve(self.contrast, float, source=self)
        factor = dlu.unit_factor(self.units["position"])
        position = dlu.positions_from_sep(
            centre * factor, separation * factor, position_angle
        )

        # Resolve total brightness into component fluxes
        mean_flux = self.flux_params(wavelengths=wavelengths)
        distribution = self.distribution_params(2, wavelengths=wavelengths)
        flux = dlu.fluxes_from_contrast(mean_flux, contrast)
        return {
            "wavelengths": wavelengths,
            "weights": weights,
            "position": position,
            "flux": flux,
            "distribution": distribution,
        }
