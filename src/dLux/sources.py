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
_VALUE_MODES = {
    "flux": ("log", "ln"),
    "distribution": ("linear", "log", "ln"),
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
        # Keep the legacy module lazy: compatibility imports Source and Spectrum.
        from . import compatibility

        return getattr(compatibility, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _brightness_unit(name, unit):
    """Resolve photon units and logarithmic value representations."""
    if unit in _VALUE_MODES[name]:
        return unit

    # Spatial distributions are relative scalings, not physical photon values
    if name == "distribution":
        modes = ", ".join(map(repr, _VALUE_MODES[name]))
        raise ValueError(f"Distribution unit must be one of {modes}.")

    try:
        return dlu.canonical_unit(unit, dimension="photon", name=f"{name} unit")
    except ValueError as error:
        label = name.capitalize()
        modes = ", ".join(map(repr, _VALUE_MODES[name]))
        raise ValueError(
            f"{label} unit must be a photon unit or one of {modes}."
        ) from error


def _merge_units(units=None):
    """Merge source unit overrides with canonical defaults."""
    units = {} if units is None else dict(units)
    unknown = set(units) - set(_DEFAULT_UNITS)
    if unknown:
        raise ValueError(f"Unknown source unit keys: {sorted(unknown)}.")

    # Merge defaults and canonicalise each physical or value representation
    units = {**_DEFAULT_UNITS, **units}
    units["wavelengths"] = dlu.canonical_unit(
        units["wavelengths"], dimension="length", name="wavelength unit"
    )
    units["position"] = dlu.canonical_unit(
        units["position"], dimension="angle", name="position unit"
    )
    units["flux"] = _brightness_unit("flux", units["flux"])
    units["distribution"] = _brightness_unit("distribution", units["distribution"])
    return units


def _convert_flux(flux, unit):
    """Convert linear or logarithmic flux into canonical units."""
    unit = str(unit).strip()
    if unit == "log":
        return 10**flux
    if unit == "ln":
        return np.exp(flux)
    factor = dlu.unit_factor(unit, dimension="photon", name="flux unit")
    return flux * factor


class BaseSource(ParametricHolder):
    """Source brightness and optional resolved distribution."""

    flux: Array | Parametric | None
    distribution: Array | Parametric | None
    units: dict

    def __init__(self, flux=None, distribution=None, units=None):
        """Initialise source brightness and optional spatial structure.

        Parameters
        ----------
        flux : Array, Parametric, or None
            Photon flux in the configured scaling. ``None`` resolves to unit flux.
        distribution : Array, Parametric, or None
            Shared or per-source relative spatial distribution.
        units : dict or None
            Overrides for the source ``flux`` and ``distribution`` conventions.
        """
        self.flux = dlu.to_value(flux, optional=True, types=Parametric)
        self.distribution = dlu.to_value(distribution, optional=True, types=Parametric)
        self.units = _merge_units(units)

    def source_params(self, nsource=None, **context):
        """Resolve flux and optional spatial distribution into linear values.

        ``nsource`` selects scalar or vectorised source validation. Additional
        ``context`` is passed to parametric leaves. Returns ``(flux, distribution)``;
        flux is in photons and distribution is ``None`` or a relative linear array.
        """
        flux = self.flux_params(nsource, **context)
        distribution = self.distribution_params(nsource, **context)
        return flux, distribution

    def flux_params(self, nsource=None, **context):
        """Resolve flux into photons using the configured source unit.

        A single source returns a scalar. With ``nsource`` supplied, scalar flux is
        broadcast and vector flux must have shape ``(nsource,)``.
        """
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
        """Resolve optional spatial distributions into linear relative weights.

        Accepted shapes are ``(y, x)`` for a shared distribution and
        ``(nsource, y, x)`` for per-source distributions. ``log`` and ``ln`` modes
        are exponentiated; no sum normalisation is applied.
        """
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

        # Convert the resolved distribution into a linear spatial scaling
        unit = str(self.units["distribution"]).strip()
        if unit == "linear":
            return distribution
        if unit == "log":
            return 10**distribution
        if unit == "ln":
            return np.exp(distribution)

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
        """Model this source through an optical system.

        ``optics`` propagates every spectral and spatial component. Returns the
        summed deterministic `Intensity`, optionally convolved by the source
        distribution, or the complete propagation mapping when ``return_all=True``.
        """
        # Resolve and propagate the source parameters
        params = self.params()
        result = self._propagate(optics, params)
        intensity = result["Intensity"]
        distribution = params["distribution"]

        # Convolve any resolved source distributions
        if distribution is not None:
            intensity = intensity.set(data=self._convolve(intensity.data, distribution))

        # Collapse vectorised spatial source components
        if params["position"].ndim > 1:
            d = intensity.grid.d[0]
            c = None if intensity.grid.c is None else intensity.grid.c[0]
            intensity = intensity.set(data=intensity.data.sum(0), d=d, c=c)

        # Package the modeled source outputs
        result = {
            **result,
            "Intensity": intensity,
            "intensity": intensity.data,
            "PSF": intensity,
            "psf": intensity.data,
        }
        if return_all:
            return result
        return intensity


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
        """Initialise wavelength samples and their spectral weights.

        Parameters
        ----------
        wavelengths : Array or Parametric
            Scalar or one-dimensional samples in the configured wavelength unit.
        weights : Array, Parametric, or None
            Values whose trailing axis matches the wavelength axis. Omitted weights
            default to ones for explicit wavelengths and are required for parametric
            wavelengths.
        units : dict or None
            Unit overrides, including ``wavelengths``.
        """
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
        Parametric weights receive the resolved metre-valued ``wavelengths`` in
        their context.

        Returns
        -------
        wavelengths : Array
            One-dimensional wavelength samples in metres.
        weights : Array
            Spectral weights whose trailing axis matches ``wavelengths``. Any leading
            axes represent vectorised spectra or source components.
        """
        # Resolve wavelengths in canonical physical units
        wavelengths = resolve(self.wavelengths, float, spectrum=self, **context)
        wavelengths = np.atleast_1d(wavelengths)
        unit = self.units["wavelengths"]
        factor = dlu.unit_factor(unit, dimension="length", name="wavelength unit")
        wavelengths = wavelengths * factor

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
        """Model this spectrum as an on-axis, unit-flux point source.

        ``optics`` is the target `OpticalSystem`. Returns its deterministic
        `Intensity`, or the complete propagation mapping when ``return_all=True``.
        """
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
        Distributions are relative spatial scalings and accept only ``"linear"``,
        ``"log"``, or ``"ln"``.
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
        """Initialise one point source or a vectorised source population.

        Parameters
        ----------
        wavelengths : Array or Parametric
            Scalar or one-dimensional samples in the configured wavelength unit.
        position : Array, Parametric, or None
            One ``(x, y)`` position or an ``(nsource, 2)`` population.
        flux : Array, Parametric, or None
            Scalar or ``(nsource,)`` photon flux in the configured scaling.
        weights : Array, Parametric, or None
            Shared ``(nwavelength,)`` or per-source
            ``(nsource, nwavelength)`` spectral weights.
        distribution : Array, Parametric, or None
            Shared ``(ny, nx)`` or per-source ``(nsource, ny, nx)`` distribution.
        units : dict or None
            Overrides for wavelength, position, flux, and distribution conventions.
        """
        self.position = dlu.to_value(position, optional=True, types=Parametric)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve all point-source parameters into a propagation mapping.

        Returns wavelengths in metres, normalised or raw spectral weights according
        to their parametric contract, position in radians, flux in photons, and the
        optional linear spatial distribution.
        """
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
        factor = dlu.unit_factor(unit, dimension="angle", name="position unit")
        position = position * factor

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
        """Initialise a binary source from relative geometry and brightness.

        Parameters
        ----------
        wavelengths : Array or Parametric
            Scalar or one-dimensional wavelength samples.
        centre : Array, Parametric, or None
            Mean ``(x, y)`` position in the configured angular unit.
        separation : Array or Parametric
            Component separation in the configured angular unit.
        position_angle : Array or Parametric
            Position angle in radians.
        contrast : Array or Parametric
            Secondary-to-primary flux ratio.
        flux : Array, Parametric, or None
            Total binary photon flux in the configured scaling.
        weights : Array, Parametric, or None
            Shared or component-dependent spectral weights.
        distribution : Array, Parametric, or None
            Shared or component-dependent resolved distributions.
        units : dict or None
            Overrides following the `Source` unit conventions.
        """
        self.centre = dlu.to_value(centre, optional=True, types=Parametric)
        self.separation = dlu.to_value(separation, types=Parametric)
        self.position_angle = dlu.to_value(position_angle, types=Parametric)
        self.contrast = dlu.to_value(contrast, types=Parametric)
        BaseSource.__init__(self, flux, distribution, units)
        Spectrum.__init__(self, wavelengths, weights, self.units)

    def params(self) -> dict:
        """Resolve the binary into per-component propagation parameters.

        Returns wavelengths in metres, spectral weights, component positions in
        radians with shape ``(2, 2)``, component photon fluxes with shape ``(2,)``,
        and any optional linear spatial distribution.
        """
        # Resolve the shared spectrum and binary geometry
        wavelengths, weights = self.spectrum_params()
        centre = resolve(self.centre, float, source=self)
        centre = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
        if centre.shape != (2,):
            raise ValueError("centre must have shape (2,).")
        separation = resolve(self.separation, float, source=self)
        position_angle = resolve(self.position_angle, float, source=self)
        contrast = resolve(self.contrast, float, source=self)
        factor = dlu.unit_factor(
            self.units["position"], dimension="angle", name="position unit"
        )
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
