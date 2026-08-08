"""Backward-compatible interfaces for dLux 0.14 and 0.15.

Legacy contracts are retained only when they translate without changing physical
meaning. Removed concepts remain importable but raise an explicit migration error
when constructed.
"""

from __future__ import annotations

import sys
import warnings

import jax.numpy as np

from .grids import BaseGridSpec, CoordTransform, DistortCoords, GridSpec, ResizeSpec
from .fields import PSF, Wavefront
from .layers.propagation import (
    ABCDFraunhofer,
    ABCDPropagator,
    Fraunhofer,
)
from .sources import Source, Spectrum
from .systems import DetectorSystem, OpticalSystem

REMOVAL_VERSION = "0.17.0"


def warn_deprecated(old, new, example, stacklevel=2):
    """Warn about a deprecated interface and provide its direct migration."""
    message = (
        f"The `{old}` interface is deprecated and will be removed in dLux "
        f"{REMOVAL_VERSION}. Use `{new}` instead: {example}."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def migration_error(old, new, example):
    """Raise an actionable error for a legacy contract without a safe wrapper."""
    message = (
        f"`{old}` was removed in dLux 0.16 and cannot be translated without "
        f"changing its behaviour. Use `{new}` instead: {example}."
    )
    raise TypeError(message)


class CoordSpec(GridSpec):
    """Deprecated one-dimensional compatibility wrapper for ``GridSpec``."""

    def __init__(self, n=None, d=None, c=0.0):
        migration = "`dl.CoordSpec(...)` -> `dl.GridSpec(...)`"
        warn_deprecated("CoordSpec", "GridSpec", migration)
        super().__init__(n, d, c)

    @property
    def xs(self):
        """Return the legacy one-dimensional coordinate array."""
        return super().xs[0]

    @property
    def fov(self):
        """Return the legacy scalar field of view."""
        return super().fov[0]

    @property
    def extent(self):
        """Return the legacy one-dimensional coordinate extent."""
        lower, upper = (
            self.c - np.asarray(self.n) * self.d / 2,
            self.c + np.asarray(self.n) * self.d / 2,
        )
        return lower[0], upper[0]


class PadSpec(ResizeSpec):
    """Deprecated compatibility wrapper for ``ResizeSpec``."""

    def __init__(self, pad=1, crop=1, c=0.0):
        migration = "`dl.PadSpec(...)` -> `dl.ResizeSpec(...)`"
        warn_deprecated("PadSpec", "ResizeSpec", migration)
        super().__init__(pad=pad, crop=crop, c=c)


class DistortedCoords(DistortCoords):
    """Deprecated compatibility wrapper for ``DistortCoords``."""

    def __init__(self, order=1, distortion=None):
        migration = "`dl.DistortedCoords(...)` -> `dl.DistortCoords(...)`"
        warn_deprecated("DistortedCoords", "DistortCoords", migration)
        super().__init__(order, distortion)

    def calculate(self, npix, diameter):
        """Evaluate the legacy 0.14 coordinate-generation contract."""
        coordinates = GridSpec(n=(npix, npix), diam=(diameter, diameter)).coordinates
        return self(coordinates)


class LayeredDetector(DetectorSystem):
    """Deprecated detector system preserving the 0.15 return contract."""

    def __init__(self, layers):
        migration = "`dl.LayeredDetector(layers)` -> `dl.DetectorSystem(layers)`"
        warn_deprecated("LayeredDetector", "DetectorSystem", migration)
        super().__init__(layers)

    def __call__(self, psf, return_psf=False):
        output = super().__call__(psf)
        return output if return_psf else output.data

    def model(self, psf, return_psf=False):
        """Apply the detector while preserving the legacy output selection."""
        return self(psf, return_psf)


class LayeredOpticalSystem(OpticalSystem):
    """Deprecated optical system using the legacy pupil-grid constructor."""

    def __init__(self, wf_npixels, diameter, layers):
        old = "`dl.LayeredOpticalSystem(n, d, layers)`"
        new = "`dl.OpticalSystem(layers, GridSpec(n=n, diam=d, unit='m'))`"
        migration = f"{old} -> {new}"
        warn_deprecated("LayeredOpticalSystem", "OpticalSystem", migration)
        super().__init__(layers, GridSpec(n=wf_npixels, diam=diameter, unit="m"))

    @property
    def wf_npixels(self):
        """Return the legacy pupil pixel count."""
        return self.grid.n[0]

    @property
    def diameter(self):
        """Return the legacy pupil diameter in metres."""
        return self.grid.n[0] * self.grid.d[0]


class PointSource(Source):
    """Deprecated wrapper for the former point-source constructor."""

    def __init__(
        self,
        wavelengths=None,
        position=None,
        flux=1.0,
        weights=None,
        spectrum=None,
    ):
        if spectrum is not None:
            example = "`dl.Source(spectrum.wavelengths, weights=spectrum.weights)`"
            migration_error("PointSource(spectrum=...)", "Source", example)
        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum(-1, keepdims=True)
        migration = "`dl.PointSource(...)` -> `dl.Source(...)`"
        warn_deprecated("PointSource", "Source", migration)
        super().__init__(wavelengths, position, flux, weights)


class PointSources(Source):
    """Deprecated wrapper for the vectorised ``Source`` contract."""

    def __init__(
        self,
        wavelengths=None,
        position=None,
        flux=None,
        weights=None,
        spectrum=None,
    ):
        if spectrum is not None:
            example = "`dl.Source(spectrum.wavelengths, weights=spectrum.weights)`"
            migration_error("PointSources(spectrum=...)", "Source", example)
        position = np.zeros((1, 2)) if position is None else np.asarray(position)
        flux = np.ones(position.shape[0]) if flux is None else flux
        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum(-1, keepdims=True)
        migration = "`dl.PointSources(...)` -> `dl.Source(...)`"
        warn_deprecated("PointSources", "Source", migration)
        super().__init__(wavelengths, position, flux, weights)


class ResolvedSource(Source):
    """Deprecated wrapper for a point source with a resolved distribution."""

    def __init__(
        self,
        wavelengths=None,
        position=None,
        flux=1.0,
        distribution=None,
        weights=None,
        spectrum=None,
    ):
        if spectrum is not None:
            example = "`dl.Source(spectrum.wavelengths, distribution=image)`"
            migration_error("ResolvedSource(spectrum=...)", "Source", example)
        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum(-1, keepdims=True)
        if distribution is not None:
            distribution = np.asarray(distribution)
            distribution = distribution / distribution.sum()
        migration = "`dl.ResolvedSource(...)` -> `dl.Source(..., distribution=...)`"
        warn_deprecated("ResolvedSource", "Source", migration)
        super().__init__(wavelengths, position, flux, weights, distribution)


class MFT(Fraunhofer):
    """Deprecated wrapper for the legacy direct MFT propagator."""

    def __init__(self, npixels, pixel_scale, focal_length=None, inverse=False):
        unit = "rad" if focal_length is None else "m"
        grid = GridSpec(n=npixels, d=pixel_scale, unit=unit)
        migration = "`dl.MFT(n, d, f)` -> `dl.Fraunhofer(GridSpec(n=n, d=d), f)`"
        warn_deprecated("MFT", "Fraunhofer", migration)
        super().__init__(grid, focal_length, "mft", inverse)


class FFT(Fraunhofer):
    """Deprecated wrapper for the legacy direct FFT propagator."""

    def __init__(self, focal_length=None, inverse=False, pad=1, crop=1, center=True):
        if not center:
            example = "`dl.Fraunhofer(dl.ResizeSpec(...), method='fft')`"
            migration_error("FFT(center=False)", "Fraunhofer", example)
        grid = ResizeSpec(pad=pad, crop=crop, c=0.0)
        migration = "`dl.FFT(...)` -> `dl.Fraunhofer(ResizeSpec(...), method='fft')`"
        warn_deprecated("FFT", "Fraunhofer", migration)
        super().__init__(grid, focal_length, "fft", inverse)


class MFTPropagator(ABCDPropagator):
    """Deprecated LCT-based ABCD propagation wrapper."""

    def __init__(self, ABCDs, grid):
        migration = (
            "`dl.MFTPropagator(ABCDs, grid)` -> `dl.ABCDPropagator(ABCDs, grid)`"
        )
        warn_deprecated("MFTPropagator", "ABCDPropagator", migration)
        super().__init__(ABCDs, grid, "lct")


class FFTPropagator(ABCDPropagator):
    """Deprecated FFT-based ABCD propagation wrapper."""

    def __init__(self, ABCDs, grid):
        old = "`dl.FFTPropagator(ABCDs, grid)`"
        new = "`dl.ABCDPropagator(ABCDs, grid, method='fft')`"
        migration = f"{old} -> {new}"
        warn_deprecated("FFTPropagator", "ABCDPropagator", migration)
        super().__init__(ABCDs, grid, "fft")


class ABCDConjugatePlane(ABCDFraunhofer):
    """Deprecated name for ``ABCDFraunhofer``."""

    def __init__(self, focal_length):
        migration = "`dl.ABCDConjugatePlane(f)` -> `dl.ABCDFraunhofer(f)`"
        warn_deprecated("ABCDConjugatePlane", "ABCDFraunhofer", migration)
        super().__init__(focal_length)


def _removed_class(name, replacement, example):
    """Construct a named migration-error sentinel."""

    def __init__(self, *args, **kwargs):
        migration_error(name, replacement, example)

    return type(name, (), {"__init__": __init__, "__module__": __name__})


_REMOVED = {
    "ASMPropagator": ("FreeSpace", "`dl.FreeSpace(distance, ResizeSpec(...))`"),
    "AberratedAperture": (
        "ApertureBuilder",
        "build an aperture and pass its basis to `dl.Optic`",
    ),
    "AngularOpticalSystem": (
        "OpticalSystem",
        "append an explicit angular-grid `dl.Fraunhofer` layer",
    ),
    "BasisLayer": ("Optic", "`dl.Optic(opd=dl.Basis(...))`"),
    "BasisOptic": ("Optic", "`dl.Optic(transmission=mask, opd=dl.Basis(...))`"),
    "CartesianOpticalSystem": (
        "OpticalSystem",
        "append a physical-grid `dl.Fraunhofer` layer",
    ),
    "CircularAperture": (
        "Circle",
        "`dl.DynamicTransmissiveLayer(dl.Circle(diameter))`",
    ),
    "CompoundAperture": (
        "ApertureBuilder",
        "compose shape definitions in an aperture builder",
    ),
    "Dither": (
        "Source and OpticalSystem",
        "model explicit source positions through an optical system",
    ),
    "Instrument": (
        "Source and OpticalSystem",
        "compose source, optics, and detector models explicitly",
    ),
    "MultiAperture": (
        "SparseApertureBuilder",
        "construct or retain sub-apertures explicitly",
    ),
    "ParametricOpticalSystem": (
        "OpticalSystem",
        "place parametrics directly in optical layers",
    ),
    "PointResolvedSource": (
        "Source",
        "use vectorised positions and resolved distributions explicitly",
    ),
    "PolySpectrum": (
        "SpectralPolynomial",
        "`dl.Source(wavelengths, weights=dl.SpectralPolynomial(...))`",
    ),
    "RectangularAperture": (
        "Rectangle",
        "`dl.DynamicTransmissiveLayer(dl.Rectangle(width, height))`",
    ),
    "RegPolyAperture": (
        "RegPolygon",
        "`dl.DynamicTransmissiveLayer(dl.RegPolygon(...))`",
    ),
    "Rotate": ("Interpolate and Affine", "`dl.Interpolate(dl.Affine(rotation=angle))`"),
    "Scene": ("Source", "combine explicitly vectorised source parameters"),
    "SquareAperture": ("Square", "`dl.DynamicTransmissiveLayer(dl.Square(width))`"),
    "Telescope": (
        "Source, OpticalSystem, and DetectorSystem",
        "compose the three models explicitly",
    ),
    "Zernike": ("ZernikeBasis", "use `dl.ZernikeBasis` as an OPD or phase parametric"),
}

ASMPropagator = _removed_class("ASMPropagator", *_REMOVED["ASMPropagator"])
AberratedAperture = _removed_class("AberratedAperture", *_REMOVED["AberratedAperture"])
AngularOpticalSystem = _removed_class(
    "AngularOpticalSystem", *_REMOVED["AngularOpticalSystem"]
)
BasisLayer = _removed_class("BasisLayer", *_REMOVED["BasisLayer"])
BasisOptic = _removed_class("BasisOptic", *_REMOVED["BasisOptic"])
CartesianOpticalSystem = _removed_class(
    "CartesianOpticalSystem", *_REMOVED["CartesianOpticalSystem"]
)
CircularAperture = _removed_class("CircularAperture", *_REMOVED["CircularAperture"])
CompoundAperture = _removed_class("CompoundAperture", *_REMOVED["CompoundAperture"])
Dither = _removed_class("Dither", *_REMOVED["Dither"])
Instrument = _removed_class("Instrument", *_REMOVED["Instrument"])
MultiAperture = _removed_class("MultiAperture", *_REMOVED["MultiAperture"])
ParametricOpticalSystem = _removed_class(
    "ParametricOpticalSystem", *_REMOVED["ParametricOpticalSystem"]
)
PointResolvedSource = _removed_class(
    "PointResolvedSource", *_REMOVED["PointResolvedSource"]
)
PolySpectrum = _removed_class("PolySpectrum", *_REMOVED["PolySpectrum"])
RectangularAperture = _removed_class(
    "RectangularAperture", *_REMOVED["RectangularAperture"]
)
RegPolyAperture = _removed_class("RegPolyAperture", *_REMOVED["RegPolyAperture"])
Rotate = _removed_class("Rotate", *_REMOVED["Rotate"])
Scene = _removed_class("Scene", *_REMOVED["Scene"])
SquareAperture = _removed_class("SquareAperture", *_REMOVED["SquareAperture"])
Telescope = _removed_class("Telescope", *_REMOVED["Telescope"])
Zernike = _removed_class("Zernike", *_REMOVED["Zernike"])


# Abstract legacy coordinate names retain aligned current contracts. The removed
# modelling bases raise migration errors because their inheritance contracts changed.
Spec = BaseGridSpec
BaseCoordTransform = CoordTransform
BaseDetector = _removed_class(
    "BaseDetector",
    "DetectorSystem or BaseDetectorLayer",
    "subclass the contract matching the object being implemented",
)
BaseOpticalSystem = _removed_class(
    "BaseOpticalSystem",
    "OpticalSystem or BaseOpticalLayer",
    "subclass the contract matching the object being implemented",
)
BaseSpectrum = _removed_class(
    "BaseSpectrum",
    "Spectrum or Parametric",
    "subclass `Spectrum` or implement spectral weights as a `Parametric`",
)


class _LegacyModule:
    """Expose a deprecated module path from one central namespace."""

    def __init__(self, name, replacement, names):
        self.__name__ = f"dLux.{name}"
        self.__all__ = tuple(names)
        self._name = name
        self._replacement = replacement

    def __getattribute__(self, name):
        if not name.startswith("_"):
            module = object.__getattribute__(self, "_name")
            replacement = object.__getattribute__(self, "_replacement")
            example = f"`dLux.{module}` -> `{replacement}`"
            warn_deprecated(f"dLux.{module} module", replacement, example, 3)
        return object.__getattribute__(self, name)


def _legacy_module(name, replacement, values):
    """Build and register a module-like legacy namespace."""
    namespace = _LegacyModule(name, replacement, values)
    for key, value in values.items():
        object.__setattr__(namespace, key, value)
    sys.modules[f"dLux.{name}"] = namespace
    return namespace


coordinates = _legacy_module(
    "coordinates",
    "dLux.grids",
    {
        "Spec": Spec,
        "PadSpec": PadSpec,
        "CoordSpec": CoordSpec,
        "BaseCoordTransform": BaseCoordTransform,
        "CoordTransform": CoordTransform,
        "DistortedCoords": DistortedCoords,
    },
)
detectors = _legacy_module(
    "detectors",
    "dLux.systems",
    {"BaseDetector": BaseDetector, "LayeredDetector": LayeredDetector},
)
instruments = _legacy_module(
    "instruments",
    "explicit model composition",
    {"Instrument": Instrument, "Telescope": Telescope, "Dither": Dither},
)
optical_systems = _legacy_module(
    "optical_systems",
    "dLux.systems",
    {
        "BaseOpticalSystem": BaseOpticalSystem,
        "ParametricOpticalSystem": ParametricOpticalSystem,
        "LayeredOpticalSystem": LayeredOpticalSystem,
        "AngularOpticalSystem": AngularOpticalSystem,
        "CartesianOpticalSystem": CartesianOpticalSystem,
    },
)
psfs = _legacy_module("psfs", "dLux.fields", {"PSF": PSF})
spectra = _legacy_module(
    "spectra",
    "dLux.sources and dLux.parametric",
    {
        "BaseSpectrum": BaseSpectrum,
        "Spectrum": Spectrum,
        "PolySpectrum": PolySpectrum,
    },
)
wavefronts = _legacy_module(
    "wavefronts",
    "dLux.fields",
    {"Wavefront": Wavefront},
)


COMPATIBILITY = {
    "0.14": ("CoordSpec", "DistortedCoords", "LayeredDetector", "LayeredOpticalSystem"),
    "0.15": tuple(
        sorted(
            {
                "ABCDConjugatePlane",
                "CoordSpec",
                "DistortedCoords",
                "FFT",
                "FFTPropagator",
                "LayeredDetector",
                "LayeredOpticalSystem",
                "MFT",
                "MFTPropagator",
                "PadSpec",
                "PointSource",
                "ResolvedSource",
                *_REMOVED,
            }
        )
    ),
}


__all__ = [
    "ABCDConjugatePlane",
    "ASMPropagator",
    "AberratedAperture",
    "AngularOpticalSystem",
    "BaseCoordTransform",
    "BaseDetector",
    "BaseOpticalSystem",
    "BaseSpectrum",
    "BasisLayer",
    "BasisOptic",
    "CartesianOpticalSystem",
    "CircularAperture",
    "CompoundAperture",
    "CoordSpec",
    "DistortedCoords",
    "Dither",
    "FFT",
    "FFTPropagator",
    "Instrument",
    "LayeredDetector",
    "LayeredOpticalSystem",
    "MFT",
    "MFTPropagator",
    "MultiAperture",
    "PadSpec",
    "ParametricOpticalSystem",
    "PointResolvedSource",
    "PointSource",
    "PointSources",
    "PolySpectrum",
    "RectangularAperture",
    "RegPolyAperture",
    "ResolvedSource",
    "Rotate",
    "Scene",
    "Spec",
    "SquareAperture",
    "Telescope",
    "Zernike",
    "coordinates",
    "detectors",
    "instruments",
    "optical_systems",
    "psfs",
    "spectra",
    "wavefronts",
]
