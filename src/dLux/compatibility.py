"""Backward-compatible interfaces for dLux 0.14 and 0.15.

Legacy contracts are retained only when they translate without changing physical
meaning. Removed concepts remain importable but raise an explicit migration error
when constructed.
"""

from __future__ import annotations

from abc import abstractmethod
import sys
import warnings

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from .grids import BaseCoordTransform, BaseGridSpec, Distortion, GridSpec, ResizeSpec
from .fields import Intensity, Wavefront
from .layers.detector import DetectorLayer
from .layers.propagation import (
    ABCDFraunhofer,
    ABCDPropagator,
    Fraunhofer,
)
from .sources import Source, Spectrum
from .systems import DetectorSystem, OpticalSystem

REMOVAL_VERSION = "0.17.0"
MIGRATION_GUIDE = "https://louisdesdoigts.github.io/dLux/latest/migration/"


def warn_deprecated(old, new, example, stacklevel=2):
    """Warn about a deprecated interface and provide its direct migration."""
    message = (
        f"The `{old}` interface is deprecated and will be removed in dLux "
        f"{REMOVAL_VERSION}. Use `{new}` instead: {example}. See the migration "
        f"guide: {MIGRATION_GUIDE}"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def migration_error(old, new, example):
    """Raise an actionable error for a legacy contract without a safe wrapper."""
    message = (
        f"`{old}` was removed in dLux 0.16 and cannot be translated without "
        f"changing its behaviour. Use `{new}` instead: {example}. See the migration "
        f"guide: {MIGRATION_GUIDE}"
    )
    raise TypeError(message)


class PSF(Intensity):
    """Deprecated compatibility wrapper for ``Intensity``."""

    def __init__(self, data, pixel_scale):
        migration = "`dl.PSF(data, pixel_scale)` -> `dl.Intensity(data, grid)`"
        warn_deprecated("PSF", "Intensity", migration)

        # Preserve both the released scalar sampling and transitional grid input
        if isinstance(pixel_scale, GridSpec):
            grid = pixel_scale
        else:
            data = np.asarray(data, dtype=float)
            pixel_scale = np.asarray(pixel_scale, dtype=float)
            spacing = np.stack((pixel_scale, pixel_scale), axis=-1)
            grid = GridSpec(n=data.shape[-2:][::-1], d=spacing, unit="rad")
        super().__init__(data, grid)

    @property
    def pixel_scale(self):
        """Return the legacy scalar or vectorised angular sampling."""
        return self.grid.d[..., 0] * self.grid.scale

    @property
    def ndim(self):
        """Return the legacy pixel-scale vectorisation rank."""
        return self.pixel_scale.ndim


class LegacyDetectorLayer(DetectorLayer):
    """Preserve the released detector-layer extension contract."""

    @abstractmethod
    def __call__(self, intensity):
        """Transform an intensity using the released callable contract."""

    def apply(self, intensity):
        """Delegate the compatibility method to the legacy callable implementation."""
        return self(intensity)


class ApplyPixelResponse(LegacyDetectorLayer):
    """Deprecated compatibility wrapper for ``Sensitivity``."""

    pixel_response: Array

    def __init__(self, pixel_response):
        migration = "`dl.ApplyPixelResponse(value)` -> `dl.Sensitivity(value)`"
        warn_deprecated("ApplyPixelResponse", "Sensitivity", migration)
        self.pixel_response = np.asarray(pixel_response, dtype=float)

        if self.pixel_response.ndim != 2:
            raise ValueError("pixel_response must be a 2d array.")

    def __call__(self, intensity):
        """Apply the legacy pixel-response multiplication."""
        return intensity * self.pixel_response


class ApplyJitter(LegacyDetectorLayer):
    """Deprecated compatibility wrapper for ``Jitter``."""

    sigma: Array
    kernel_size: int
    oversample: int

    def __init__(self, sigma, kernel_size=9, oversample=3):
        migration = "`dl.ApplyJitter(sigma)` -> `dl.Jitter(sigma)`"
        warn_deprecated("ApplyJitter", "Jitter", migration)
        self.sigma = np.asarray(sigma, dtype=float)
        self.kernel_size = int(kernel_size)
        self.oversample = int(oversample)

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be greater than 0.")
        if self.oversample <= 0:
            raise ValueError("oversample must be greater than 0.")

    @property
    def kernel(self):
        """Return the legacy eagerly resolved kernel property."""
        kernel = dlu.gaussian(
            mean=np.zeros(2),
            std=np.repeat(self.sigma, 2),
            npixels=self.kernel_size * self.oversample,
        )
        return dlu.downsample(kernel, self.oversample, mean=False)

    def __call__(self, intensity):
        """Apply the legacy jitter convolution."""
        return intensity.convolve(self.kernel)


class ApplySaturation(LegacyDetectorLayer):
    """Deprecated compatibility wrapper for ``Saturation``."""

    threshold: Array

    def __init__(self, threshold):
        migration = "`dl.ApplySaturation(value)` -> `dl.Saturation(value)`"
        warn_deprecated("ApplySaturation", "Saturation", migration)
        self.threshold = np.asarray(threshold, dtype=float)

    def __call__(self, intensity):
        """Apply the legacy saturation threshold."""
        return intensity.set(data=np.minimum(intensity.data, self.threshold))


class AddConstant(LegacyDetectorLayer):
    """Deprecated compatibility wrapper for ``Bias``."""

    value: Array

    def __init__(self, value):
        migration = "`dl.AddConstant(value)` -> `dl.Bias(value)`"
        warn_deprecated("AddConstant", "Bias", migration)
        self.value = np.asarray(value, dtype=float)

    def __call__(self, intensity):
        """Apply the legacy additive constant."""
        return intensity + self.value


class LegacyDownsample(LegacyDetectorLayer):
    """Preserve the released detector downsampling contract."""

    kernel_size: int

    def __init__(self, kernel_size):
        self.kernel_size = int(kernel_size)

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be greater than 0.")

    def __call__(self, intensity):
        """Downsample by summing detector pixels in fixed blocks."""
        return intensity.downsample(self.kernel_size)


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


class CoordTransform(BaseCoordTransform):
    """Deprecated 0.15 semantic coordinate transformation."""

    translation: Array | None
    rotation: Array | None
    compression: Array | None
    shear: Array | None

    def __init__(
        self,
        translation=None,
        rotation=None,
        compression=None,
        shear=None,
    ):
        migration = "`dl.CoordTransform(...)` -> `dl.Affine(...)`"
        warn_deprecated("CoordTransform", "Affine", migration)

        self.translation = dlu.to_value(translation, optional=True)
        self.rotation = dlu.to_value(rotation, optional=True)
        self.compression = dlu.to_value(compression, optional=True)
        self.shear = dlu.to_value(shear, optional=True)

        for name in ("translation", "compression", "shear"):
            value = getattr(self, name)
            if value is not None and value.shape != (2,):
                raise ValueError(f"{name} must have shape (2,).")
        if self.rotation is not None and self.rotation.shape != ():
            raise ValueError("rotation must have shape ().")

    def __call__(self, coords):
        """Apply the legacy transformation order to coordinates."""
        if self.translation is not None:
            coords = dlu.translate_coords(coords, self.translation)
        if self.shear is not None:
            coords = dlu.shear_coords(coords, self.shear)
        if self.compression is not None:
            coords = dlu.compress_coords(coords, self.compression)
        if self.rotation is not None:
            coords = dlu.rotate_coords(coords, self.rotation)
        return coords


class DistortedCoords(Distortion):
    """Deprecated compatibility wrapper for ``Distortion``."""

    def __init__(self, order=1, distortion=None):
        migration = "`dl.DistortedCoords(...)` -> `dl.Distortion(...)`"
        warn_deprecated("DistortedCoords", "Distortion", migration)
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
    "ParametricLayeredOpticalSystem": (
        "OpticalSystem",
        "place parametrics directly in layers passed to `dl.OpticalSystem`",
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
ParametricLayeredOpticalSystem = _removed_class(
    "ParametricLayeredOpticalSystem", *_REMOVED["ParametricLayeredOpticalSystem"]
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
        "ParametricLayeredOpticalSystem": ParametricLayeredOpticalSystem,
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
detector_layers = _legacy_module(
    "layers.detector_layers",
    "dLux.layers.detector and dLux.layers.unified",
    {
        "DetectorLayer": LegacyDetectorLayer,
        "ApplyPixelResponse": ApplyPixelResponse,
        "ApplyJitter": ApplyJitter,
        "ApplySaturation": ApplySaturation,
        "AddConstant": AddConstant,
        "Downsample": LegacyDownsample,
    },
)


COMPATIBILITY = {
    "0.14": (
        "AddConstant",
        "ApplyJitter",
        "ApplyPixelResponse",
        "ApplySaturation",
        "CoordSpec",
        "CoordTransform",
        "DistortedCoords",
        "LayeredDetector",
        "LayeredOpticalSystem",
        "PSF",
    ),
    "0.15": tuple(
        sorted(
            {
                "ABCDConjugatePlane",
                "AddConstant",
                "ApplyJitter",
                "ApplyPixelResponse",
                "ApplySaturation",
                "CoordSpec",
                "CoordTransform",
                "DistortedCoords",
                "FFT",
                "FFTPropagator",
                "LayeredDetector",
                "LayeredOpticalSystem",
                "MFT",
                "MFTPropagator",
                "PadSpec",
                "PointSource",
                "PSF",
                "ResolvedSource",
                *_REMOVED,
            }
        )
    ),
}


__all__ = [
    "ABCDConjugatePlane",
    "AddConstant",
    "ASMPropagator",
    "AberratedAperture",
    "AngularOpticalSystem",
    "ApplyJitter",
    "ApplyPixelResponse",
    "ApplySaturation",
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
    "CoordTransform",
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
    "ParametricLayeredOpticalSystem",
    "PointResolvedSource",
    "PointSource",
    "PointSources",
    "PSF",
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
