import importlib.metadata

__version__ = importlib.metadata.version("dLux")

from . import (
    containers,
    layers,
    optical_systems,
    detectors,
    instruments,
    sources,
    spectra,
)

# Add to __all__
modules = [
    optical_systems,
    detectors,
    sources,
    spectra,
    instruments,
    layers,
    containers,
]

__all__ = [module.__all__ for module in modules]


from .detectors import LayeredDetector as LayeredDetector
from .spectra import Spectrum as Spectrum, PolySpectrum as PolySpectrum

from .instruments import (
    Instrument as Instrument,
    Telescope as Telescope,
    Dither as Dither,
)

from .optical_systems import (
    BaseOpticalSystem as BaseOpticalSystem,
    AngularOptics as AngularOptics,
    CartesianOptics as CartesianOptics,
    LayeredOptics as LayeredOptics,
)

from .layers import (
    optical_layers as optical_layers,
    detector_layers as detector_layers,
    unified_layers as unified_layers,
    propagators as propagators,
    apertures as apertures,
    aberrations as aberrations,
)

from .sources import (
    BaseSource as BaseSource,
    Scene as Scene,
    PointSource as PointSource,
    PointSources as PointSources,
    BinarySource as BinarySource,
    ResolvedSource as ResolvedSource,
    PointResolvedSource as PointResolvedSource,
)
