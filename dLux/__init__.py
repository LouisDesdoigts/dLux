import importlib.metadata


__version__ = importlib.metadata.version("dLux")

# Import as modules
# Wavefronts and Optics
from . import base
from . import wavefronts
from . import optical_systems
from . import layers as layers

# Images and Detectors
from . import images
from . import detectors

# All other classes
from . import instruments

# from . import observations
from . import sources
from . import spectra

# Sub Modules
from . import utils as utils


# Import core functions from modules
# Wavefronts and Optics
from .wavefronts import (
    Wavefront as Wavefront,
)
from .optical_systems import (
    AngularOptics as AngularOptics,
    CartesianOptics as CartesianOptics,
    FlexibleOptics as FlexibleOptics,
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

# Images and Detectors
from .psfs import PSF as PSF
from .detectors import LayeredDetector as LayeredDetector

# All other classes
from .sources import (
    Scene as Scene,
    PointSource as PointSource,
    PointSources as PointSources,
    BinarySource as BinarySource,
    ResolvedSource as ResolvedSource,
    PointResolvedSource as PointResolvedSource,
)
from .spectra import (
    Spectrum as Spectrum,
    PolySpectrum as PolySpectrum,
)
from .instruments import Instrument as Instrument, Dither as Dither

from .layers import (
    Optic as Optic,
    BasisOptic as BasisOptic,
    Tilt as Tilt,
    Normalise as Normalise,
    Rotate as Rotate,
    Flip as Flip,
    Resize as Resize,
    MFT as MFT,
    FFT as FFT,
    ShiftedMFT as ShiftedMFT,
    FarFieldFresnel as FarFieldFresnel,
    CircularAperture as CircularAperture,
    RectangularAperture as RectangularAperture,
    RegPolyAperture as RegPolyAperture,
    IrregPolyAperture as IrregPolyAperture,
    AberratedAperture as AberratedAperture,
    UniformSpider as UniformSpider,
    CompoundAperture as CompoundAperture,
    MultiAperture as MultiAperture,
    ApertureFactory as ApertureFactory,
    Zernike as Zernike,
    ZernikeBasis as ZernikeBasis,
    ApplyPixelResponse as ApplyPixelResponse,
    ApplyJitter as ApplyJitter,
    ApplySaturation as ApplySaturation,
    AddConstant as AddConstant,
    IntegerDownsample as IntegerDownsample,
)

# Add to __all__
modules = [
    base,
    wavefronts,
    optical_systems,
    images,
    detectors,
    sources,
    spectra,
    instruments,
    layers,
]

__all__ = [module.__all__ for module in modules]
