import importlib.metadata


__version__ = importlib.metadata.version("dLux")

# Import as modules
# Wavefronts and Optics
from . import wavefronts
from . import optical_systems
from . import optical_layers
from . import detector_layers
from . import unified_layers
from . import propagators
from . import apertures
from . import aberrations

# Images and Detectors
from . import images
from . import detectors

# from . import detector_layers

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
from .optical_layers import (
    Optic as Optic,
    BasisOptic as BasisOptic,
    Tilt as Tilt,
    Normalise as Normalise,
)
from .unified_layers import (
    Rotate as Rotate,
    Flip as Flip,
    Resize as Resize,
)
from .propagators import (
    MFT as MFT,
    FFT as FFT,
    ShiftedMFT as ShiftedMFT,
    FarFieldFresnel as FarFieldFresnel,
)
from .apertures import (
    CircularAperture as CircularAperture,
    RectangularAperture as RectangularAperture,
    RegPolyAperture as RegPolyAperture,
    IrregPolyAperture as IrregPolyAperture,
    AberratedAperture as AberratedAperture,
    UniformSpider as UniformSpider,
    CompoundAperture as CompoundAperture,
    MultiAperture as MultiAperture,
    ApertureFactory as ApertureFactory,
)
from .aberrations import (
    Zernike as Zernike,
    ZernikeBasis as ZernikeBasis,
)

# Images and Detectors
from .psfs import PSF as PSF
from .detectors import LayeredDetector as LayeredDetector
from .detector_layers import (
    ApplyPixelResponse as ApplyPixelResponse,
    ApplyJitter as ApplyJitter,
    ApplySaturation as ApplySaturation,
    AddConstant as AddConstant,
    IntegerDownsample as IntegerDownsample,
)

# All other classes
from .sources import (
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

# from .observations import Dither as Dither


# Add to __all__
modules = [
    wavefronts,
    optical_systems,
    optical_layers,
    unified_layers,
    propagators,
    apertures,
    aberrations,
    images,
    detectors,
    detector_layers,
    sources,
    spectra,
    instruments,
    # observations,
]

__all__ = [module.__all__ for module in modules]
