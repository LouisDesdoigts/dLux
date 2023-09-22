import importlib.metadata

__version__ = importlib.metadata.version("dLux")

from . import (
    layers,
    optical_systems,
    detectors,
    instruments,
    sources,
    spectra,
    wavefronts,
    psfs,
    transformations,
)

# Add to __all__
modules = [
    optical_systems,
    detectors,
    sources,
    spectra,
    instruments,
    layers,
    wavefronts,
    psfs,
    transformations,
]

__all__ = [module.__all__ for module in modules]


from .detectors import (
    BaseDetector as BaseDetector,
    LayeredDetector as LayeredDetector,
)
from .spectra import (
    BaseSpectrum as BaseSpectrum,
    Spectrum as Spectrum,
    PolySpectrum as PolySpectrum,
)
from .wavefronts import Wavefront as Wavefront
from .psfs import PSF as PSF
from .transformations import CoordTransform as CoordTransform

from .instruments import (
    Instrument as Instrument,
    Telescope as Telescope,
    Dither as Dither,
)

from .optical_systems import (
    BaseOpticalSystem as BaseOpticalSystem,
    AngularOpticalSystem as AngularOpticalSystem,
    CartesianOpticalSystem as CartesianOpticalSystem,
    LayeredOpticalSystem as LayeredOpticalSystem,
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

from .layers import (
    optical_layers as optical_layers,
    detector_layers as detector_layers,
    unified_layers as unified_layers,
    propagators as propagators,
    apertures as apertures,
    aberrations as aberrations,
)

# Import core functions from modules
from .layers.optical_layers import (
    BaseLayer as BaseLayer,
    TransmissiveLayer as TransmissiveLayer,
    AberratedLayer as AberratedLayer,
    BasisLayer as BasisLayer,
    Tilt as Tilt,
    Normalise as Normalise,
)
from .layers.optics import (
    Optic as Optic,
    BasisOptic as BasisOptic,
)
from .layers.detector_layers import (
    ApplyPixelResponse as ApplyPixelResponse,
    ApplyJitter as ApplyJitter,
    ApplySaturation as ApplySaturation,
    AddConstant as AddConstant,
    Downsample as Downsample,
)
from .layers.unified_layers import (
    Rotate as Rotate,
    Flip as Flip,
    Resize as Resize,
)
from .layers.propagators import (
    MFT as MFT,
    FFT as FFT,
    ShiftedMFT as ShiftedMFT,
    FarFieldFresnel as FarFieldFresnel,
)
from .layers.apertures import (
    CircularAperture as CircularAperture,
    SquareAperture as SquareAperture,
    RectangularAperture as RectangularAperture,
    RegPolyAperture as RegPolyAperture,
    # IrregPolyAperture as IrregPolyAperture,
    Spider as Spider,
    AberratedAperture as AberratedAperture,
    CompoundAperture as CompoundAperture,
    MultiAperture as MultiAperture,
    # ApertureFactory as ApertureFactory,
)
from .layers.aberrations import (
    Zernike as Zernike,
    ZernikeBasis as ZernikeBasis,
)
