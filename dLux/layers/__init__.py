# Import as modules
from . import (
    optical_layers,
    detector_layers,
    propagators,
    apertures,
    aberrations,
    unified_layers,
)

# Add to __all__
modules = [
    optical_layers,
    unified_layers,
    propagators,
    apertures,
    aberrations,
    detector_layers,
]

__all__ = [module.__all__ for module in modules]

# Import core functions from modules
from .optical_layers import (
    TransmissiveLayer as TransmissiveLayer,
    AberratedLayer as AberratedLayer,
    BasisLayer as BasisLayer,
    Tilt as Tilt,
    Normalise as Normalise,
)
from .optics import (
    Optic as Optic,
    BasisOptic as BasisOptic,
)
from .detector_layers import (
    ApplyPixelResponse as ApplyPixelResponse,
    ApplyJitter as ApplyJitter,
    ApplySaturation as ApplySaturation,
    AddConstant as AddConstant,
    Downsample as Downsample,
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
from .aberrations import (
    Zernike as Zernike,
    ZernikeBasis as ZernikeBasis,
)
