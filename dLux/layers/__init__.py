import importlib.metadata


__version__ = importlib.metadata.version("dLux")

# Import as modules
from . import optical_layers
from . import detector_layers
from . import propagators
from . import apertures
from . import aberrations
from . import unified_layers

# from . import geometric_apertures

# Import core functions from modules
from .optical_layers import (
    Optic as Optic,
    BasisOptic as BasisOptic,
    Tilt as Tilt,
    Normalise as Normalise,
)
from .detector_layers import (
    ApplyPixelResponse as ApplyPixelResponse,
    ApplyJitter as ApplyJitter,
    ApplySaturation as ApplySaturation,
    AddConstant as AddConstant,
    IntegerDownsample as IntegerDownsample,
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
    ApertureFactory as ApertureFactory,
)
from .aberrations import (
    Zernike as Zernike,
    ZernikeBasis as ZernikeBasis,
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
