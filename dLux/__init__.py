import importlib.metadata
__version__ = importlib.metadata.version("dLux")

# Import as modules
# Wavefronts and Optics
from . import wavefronts  # noqa
from . import optics  # noqa
from . import optical_layers  # noqa
from . import propagators  # noqa
from . import apertures  # noqa
from . import aberrations  # noqa

# Images and Detectors
from . import images  # noqa
from . import detectors  # noqa
from . import detector_layers  # noqa

# All other classes
from . import instruments  # noqa
from . import observations  # noqa
from . import sources  # noqa
from . import spectra  # noqa

# Sub Modules
from . import utils  # noqa


# Import core functions from modules
# Wavefronts and Optics
from .wavefronts import *  # noqa
from .optics import *  # noqa
from .optical_layers import *  # noqa
from .propagators import *  # noqa
from .apertures import *  # noqa
from .aberrations import *  # noqa

# Images and Detectors
from .images import *  # noqa
from .detectors import *  # noqa
from .detector_layers import *  # noqa

# All other classes
from .sources import *  # noqa
from .spectra import *  # noqa
from .instruments import *  # noqa
from .observations import *  # noqa

# Sub Modules
# from .utils      import *


# Add to __all__
modules = [
    wavefronts,
    optics,
    optical_layers,
    propagators,
    apertures,
    aberrations,
    images,
    detectors,
    detector_layers,
    sources,
    spectra,
    instruments,
    observations,
]

__all__ = [module.__all__ for module in modules]
