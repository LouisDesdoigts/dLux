name = "dLux"
__version__ = "0.12.0"

# Import as modules
# Wavefronts and Optics
from . import wavefronts
from . import optics
from . import optical_layers
from . import propagators
from . import apertures
from . import aberrations

# Images and Detectors
from . import images
from . import detectors
from . import detector_layers

# All other classes
from . import instruments
from . import observations
from . import sources
from . import spectra

# Sub Modules
# from . import models
from . import utils


# Import core functions from modules
# Wavefronts and Optics
from .wavefronts     import *
from .optics         import *
from .optical_layers import *
from .propagators    import *
from .apertures      import *
from .aberrations    import *

# Images and Detectors
from .images          import *
from .detectors       import *
from .detector_layers import *

# All other classes
from .instruments  import *
from .observations import *
from .sources      import *
from .spectra    import *

# Sub Modules
# from .models     import *
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

    instruments,
    observations,
    sources,
    spectra,
]

__all__ = [module.__all__ for module in modules]