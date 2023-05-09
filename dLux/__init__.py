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
from . import spectrums

# Sub Modules
from . import models
from . import utils
from . import exceptions


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
from .spectrums    import *

# Sub Modules
from .models     import *
from .utils      import *
from .exceptions import *


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
    spectrums,
]

__all__ = [module.__all__ for module in modules]


# Check for 64-bit
from jax import config
if not config.x64_enabled:
    print("dLux: Jax is running in 32-bit, to enable 64-bit visit: "
          "https://jax.readthedocs.io/en/latest/notebooks/"
          "Common_Gotchas_in_JAX.html#double-64bit-precision")