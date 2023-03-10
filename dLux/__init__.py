name = "dLux"
__version__ = "0.10.1"

# Import as modules
from . import core
from . import detectors
from . import optics
from . import wavefronts
from . import propagators
from . import utils
from . import apertures
from . import aberrations
from . import observations
from . import sources
from . import spectrums
from . import exceptions

# Import core functions from modules
from .core         import *
from .detectors    import *
from .optics       import *
from .wavefronts   import *
from .propagators  import *
from .apertures    import *
from .aberrations  import *
from .observations import *
from .sources      import *
from .spectrums    import *
from .exceptions   import *

# Add to __all__
__all__ = core.__all__ + detectors.__all__ + optics.__all__ + \
    wavefronts.__all__ + propagators.__all__ + sources.__all__ + \
    spectrums.__all__ + apertures.__all__ + aberrations.__all__ + \
    observations.__all__

# Check for 64-bit
from jax import config
if not config.x64_enabled:
    print("dLux: Jax is running in 32-bit, to enable 64-bit visit: "
          "https://jax.readthedocs.io/en/latest/notebooks/"
          "Common_Gotchas_in_JAX.html#double-64bit-precision")