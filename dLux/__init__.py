name = "dLux"
__version__ = "0.9.1"

# Import as modules
# from . import base
from . import core
from . import detectors
from . import optics
from . import wavefronts
from . import propagators
from . import utils
from . import apertures
from . import sources
from . import spectrums
from . import basis
from . import spiders

# Import core functions from modules
# from .base        import *
from .core        import *
from .detectors   import *
from .optics      import *
from .wavefronts  import *
from .propagators import *
from .apertures   import *
from .sources     import*
from .spectrums   import *

# Add to __all__
__all__ = core.__all__ + detectors.__all__ + optics.__all__ + \
            wavefronts.__all__ + propagators.__all__ + sources.__all__ + \
            spectrums.__all__ + apertures.__all__ 

# Check for 64-bit
from jax import config
if not config.x64_enabled:
    print("dLux warning: Jax is running in 32-bit, to enable 64-bit visit:\nhttps://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision")
