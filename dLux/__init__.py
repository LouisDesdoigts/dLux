name = "dLux"
__version__ = "0.2"

# Import as modules
from . import base
from . import core
from . import detectors
from . import optics
from . import wavefronts
from . import propagators
from . import utils
from . import sources
from . import spectrums

# Import core functions from modules
from .base        import *
from .core        import *
from .detectors   import *
from .optics      import *
from .wavefronts  import *
from .propagators import *
from .sources     import *
from .spectrums   import *

# Add to __all__
__all__ = core.__all__ + detectors.__all__ + optics.__all__ + \
            wavefronts.__all__ + propagators.__all__ + sources.__all__ + \
            spectrums.__all__