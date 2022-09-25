name = "dLux"
__version__ = "0.2"

# Import as modules
from . import base
from . import detectors
from . import layers
from . import wavefronts
from . import propagators
from . import utils
from . import apertures
from . import fresnel

# Import core functions from modules
from .base        import *
from .detectors   import *
from .layers      import *
from .wavefronts  import *
from .propagators import *
from .apertures   import *
from .fresnel     import *

# Add to __all__
__all__ = base.__all__ + detectors.__all__ + layers.__all__ + \
            wavefronts.__all__ + propagators.__all__ + apertures.__all__ +\
            fresnel.__all__
