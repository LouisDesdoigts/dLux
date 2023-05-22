name = "toliman"

# Import as modules
from . import optics
from . import optical_layers
from . import sources
from . import gradient_energy

# Dont import all functions from modules
from .optics          import *
from .optical_layers  import *
from .sources         import *
from .gradient_energy import *

# Add to __all__
__all__ = optics.__all__ + optical_layers.__all__ + sources.__all__ + \
    gradient_energy.__all__
