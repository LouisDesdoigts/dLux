name = "toliman"

# Import as modules
from . import optics
from . import sources
# from . import gradient_energy

# Dont import all functions from modules
from .optics          import *
from .sources         import *
# from .gradient_energy import *

# Add to __all__
__all__ = optics.__all__ + sources.__all__ #+ gradient_energy.__all__
