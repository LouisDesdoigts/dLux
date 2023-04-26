name = "toliman"

# Import as modules
from . import optics
from . import sources

# Dont import all functions from modules
from .optics     import *
from .sources     import *

# Add to __all__
__all__ = optics.__all__ + sources.__all__
