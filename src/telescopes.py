"""
src/telescopes.py
-----------------
Pre-built space telescope models.
"""
__author__ = "Louis Desdoigts"
__author__ = "Jordan Dennis"
__date__ = "05/07/2022"


from dLux import *
from typing import NewType


Telescope = NewType("Telescope", OpticalSystem)


class Toliman(OpticalSystem):
    """
    """
    def __init__(self : Telescope):   
