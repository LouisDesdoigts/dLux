"""
tests/layers.py
---------------
This file contains the tests for the various layers that are offered 
by `dLux`. 

Concrete Classes
----------------
- TestCreateWavefront
- TestTiltWavefront
- TestCircularAperture
- TestNormaliseWavefront
- TestApplyBasicOPD
- TestAddPhase
- TestApplyOPD
- TestApplyAperture
- TestApplyBasisCLIMB
"""
__author__ = "Jordan Dennis"
__date__ = "05/07/2022"


import dLux
import typing
from utilities import *


Tester = typing.NewType("Tester", object)


class TestCreateWavefront(UtilityUser):
    """
    A `pytest` container for the `CreateWavefront` layer. Tests 
    the following methods.
    - __init__
    - __call__
    - _interact

    Attributes
    ----------
    utility : Utility
        A container for a safe constructor and other useful methods
    """
    utility : Utility = CreateWavefrontUtility()


    def test_constructor(self : Tester) -> None:
        """
        Tests that the state of the object is correctly instantiated 
        by the constructor. 
        """
        layer = self.get_utility().constuct()

        assert layer.number_of_pixels == \
            self.get_utility().get_number_of_pixels()
        assert layer.wavefront_size == \
            self.get_utility().get_wavefront_size()
        

    def test_interact(self : Tester) -> None:


class TestTiltWavefront(UtilityUser):
    """
    """
    pass

    
