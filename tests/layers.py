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


# NOTE: While the utilities system proved very useful for testing 
# The more complicated wavefronts and propagators I can see that
# It is going to produce a very large amount of code when used 
# for the layers. I should be anle to reverse engineer this 
# to a junit style for better efficiency.
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
    def test_call(self : Tester) -> None:


class TestCircularAperture(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_create_apperture(self : Tester) -> None:
    def test_call(self : Tester) -> None:


class TestNormaliseWavefront(UtilityUser):
    def test_call(self : Tester) -> None:


class TestApplyBasicOPD(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_get_total_opd(self : Tester) -> None:
    def test_call(self : Tester) -> None:


class TestAddPhase(UtiltiyUser):
    def test_constructor(self : Tester) -> None:
    def test_call(self : Tester) -> None:


class TestApplyOPD(UtilityUser): 
    def test_constructor(self : Tester) -> None:
    def test_call(self : Tester) -> None:


class TestApplyAperture(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_call(self : Tester) -> None:
