from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)

Array = np.ndarray


class TestApplyPixelResponse(UtilityUser):
    """
    Tests the ApplyPixelResponse class.
    """
    utility : ApplyPixelResponseUtility = ApplyPixelResponseUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(pixel_response=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(pixel_response=np.array(1.))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(pixel_response=np.ones((1, 1, 1)))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class TestApplyJitter(UtilityUser):
    """
    Tests the ApplyJitter class.
    """
    utility : ApplyJitterUtility = ApplyJitterUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(sigma=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class TestApplySaturation(UtilityUser):
    """
    Tests the ApplySaturation class.
    """
    utility : ApplySaturationUtility = ApplySaturationUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(saturation=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class TestAddConstant(UtilityUser):
    """
    Tests the AddConstant class.
    """
    utility : AddConstantUtility = AddConstantUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(value=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class TestIntegerDownsample(UtilityUser):
    """
    Tests the IntegerDownsample class.
    """
    utility : IntegerDownsampleUtility = IntegerDownsampleUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class RotateUtility(Utility):
    """
    Utility for Rotate class.
    """
    angle          : Array
    fourier        : bool
    padding        : int


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Rotate Utility.
        """
        self.angle          = np.array(np.pi)
        self.fourier        = False
        self.padding        = 2


    def construct(self           : Utility, 
                  angle          : Array = None,
                  fourier        : bool  = None,
                  padding        : bool  = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        angle   = self.angle   if angle   is None else angle
        fourier = self.fourier if fourier is None else fourier
        padding = self.padding if padding is None else padding
        return dLux.detectors.Rotate(angle, fourier, padding)


class TestRotate(UtilityUser):
    """
    Tests the Rotate class.
    """
    utility : RotateUtility = RotateUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(angle=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        # Test regular rotation
        image = np.ones((16, 16))
        self.utility.construct()(image)

        # Test fourier
        with pytest.raises(NotImplementedError):
            self.utility.construct(fourier=True)(image)
