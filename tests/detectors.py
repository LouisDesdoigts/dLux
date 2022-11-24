from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)

Array = np.ndarray


class ApplyPixelResponseUtility(Utility):
    """
    Utility for ApplyPixelResponse class.
    """
    pixel_response : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ApplyPixelResponse Utility.
        """
        self.pixel_response = np.ones((16, 16))


    def construct(self           : Utility,
                  pixel_response : Array = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_response = self.pixel_response if pixel_response is None \
                                           else pixel_response
        return dLux.detectors.ApplyPixelResponse(pixel_response)


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


class ApplyJitterUtility(Utility):
    """
    Utility for ApplyJitter class.
    """
    sigma       : Array
    kernel_size : int


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ApplyJitter Utility.
        """
        self.sigma       = np.array(1.)
        self.kernel_size = 10


    def construct(self        : Utility,
                  sigma       : Array = None,
                  kernel_size : int   = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        sigma       = self.sigma       if sigma       is None else sigma
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        return dLux.detectors.ApplyJitter(sigma, kernel_size)


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


class ApplySaturationUtility(Utility):
    """
    Utility for ApplySaturation class.
    """
    saturation : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ApplySaturation Utility.
        """
        self.saturation = np.array(1.)


    def construct(self : Utility, saturation : Array = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        saturation = self.saturation if saturation is None else saturation
        return dLux.detectors.ApplySaturation(saturation)


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


class AddConstantUtility(Utility):
    """
    Utility for AddConstant class.
    """
    value : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AddConstant Utility.
        """
        self.value = np.array(1.)


    def construct(self : Utility, value : Array = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        value = self.value if value is None else value
        return dLux.detectors.AddConstant(value)


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


class IntegerDownsampleUtility(Utility):
    """
    Utility for IntegerDownsample class.
    """
    kernel_size : int


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the IntegerDownsample Utility.
        """
        self.kernel_size = 4


    def construct(self : Utility, kernel_size : int = None) -> DetectorLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        return dLux.detectors.IntegerDownsample(kernel_size)


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