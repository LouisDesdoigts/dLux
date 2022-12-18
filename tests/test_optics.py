from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config

config.update("jax_debug_nans", True)

Array = np.ndarray

class TestCreateWavefront(UtilityUser):
    """
    Tests the CreateWavefront class.
    """
    utility : CreateWavefrontUtility = CreateWavefrontUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(diameter=np.array([]))

        # Test wrong string input
        with pytest.raises(AssertionError):
            self.utility.construct(wavefront_type='cartesian')

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        params = {'wavelength': np.array(1e-6), 'offset': np.zeros(2)}
        self.utility.construct()(None, params)


class TestTiltWavefront(UtilityUser):
    """
    Tests the TiltWavefront class.
    """
    utility : TiltWavefrontUtility = TiltWavefrontUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(tilt_angles=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(tilt_angles=np.array([]))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        self.utility.construct()(wf)


class TestNormaliseWavefront(UtilityUser):
    """
    Tests the NormaliseWavefront class.
    """
    utility : NormaliseWavefrontUtility = NormaliseWavefrontUtility()


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
        wf = WavefrontUtility().construct()
        wf = self.utility.construct()(wf)
        assert wf.psf.sum() == 1.


class TestApplyBasisOPD(UtilityUser):
    """
    Tests the ApplyBasisOPD class.
    """
    utility : ApplyBasisOPDUtility = ApplyBasisOPDUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((1, 1, 16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=np.array([]))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=np.zeros((1, 1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((2, 15, 15)),
                                   coefficients=np.zeros((3)))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        self.utility.construct()(wf)


class TestAddPhase(UtilityUser):
    """
    Tests the AddPhase class.
    """
    utility : AddPhaseUtility = AddPhaseUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(phase=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        npix = wf.npixels

        # Test 0d
        self.utility.construct(phase=np.array(1.))(wf)

        # Test 2d
        self.utility.construct(phase=np.ones((npix, npix)))(wf)

        # Test 3d
        self.utility.construct(phase=np.ones((1, npix, npix)))(wf)


class TestAddOPD(UtilityUser):
    """
    Tests the AddOPD class.
    """
    utility : AddOPDUtility = AddOPDUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(opd=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        npix = wf.npixels

        # Test 0d
        self.utility.construct(opd=np.array(1.))(wf)

        # Test 2d
        self.utility.construct(opd=np.ones((npix, npix)))(wf)

        # Test 3d
        self.utility.construct(opd=np.ones((1, npix, npix)))(wf)


class TestTransmissiveOptic(UtilityUser):
    """
    Tests the TransmissiveOptic class.
    """
    utility : TransmissiveOpticUtility = TransmissiveOpticUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(transmission=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        npix = wf.npixels

        # Test 0d
        self.utility.construct(transmission=np.array(1.))(wf)

        # Test 2d
        self.utility.construct(transmission=np.ones((npix, npix)))(wf)

        # Test 3d
        self.utility.construct(transmission=np.ones((1, npix, npix)))(wf)


class TestApplyBasisCLIMB(UtilityUser):
    """
    Tests the ApplyBasisCLIMB class.
    """
    utility : ApplyBasisCLIMBUtility = ApplyBasisCLIMBUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((1, 1, 16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(ideal_wavelength=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=np.array([]))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=np.zeros((1, 1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(basis=np.ones((2, 15, 15)),
                                   coefficients=np.zeros((3)))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()
        npix = wf.npixels
        basis = np.ones((3, 3*npix, 3*npix))
        self.utility.construct(basis=basis)(wf)


class RotateUtility(Utility):
    """
    Utility for Rotate class.
    """
    angle          : Array
    real_imaginary : bool
    fourier        : bool
    padding        : int


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Rotate Utility.
        """
        self.angle          = np.array(np.pi)
        self.real_imaginary = False
        self.fourier        = False
        self.padding        = 2


    def construct(self           : Utility,
                  angle          : Array = None,
                  real_imaginary : Array = None,
                  fourier        : bool  = None,
                  padding        : bool  = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        angle   = self.angle   if angle   is None else angle
        fourier = self.fourier if fourier is None else fourier
        padding = self.padding if padding is None else padding
        real_imaginary = self.real_imaginary if real_imaginary is None \
                                           else real_imaginary
        return dLux.optics.Rotate(angle, real_imaginary, fourier, padding)


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
        wf = WavefrontUtility().construct()
        self.utility.construct()(wf)

        # Test real imaginary rotation
        wf = WavefrontUtility().construct()
        self.utility.construct(real_imaginary=True)(wf)

        # Test fourier
        with pytest.raises(NotImplementedError):
            wf = WavefrontUtility().construct()
            self.utility.construct(fourier=True)(wf)
