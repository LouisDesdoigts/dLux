from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
from wavefronts import WavefrontUtility
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)

Array = np.ndarray


class CreateWavefrontUtility(Utility):
    """
    Utility for CreateWavefront class.
    """
    npixels        : int
    diameter       : Array
    wavefront_type : str


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CreateWavefront Utility.
        """
        self.npixels = 16
        self.diameter = np.array(1.)
        self.wavefront_type = "Cartesian"


    def construct(self            : Utility,
                  npixels         : int   = None,
                  diameter        : Array = None,
                  wavefront_type  : str   = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        npixels  = self.npixels  if npixels  is None else npixels
        diameter = self.diameter if diameter is None else diameter
        wavefront_type = self.wavefront_type if wavefront_type is None else \
        wavefront_type
        return dLux.optics.CreateWavefront(npixels, diameter, wavefront_type)


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


class TiltWavefrontUtility(Utility):
    """
    Utility for TiltWavefront class.
    """
    tilt_angles : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the TiltWavefront Utility.
        """
        self.tilt_angles = np.ones(2)


    def construct(self : Utility, tilt_angles : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        tilt_angles = self.tilt_angles if tilt_angles is None else tilt_angles
        return dLux.optics.TiltWavefront(tilt_angles)


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


class CircularApertureUtility(Utility):
    """
    Utility for CircularAperture class.
    """
    npixels : int
    rmin    : float
    rmax    : float


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CircularAperture Utility.
        """
        self.npixels = 16
        self.rmin    = 0.05
        self.rmax    = 0.95


    def construct(self    : Utility,
                  npixels : int   = None,
                  rmin    : float = None,
                  rmax    : float = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        npixels = self.npixels if npixels is None else npixels
        rmin    = self.rmin    if rmin    is None else rmin
        rmax    = self.rmax    if rmax    is None else rmax
        return dLux.optics.CircularAperture(npixels, rmin, rmax)


class TestCircularAperture(UtilityUser):
    """
    Tests the CircularAperture class.
    """
    utility : CircularApertureUtility = CircularApertureUtility()


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
        self.utility.construct(npixels=wf.npixels)(wf)


class NormaliseWavefrontUtility(Utility):
    """
    Utility for NormaliseWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the NormaliseWavefront Utility.
        """
        pass


    def construct(self : Utility) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.optics.NormaliseWavefront()


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


class ApplyBasisOPDUtility(Utility):
    """
    Utility for ApplyBasisOPD class.
    """
    basis        : Array
    coefficients : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ApplyBasisOPD Utility.
        """
        self.basis        = np.ones((3, 16, 16))
        self.coefficients = np.ones(3)


    def construct(self         : Utility,
                  basis        : Array = None,
                  coefficients : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        basis        = self.basis        if basis        is None else basis
        coefficients = self.coefficients if coefficients is None \
                                         else coefficients
        return dLux.optics.ApplyBasisOPD(basis, coefficients)


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


class AddPhaseUtility(Utility):
    """
    Utility for AddPhase class.
    """
    phase : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AddPhase Utility.
        """
        self.phase = np.ones((16, 16))


    def construct(self : Utility, phase : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        phase = self.phase if phase is None else phase
        return dLux.optics.AddPhase(phase)


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


class AddOPDUtility(Utility):
    """
    Utility for AddOPD class.
    """
    opd : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AddOPD Utility.
        """
        self.opd = np.ones((16, 16))


    def construct(self : Utility, opd : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        opd = self.opd if opd is None else opd
        return dLux.optics.AddOPD(opd)


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


class TransmissiveOpticUtility(Utility):
    """
    Utility for TransmissiveOptic class.
    """
    transmission : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the TransmissiveOptic Utility.
        """
        self.transmission = np.ones((16, 16))


    def construct(self : Utility, transmission : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        transmission = self.transmission if transmission is None \
                                         else transmission
        return dLux.optics.TransmissiveOptic(transmission)


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


class CompoundApertureUtility(Utility):
    """
    Utility for CompoundAperture class.
    """
    aperture_radii  : Array
    aperture_coords : Array
    occulter_radii  : Array
    occulter_coords : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CompoundAperture Utility.
        """
        self.aperture_radii  = np.ones(2)
        self.aperture_coords = np.array([( 0.1,  0.1), (-0.1, -0.1)])
        self.occulter_radii  = 0.1*np.ones(2)
        self.occulter_coords = np.array([( 0.1,  0.1), (-0.1, -0.1)])


    def construct(self                  : Utility, 
                  aperture_radii        : Array = None,
                  aperture_coords       : Array = None,
                  occulter_radii        : Array = None,
                  occulter_coords       : Array = None,
                  input_aperture_coords : bool  = True,
                  input_occulter_radii  : bool  = True,
                  input_occulter_coords : bool  = True,) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        aperture_radii  = self.aperture_radii  if   aperture_radii  is None \
                                               else aperture_radii
        aperture_coords = self.aperture_coords if   aperture_coords is None \
                                               else aperture_coords
        occulter_radii  = self.occulter_radii  if   occulter_radii  is None \
                                               else occulter_radii
        occulter_coords = self.occulter_coords if   occulter_coords is None \
                                               else occulter_coords

        args = [aperture_radii]
        args = args + [aperture_coords] if input_aperture_coords else args
        args = args + [occulter_radii]  if input_occulter_radii  else args
        args = args + [occulter_coords] if input_occulter_coords else args
        return dLux.optics.CompoundAperture(*args)


class TestCompoundAperture(UtilityUser):
    """
    Tests the CompoundAperture class.
    """
    utility : CompoundApertureUtility = CompoundApertureUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(aperture_radii=np.ones((1, 1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(occulter_radii=np.ones((1, 1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(aperture_coords=np.ones((1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(aperture_radii=np.ones((1)),
                                   aperture_coords=np.ones((2, 2)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(occulter_coords=np.ones((2, 3)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(occulter_coords=np.ones((1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(occulter_radii=np.ones((1)),
                                   occulter_coords=np.ones((2, 2)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(occulter_coords=np.ones((2, 3)))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        wf = WavefrontUtility().construct()

        # Test 0d
        self.utility.construct()(wf)


class ApplyBasisCLIMBUtility(Utility):
    """
    Utility for ApplyBasisCLIMB class.
    """
    basis            : Array
    coefficients     : Array
    ideal_wavelength : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ApplyBasisCLIMB Utility.
        """
        self.basis            = np.ones((3, 16, 16))
        self.coefficients     = np.ones(3)
        self.ideal_wavelength = np.array(5e-7)


    def construct(self             : Utility, 
                  basis            : Array = None,
                  coefficients     : Array = None,
                  ideal_wavelength : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        basis = self.basis if basis is None else basis
        ideal_wavelength = self.ideal_wavelength if ideal_wavelength is None \
                                               else ideal_wavelength
        coefficients = self.coefficients if coefficients is None \
                                       else coefficients
        return dLux.optics.ApplyBasisCLIMB(basis, ideal_wavelength, 
                                           coefficients)


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