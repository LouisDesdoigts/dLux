from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config

config.update("jax_debug_nans", True)

Array = np.ndarray


class TestCreateWavefront(object):
    """
    Tests the CreateWavefront class.
    """


    def test_constructor(self, create_create_wavefront: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_create_wavefront(diameter=np.array([]))

        # Test wrong string input
        with pytest.raises(AssertionError):
            create_create_wavefront(wavefront_type='cartesian')

        # Test functioning
        create_create_wavefront()


    def test_call(self, create_create_wavefront: callable) -> None:
        """
        Tests the __call__ method.
        """
        params = {'wavelength': np.array(1e-6), 'offset': np.zeros(2)}
        create_create_wavefront()(None, params)


class TestTiltWavefront(object):
    """
    Tests the TiltWavefront class.
    """


    def test_constructor(self, create_tilt_wavefront: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_tilt_wavefront(tilt_angles=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_tilt_wavefront(tilt_angles=np.array([]))

        # Test functioning
        create_tilt_wavefront()


    def test_call(self, 
            create_tilt_wavefront: callable,
            create_wavefront: callable) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        create_tilt_wavefront()(wf)


class TestNormaliseWavefront(object):
    """
    Tests the NormaliseWavefront class.
    """


    def test_constructor(self, create_normalise_wavefront: callable) -> None:
        """
        Tests the constructor.
        """
        # Test functioning
        create_normalise_wavefront()


    def test_call(self, 
            create_normalise_wavefront: callable,
            create_wavefront: callable) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        wf = create_normalise_wavefront()(wf)
        assert wf.psf.sum() == 1.


class TestApplyBasisOPD(object):
    """
    Tests the ApplyBasisOPD class.
    """


    def test_constructor(self, create_apply_basis_opd: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_apply_basis_opd(basis=np.ones((16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_apply_basis_opd(basis=np.ones((1, 1, 16, 16)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_apply_basis_opd(coefficients=np.array([]))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_apply_basis_opd(coefficients=np.zeros((1, 1)))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_apply_basis_opd(basis=np.ones((2, 15, 15)),
                                   coefficients=np.zeros((3)))

        # Test functioning
        create_apply_basis_opd()


    def test_call(self, 
            create_apply_basis_opd: callable, 
            create_wavefront: callable) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        create_apply_basis_opd()(wf)


class TestAddPhase(object):
    """
    Tests the AddPhase class.
    """


    def test_constructor(self, create_add_phase: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_add_phase(phase=np.ones(1))

        # Test functioning
        create_add_phase()


    def test_call(self, 
            create_add_phase: callable, 
            create_wavefront: callable) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        npix = wf.npixels

        # Test 0d
        create_add_phase(phase=np.array(1.))(wf)

        # Test 2d
        create_add_phase(phase=np.ones((npix, npix)))(wf)

        # Test 3d
        create_add_phase(phase=np.ones((1, npix, npix)))(wf)


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
