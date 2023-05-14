from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


class TestCreateWavefront(object):
    """
    Tests the CreateWavefront class.
    """


    def test_constructor(self, create_create_wavefront) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_create_wavefront(diameter=np.array([]))

        # Test functioning
        create_create_wavefront()


    def test_call(self, create_create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        create_create_wavefront()(1e-6, np.array([0, 0]))


class TestTiltWavefront(object):
    """
    Tests the TiltWavefront class.
    """


    def test_constructor(self, create_tilt) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_tilt(angles=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_tilt(angles=np.array([]))

        # Test functioning
        create_tilt()


    def test_call(self, 
            create_tilt,
            create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        create_tilt()(wf)


class TestNormaliseWavefront(object):
    """
    Tests the NormaliseWavefront class.
    """


    def test_constructor(self, create_normalise_wavefront) -> None:
        """
        Tests the constructor.
        """
        # Test functioning
        create_normalise_wavefront()


    def test_call(self, 
            create_normalise_wavefront,
            create_wavefront) -> None:
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


    def test_constructor(self, create_apply_basis_opd) -> None:
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
            create_apply_basis_opd, 
            create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        create_apply_basis_opd()(wf)


class TestAddPhase(object):
    """
    Tests the AddPhase class.
    """


    def test_constructor(self, create_add_phase) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_add_phase(phase=np.ones(1))

        # Test functioning
        create_add_phase()


    def test_call(self, 
            create_add_phase, 
            create_wavefront) -> None:
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


class TestAddOPD(object):
    """
    Tests the AddOPD class.
    """


    def test_constructor(self, create_add_opd) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_add_opd(opd=np.ones(1))

        # Test functioning
        create_add_opd()


    def test_call(self, 
            create_add_opd, 
            create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        npix = wf.npixels

        # Test 0d
        create_add_opd(opd=np.array(1.))(wf)

        # Test 2d
        create_add_opd(opd=np.ones((npix, npix)))(wf)

        # Test 3d
        create_add_opd(opd=np.ones((1, npix, npix)))(wf)


class TestTransmissiveOptic(object):
    """
    Tests the TransmissiveOptic class.
    """


    def test_constructor(self, create_transmissive_optic) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_transmissive_optic(trans=np.ones(1))

        # Test functioning
        create_transmissive_optic()


    def test_call(self, 
            create_transmissive_optic, 
            create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        wf = create_wavefront() 
        npix = wf.npixels

        # Test 0d
        create_transmissive_optic(trans=np.array(1.))(wf)

        # Test 2d
        create_transmissive_optic(trans=np.ones((npix, npix)))(wf)

        # Test 3d
        create_transmissive_optic(trans=np.ones((1, npix, npix)))(wf)


# class TestApplyBasisCLIMB(object):
#     """
#     Tests the ApplyBasisCLIMB class.
#     """


#     def test_constructor(self, create_basis_climb) -> None:
#         """
#         Tests the constructor.
#         """
#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(basis=np.ones((16, 16)))

#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(basis=np.ones((1, 1, 16, 16)))

#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(ideal_wavelength=np.ones(1))

#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(coefficients=np.array([]))

#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(coefficients=np.zeros((1, 1)))

#         # Test wrong dims
#         with pytest.raises(AssertionError):
#             create_basis_climb(basis=np.ones((2, 15, 15)),
#                                    coefficients=np.zeros((3)))

#         # Test functioning
#         create_basis_climb()


#     def test_call(self, 
#             create_basis_climb, 
#             create_wavefront) -> None:
#         """
#         Tests the __call__ method.
#         """
#         npix = 256
#         wf = create_wavefront(npixels = npix) 
#         basis = np.ones((3, 3*npix, 3*npix))
#         create_basis_climb(basis=basis)(wf)


class TestRotate(object):
    """
    Tests the Rotate class.
    """


    def test_constructor(self, create_rotate) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_rotate(angle=np.ones(1))

        # Test functioning
        create_rotate()


    def test_call(self, 
            create_rotate, 
            create_wavefront) -> None:
        """
        Tests the __call__ method.
        """
        # Test regular rotation
        wf = create_wavefront()
        create_rotate()(wf)

        # Test real imaginary rotation
        wf = create_wavefront()
        create_rotate(real_imaginary=True)(wf)

        # Test fourier
        with pytest.raises(NotImplementedError):
            wf = create_wavefront()
            create_rotate(fourier=True)(wf)
