import jax.numpy as np
import pytest
from jax import config


config.update("jax_debug_nans", True)


def _test_transmissive_layer(constructor):
    """Tests the constructor of a transmissive layer."""
    constructor()


def _test_call_transmissive_layer(constructor, create_wavefront):
    """Tests the __call__ method of a transmissive layer."""
    wf = create_wavefront()
    constructor(normalise=True)(wf)
    constructor(normalise=False)(wf)


def _test_applied_shape(constructor):
    """Tests the applied_shape method of a shaped layer."""
    constructor().applied_shape


def _test_basis_layer_constructor(constructor):
    """Tests the constructor of a basis layer."""
    constructor()
    constructor(basis=None)
    constructor(coefficients=None)
    with pytest.raises(ValueError):
        constructor(basis=np.ones((2, 1, 1)), coefficients=np.ones(1))
    with pytest.raises(ValueError):
        constructor(basis=[(2, 1, 1)], coefficients=np.ones(1))


def _test_base_transmissive_optic_constructor(constructor):
    """Tests the constructor of a base transmissive optic."""
    constructor()
    constructor(transmission=None)


def _test_base_opd_optic_constructor(constructor):
    """Tests the constructor of a base opd optic."""
    constructor()
    constructor(opd=None)


def _test_base_phase_optic_constructor(constructor):
    """Tests the constructor of a base phase optic."""
    constructor()
    constructor(phase=None)


def _test_base_basis_optic_constructor(constructor):
    """Tests the constructor of a base basis optic."""
    constructor()
    with pytest.raises(ValueError):
        constructor(basis=np.ones((2, 1, 1)), coefficients=np.ones(1))


class TestOptic:
    """Tests the Optic class."""

    def test_constructor(self, create_optic):
        """Tests the constructor."""
        _test_base_transmissive_optic_constructor(create_optic)
        _test_base_opd_optic_constructor(create_optic)
        with pytest.raises(ValueError):
            create_optic(transmission=np.ones(1), opd=np.ones(2))

    def test_call(self, create_optic, create_wavefront):
        """Tests the __call__ method."""
        _test_call_transmissive_layer(create_optic, create_wavefront)

    def test_applied_shape(self, create_optic):
        """Tests the applied_shape method."""
        _test_applied_shape(create_optic)


class TestPhaseOptic:
    """Tests the Optic class."""

    def test_constructor(self, create_phase_optic):
        """Tests the constructor."""
        _test_base_transmissive_optic_constructor(create_phase_optic)
        _test_base_phase_optic_constructor(create_phase_optic)
        with pytest.raises(ValueError):
            create_phase_optic(transmission=np.ones(1), phase=np.ones(2))

    def test_call(self, create_phase_optic, create_wavefront):
        """Tests the __call__ method."""
        _test_call_transmissive_layer(create_phase_optic, create_wavefront)

    def test_applied_shape(self, create_phase_optic):
        """Tests the applied_shape method."""
        _test_applied_shape(create_phase_optic)


class TestBasisOptic:
    """Tests the BasisOptics class."""

    def test_constructor(self, create_basis_optic):
        """Tests the constructor."""
        _test_base_transmissive_optic_constructor(create_basis_optic)
        _test_base_basis_optic_constructor(create_basis_optic)

    def test_call(self, create_basis_optic, create_wavefront):
        """Tests the __call__ method."""
        _test_call_transmissive_layer(create_basis_optic, create_wavefront)

    def test_applied_shape(self, create_basis_optic):
        """Tests the applied_shape method."""
        _test_applied_shape(create_basis_optic)


class TestPhaseBasisOptic:
    """Tests the PhaseBasisOptics class."""

    def test_constructor(self, create_phase_basis_optic):
        """Tests the constructor."""
        _test_base_transmissive_optic_constructor(create_phase_basis_optic)
        _test_base_basis_optic_constructor(create_phase_basis_optic)

    def test_call(self, create_phase_basis_optic, create_wavefront):
        """Tests the __call__ method."""
        _test_call_transmissive_layer(
            create_phase_basis_optic, create_wavefront
        )

    def test_applied_shape(self, create_phase_basis_optic):
        """Tests the applied_shape method."""
        _test_applied_shape(create_phase_basis_optic)


class TestTilt:
    """Tests the Tilt class."""

    def test_constructor(self, create_tilt):
        """Tests the constructor."""
        create_tilt()
        with pytest.raises(ValueError):
            create_tilt(angles=np.ones(1))

    def test_call(self, create_tilt, create_wavefront):
        """Tests the __call__ method."""
        create_tilt()(create_wavefront())


class TestNormalise:
    """Tests the Normalise class."""

    def test_call(self, create_normalise, create_wavefront):
        """Tests the __call__ method."""
        create_normalise()(create_wavefront())


class TestRotate:
    """Tests the Rotate class."""

    def test_constructor(self, create_rotate):
        """Tests the constructor."""
        create_rotate()
        with pytest.raises(ValueError):
            create_rotate(angle=np.ones(1))
        with pytest.raises(ValueError):
            create_rotate(order=2)

    def test_call(self, create_rotate, create_wavefront):
        """Tests the __call__ method."""
        wf = create_wavefront()
        create_rotate()(wf)
        create_rotate(complex=True)(wf)


class TestFlip:
    """Tests the Flip class."""

    def test_constructor(self, create_flip):
        """Tests the constructor."""
        create_flip()
        create_flip((0, 1, 2))
        with pytest.raises(ValueError):
            create_flip(axes="1")
        with pytest.raises(ValueError):
            create_flip(axes=1.0)
        with pytest.raises(ValueError):
            create_flip(axes=(1, 1.0))

    def test_call(self, create_flip, create_wavefront):
        """Tests the __call__ method."""
        wf = create_wavefront()
        create_flip()(wf)


class TestRezise:
    """Tests the Resize class."""

    def test_constructor(self, create_resize):
        """Tests the constructor."""
        create_resize()

    def test_call(self, create_resize, create_wavefront):
        """Tests the __call__ method."""

        wf = create_wavefront(npixels=10)
        create_resize(npixels=16)(wf)

        wf = create_wavefront(npixels=20)
        create_resize(npixels=16)(wf)

        wf = create_wavefront(npixels=16)
        create_resize(npixels=16)(wf)
