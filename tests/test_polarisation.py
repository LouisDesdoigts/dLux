from jax import numpy as np, config

config.update("jax_debug_nans", True)

import jax
import pytest
from dLux.wavefronts import PolarisedWavefront, Wavefront
from dLux.layers.polarised import PolarisingOptic, UniformPolarisingOptic
from dLux.layers.polarised import jones_matrix_rotated

# Import the abstract base tests to inherit standard functionality
from .test_wavefronts import BaseWavefrontTests

n_pix = 16

# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture
def basic_wavefront():
    return Wavefront(
        npixels=n_pix,
        diameter=1.0,
        wavelength=1e-6,
    )


@pytest.fixture
def unpol_wavefront():
    return PolarisedWavefront(
        npixels=n_pix,
        diameter=1.0,
        wavelength=1e-6,
        initial_stokes=np.array([1.0, 0.0, 0.0, 0.0]),
    )


@pytest.fixture
def pol_wavefront():
    # 100% Horizontally polarized light
    return PolarisedWavefront(
        npixels=n_pix,
        diameter=1.0,
        wavelength=1e-6,
        initial_stokes=np.array([1.0, 1.0, 0.0, 0.0]),
    )


@pytest.fixture
def horizontal_polariser_jones():
    return np.array([[1.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def vertical_polariser_jones():
    return np.array([[0.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def vertical_polariser_jones_4D(vertical_polariser_jones):
    return vertical_polariser_jones[:, :, np.newaxis, np.newaxis]


@pytest.fixture
def circular_polariser_jones():
    # Right circular polariser Jones matrix
    return np.array([[0.5, 0.5j], [-0.5j, 0.5]])


@pytest.fixture
def spatially_varying_liquid_crystal_optic():
    """Generates a spatially varying optic utilizing native Jones batching."""
    diameter = 1.0
    x = np.linspace(-diameter / 2, diameter / 2, n_pix)
    y = np.linspace(-diameter / 2, diameter / 2, n_pix)
    X, Y = np.meshgrid(x, y)
    angle = np.arctan2(Y, X)

    init_jones = np.array([[1.0, 0.0], [0.0, 0.0]])

    # NATIVE BATCHING: No vmaps, flattens, or complex transpositions required!
    jones_matrix = jones_matrix_rotated(init_jones, angle)

    return PolarisingOptic(jones_matrix=jones_matrix)


# ==========================================
# RUN INHERITED WAVEFRONT TESTS ON POLARISED
# ==========================================


class TestPolarisedWavefrontShared(BaseWavefrontTests):
    """Executes every single basic wavefront operation test against PolarisedWavefront."""

    @pytest.fixture
    def wavefront_cls(self):
        return PolarisedWavefront

    @pytest.fixture
    def wavefront_factory(self, wavefront_cls):
        def factory(**kwargs):
            # Inject unpolarized fallback state if not explicitly specified by standard tests
            if "initial_stokes" not in kwargs:
                kwargs["initial_stokes"] = np.array([1.0, 0.0, 0.0, 0.0])
            return wavefront_cls(**kwargs)

        return factory

    @pytest.fixture
    def wavefront(self, wavefront_factory):
        return wavefront_factory(npixels=16, diameter=1.0, wavelength=1e-6)


# ==========================================
# TEST CLASSES (POLARISATION SPECIFIC)
# ==========================================


class TestPolarisedWavefront:
    def test_constructor(self, unpol_wavefront):
        assert unpol_wavefront.npixels == n_pix
        assert unpol_wavefront.diameter == 1.0
        assert unpol_wavefront.wavelength == 1e-6
        assert unpol_wavefront.initial_stokes.shape == (4,)
        assert unpol_wavefront.phasor.shape == (2, 2, n_pix, n_pix)

    def test_from_wavefront(self, unpol_wavefront):
        from dLux.wavefronts import Wavefront

        wf = Wavefront(npixels=n_pix, diameter=1.0, wavelength=1e-6)
        pol_wf = PolarisedWavefront.from_wavefront(wf)
        assert isinstance(pol_wf, PolarisedWavefront)
        assert np.allclose(pol_wf.initial_stokes, np.array([1.0, 0.0, 0.0, 0.0]))


class TestPolarisingOptic:
    def test_create_from_jones(self, horizontal_polariser_jones):
        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        assert np.allclose(
            optic.jones_matrix, horizontal_polariser_jones[:, :, np.newaxis, np.newaxis]
        )

    def test_apply_horizontal_polariser(
        self, horizontal_polariser_jones, pol_wavefront
    ):
        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        assert np.allclose(output_wavefront.phasor[0, 0], pol_wavefront.phasor[0, 0])
        assert np.allclose(output_wavefront.phasor[1, 1], 0.0)

    def test_apply_vertical_polariser(self, vertical_polariser_jones, pol_wavefront):
        optic = PolarisingOptic(jones_matrix=vertical_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[1, 1])

    def test_apply_vertical_polariser_4D(
        self, vertical_polariser_jones_4D, pol_wavefront
    ):
        optic = PolarisingOptic(jones_matrix=vertical_polariser_jones_4D)
        output_wavefront = optic(pol_wavefront)
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[1, 1])

    def test_rotate_horizontal_to_vertical(
        self, horizontal_polariser_jones, pol_wavefront
    ):
        optic = UniformPolarisingOptic(
            jones_matrix=horizontal_polariser_jones, angle=np.pi / 2
        )
        output_wavefront = optic(pol_wavefront)
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[0, 0])

    def test_transmission_and_malus_law(
        self, horizontal_polariser_jones, pol_wavefront
    ):
        """Tests polariser behavior against Malus' Law (State-dependent attenuation)."""
        optic = UniformPolarisingOptic(jones_matrix=horizontal_polariser_jones)
        output_wavefront = optic(pol_wavefront)

        input_power = np.sum(pol_wavefront.psf)
        output_power = np.sum(output_wavefront.psf)
        assert np.isclose(input_power, output_power, atol=1e-6)

        # Test Malus' Law: Power should drop by half for a 45-degree rotation
        optic_45 = UniformPolarisingOptic(
            jones_matrix=horizontal_polariser_jones, angle=np.pi / 4
        )
        output_wavefront_45 = optic_45(pol_wavefront)
        output_power_45 = np.sum(output_wavefront_45.psf)
        assert np.isclose(output_power_45, input_power / 2, atol=1e-6)

    def test_circular_polariser_iau_convention(
        self, circular_polariser_jones, unpol_wavefront
    ):
        """Verifies circular polarisation matches the IAU standard convention."""
        optic = UniformPolarisingOptic(jones_matrix=circular_polariser_jones)
        output_wavefront = optic(unpol_wavefront)

        # Ensure no linear polarization remains residual
        assert np.allclose(output_wavefront.stokes[1], 0.0, atol=1e-6)
        assert np.allclose(output_wavefront.stokes[2], 0.0, atol=1e-6)

        # IAU convention alignment check: Right-circular states must produce +V
        assert np.allclose(
            output_wavefront.stokes[3], output_wavefront.stokes[0], atol=1e-6
        )

    def test_spatially_varying_optic(
        self, spatially_varying_liquid_crystal_optic, unpol_wavefront
    ):
        optic = spatially_varying_liquid_crystal_optic
        output_wavefront = optic(unpol_wavefront)

        x = np.linspace(
            -unpol_wavefront.diameter / 2,
            unpol_wavefront.diameter / 2,
            unpol_wavefront.npixels,
        )
        y = np.linspace(
            -unpol_wavefront.diameter / 2,
            unpol_wavefront.diameter / 2,
            unpol_wavefront.npixels,
        )
        X, Y = np.meshgrid(x, y)
        expected_angle = np.arctan2(Y, X)

        expected_Q = np.cos(2 * expected_angle)
        expected_U = np.sin(2 * expected_angle)

        normalized_Q = output_wavefront.stokes[1] / output_wavefront.stokes[0]
        normalized_U = output_wavefront.stokes[2] / output_wavefront.stokes[0]

        assert np.allclose(normalized_Q, expected_Q, atol=1e-6)
        assert np.allclose(normalized_U, expected_U, atol=1e-6)

    def test_wf_promotion(self, horizontal_polariser_jones, basic_wavefront):
        """Tests that a non-polarised wavefront is correctly promoted to polarised when passed through the optic."""
        basic_wavefront /= np.sqrt(basic_wavefront.psf)  # unity everywhere
        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        output_wavefront = optic(basic_wavefront)

        print(basic_wavefront.phasor)

        assert isinstance(output_wavefront, PolarisedWavefront)
        assert np.allclose(
            output_wavefront.stokes[0], 0.5, atol=1e-6
        )  # Total intensity should be preserved
        assert np.allclose(
            output_wavefront.stokes[1], 0.5, atol=1e-6
        )  # Should become horizontally polarized
        assert np.allclose(output_wavefront.stokes[2], 0.0, atol=1e-6)  # No U component
        assert np.allclose(output_wavefront.stokes[3], 0.0, atol=1e-6)  # No V component


def test_jones_rotation_batching():
    init_jones = np.array([[1.0, 0.0], [0.0, 0.0]])
    angle = np.pi / 2

    # Simple 2D test
    rotated_jones = jones_matrix_rotated(init_jones, angle)
    assert np.allclose(rotated_jones, np.array([[0.0, 0.0], [0.0, 1.0]]), atol=1e-6)

    # Multi-dimensional batch test (2, 2, 3, 4)
    batched_angles = np.ones((3, 4)) * (np.pi / 2)
    rotated_batched = jones_matrix_rotated(init_jones, batched_angles)
    assert rotated_batched.shape == (2, 2, 3, 4)


def test_jax_jit_and_grad_compatibility(horizontal_polariser_jones, pol_wavefront):
    """Ensures that executing and differentiating the optic works seamlessly within JAX."""

    # 1. Test JIT compilation compatibility
    @jax.jit
    def jitted_execution(angle):
        # We pass a dynamic angle to evaluate conditional-branch robustness
        optic = UniformPolarisingOptic(
            jones_matrix=horizontal_polariser_jones, angle=angle
        )
        out_wf = optic(pol_wavefront)
        return np.sum(out_wf.psf)

    # If your code contains static 'if' checks on dynamic parameters, this will fail here:
    try:
        power = jitted_execution(np.array(np.pi / 4))
    except jax.errors.ConcretizationTypeError as e:
        pytest.fail(f"JIT Compilation failed due to dynamic tracer checking: {e}")

    # 2. Test Gradient/Differentiability compatibility
    # Ensure we can compute derivatives with respect to optical alignments
    grad_fn = jax.grad(jitted_execution)

    try:
        gradient = grad_fn(np.array(np.pi / 4))
        assert not np.isnan(gradient)
    except Exception as e:
        pytest.fail(f"Differentiability check failed: {e}")
