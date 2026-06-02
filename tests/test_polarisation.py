from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.wavefronts import PolarisedWavefront
from dLux.layers.polarised import PolarisingOptic
from dLux.layers.polarised import jones_matrix_rotated

n_pix = 16


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
    return PolarisedWavefront(
        npixels=n_pix,
        diameter=1.0,
        wavelength=1e-6,
        initial_stokes=np.array([1.0, 1.0, 0.0, 0.0]),
    )


class TestPolarisedWavefront:
    def test_constructor(self, unpol_wavefront):
        assert unpol_wavefront.npixels == n_pix
        assert unpol_wavefront.diameter == 1.0
        assert unpol_wavefront.wavelength == 1e-6
        assert unpol_wavefront.initial_stokes.shape == (4,)

        assert unpol_wavefront.phasor.shape == (2, 2, n_pix, n_pix)

    def test_from_wavefront(self, unpol_wavefront):
        from dLux.wavefronts import Wavefront

        wf = Wavefront(
            npixels=n_pix,
            diameter=1.0,
            wavelength=1e-6,
        )
        pol_wf = PolarisedWavefront.from_wavefront(wf)
        assert isinstance(pol_wf, PolarisedWavefront)
        assert np.allclose(pol_wf.initial_stokes, np.array([1.0, 0.0, 0.0, 0.0]))


def test_jones_rotation():
    # Test that a horizontal polariser rotated by 90 degrees becomes a vertical polariser
    init_jones = np.array([[1, 0], [0, 0]])
    angle = np.pi / 2
    rotated_jones = jones_matrix_rotated(init_jones, angle)
    expected_jones = np.array([[0, 0], [0, 1]])
    assert np.allclose(rotated_jones, expected_jones, atol=1e-6)


@pytest.fixture
def horizontal_polariser_jones():
    return np.array([[1, 0], [0, 0]])  # Horizontal polariser Jones matrix


@pytest.fixture
def vertical_polariser_jones():
    return np.array([[0, 0], [0, 1]])  # Vertical polariser Jones matrix


class TestPolarisingOptic:
    def test_create_from_jones(self, horizontal_polariser_jones):

        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        assert np.allclose(optic.jones_matrix, horizontal_polariser_jones)

    def test_apply_horizontal_polariser(
        self, horizontal_polariser_jones, pol_wavefront
    ):
        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        # The output wavefront should have only the horizontal component
        assert np.allclose(output_wavefront.phasor[0, 0], pol_wavefront.phasor[0, 0])
        assert np.allclose(output_wavefront.phasor[1, 1], 0.0)

    def test_apply_vertical_polariser(self, vertical_polariser_jones, pol_wavefront):
        optic = PolarisingOptic(jones_matrix=vertical_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        # The output wavefront should have only the vertical component
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[1, 1])

    def test_rotate_horizontal_to_vertical(
        self, horizontal_polariser_jones, pol_wavefront
    ):
        optic = PolarisingOptic(
            jones_matrix=horizontal_polariser_jones, angle=np.pi / 2
        )
        output_wavefront = optic(pol_wavefront)
        # The output wavefront should have only the vertical component
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[0, 0])

    def test_power_conservation(self, horizontal_polariser_jones, pol_wavefront):
        optic = PolarisingOptic(jones_matrix=horizontal_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        # The total power should be conserved (since it's an ideal polariser)
        input_power = np.sum(pol_wavefront.psf)
        output_power = np.sum(output_wavefront.psf)
        print(f"Input power: {input_power}, Output power: {output_power}")
        assert np.isclose(input_power, output_power, atol=1e-6)

        # Test that the power is halved for a 45 degree polariser
        optic_45 = PolarisingOptic(
            jones_matrix=horizontal_polariser_jones, angle=np.pi / 4
        )
        output_wavefront_45 = optic_45(pol_wavefront)
        output_power_45 = np.sum(output_wavefront_45.psf)
        assert np.isclose(output_power_45, input_power / 2, atol=1e-6)
