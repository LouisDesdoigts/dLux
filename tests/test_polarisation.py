from jax import numpy as np, config

config.update("jax_debug_nans", True)
import jax
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

    # test for case where jones is already 4D
    init_jones_4D = init_jones[:, :, np.newaxis, np.newaxis]
    rotated_jones_4D = jones_matrix_rotated(init_jones_4D, angle)
    assert np.allclose(
        rotated_jones_4D, expected_jones[:, :, np.newaxis, np.newaxis], atol=1e-6
    )


@pytest.fixture
def horizontal_polariser_jones():
    return np.array([[1, 0], [0, 0]])  # Horizontal polariser Jones matrix


@pytest.fixture
def vertical_polariser_jones():
    return np.array([[0, 0], [0, 1]])  # Vertical polariser Jones matrix


@pytest.fixture
def vertical_polariser_jones_4D():
    x = np.array([[0, 0], [0, 1]])  # Vertical polariser Jones matrix
    return x[:, :, np.newaxis, np.newaxis]


# test circular polarisation too
@pytest.fixture
def circular_polariser_jones():
    return np.array(
        [[0.5, 0.5j], [-0.5j, 0.5]]
    )  # Right circular polariser Jones matrix


@pytest.fixture
def spatially_varying_liquid_crystal_optic():
    # Create a spatially varying liquid crystal optic that creates a radial polarisation pattern
    npixels = n_pix
    diameter = 1.0
    wavelength = 1e-6
    x = np.linspace(-diameter / 2, diameter / 2, npixels)
    y = np.linspace(-diameter / 2, diameter / 2, npixels)
    X, Y = np.meshgrid(x, y)
    angle = np.arctan2(Y, X)  # Angle of the optic varies with position
    init_jones = np.array([[1, 0], [0, 0]])  # Start with a horizontal polariser

    # vmap over x,y to create a spatially varying Jones matrix
    jones_matrix = (
        jax.vmap(lambda a: jones_matrix_rotated(init_jones, a), in_axes=0)(
            angle.flatten()
        )
        .reshape((npixels, npixels, 2, 2))
        .transpose(2, 3, 0, 1)
    )

    print(f"Jones matrix shape: {jones_matrix.shape}")
    return PolarisingOptic(jones_matrix=jones_matrix)


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
        # The output wavefront should have only the horizontal component
        assert np.allclose(output_wavefront.phasor[0, 0], pol_wavefront.phasor[0, 0])
        assert np.allclose(output_wavefront.phasor[1, 1], 0.0)

    def test_apply_vertical_polariser(self, vertical_polariser_jones, pol_wavefront):
        optic = PolarisingOptic(jones_matrix=vertical_polariser_jones)
        output_wavefront = optic(pol_wavefront)
        # The output wavefront should have only the vertical component
        assert np.allclose(output_wavefront.phasor[0, 0], 0.0)
        assert np.allclose(output_wavefront.phasor[1, 1], pol_wavefront.phasor[1, 1])

    def test_apply_vertical_polariser_4D(
        self, vertical_polariser_jones_4D, pol_wavefront
    ):
        # the 4D test just makes sure a (2,2,1,1) Jones matrix works the same way as a (2,2) Jones matrix
        optic = PolarisingOptic(jones_matrix=vertical_polariser_jones_4D)
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

    def test_circular_polariser(self, circular_polariser_jones, unpol_wavefront):
        optic = PolarisingOptic(jones_matrix=circular_polariser_jones)
        output_wavefront = optic(unpol_wavefront)
        # The output wavefront should have circular polarisation
        assert np.allclose(
            output_wavefront.stokes[1], 0.0, atol=1e-6
        )  # No linear polarisation
        assert np.allclose(
            output_wavefront.stokes[2], 0.0, atol=1e-6
        )  # No linear polarisation

        # check circular polarisation by checking that the Stokes V component is equal to the total intensity (Stokes I)
        print(
            f"Stokes I: {output_wavefront.stokes[0]}, \n Stokes V: {output_wavefront.stokes[3]}"
        )
        assert np.allclose(
            output_wavefront.stokes[3], output_wavefront.stokes[0], atol=1e-6
        )  # Circular polarisation should have Stokes V equal to Stokes I

    # Now tests that have spatially varying polarisation
    def test_spatially_varying_optic(
        self, spatially_varying_liquid_crystal_optic, unpol_wavefront
    ):
        optic = spatially_varying_liquid_crystal_optic
        output_wavefront = optic(unpol_wavefront)
        # The output wavefront should have a radial polarisation pattern
        # Check that the Stokes Q and U components vary with position in a way consistent with a radial pattern
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

        print(f"Expected Q: {expected_Q}, \n Normalized Q: {normalized_Q}")
        print(f"Expected U: {expected_U}, \n Normalized U: {normalized_U}")

        assert np.allclose(normalized_Q, expected_Q, atol=1e-6)
        assert np.allclose(normalized_U, expected_U, atol=1e-6)
