"""Direct coordinate queries retain wavelength, population and Jones axes."""

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl


class CoordinateRamp(zdx.Expression):
    def evaluate(self, *, coordinates, **context):
        return 1 + coordinates[..., 0, :, :]


@pytest.mark.parametrize("jones_kind", ["scalar", "uniform", "chromatic"])
def test_wavefront_sample_matches_linear_complex_field(jones_kind):
    grid = dl.Grid(n=(6, 8), d=0.2)
    x, y = grid.coordinates
    # Equal wavelength/population sizes make accidentally paired axes observable.
    amplitudes = np.array([[1.0, 3.0], [2.0, 5.0]])
    phasor = amplitudes[..., None, None] * (1 + x + 1j * y)
    matrix = np.array([[1.0, 0.2j], [0.3, 0.8]])
    if jones_kind == "scalar":
        jones = None
    elif jones_kind == "uniform":
        jones = matrix
    else:
        jones = np.stack((matrix, 0.7 * matrix))
    wavefront = dl.Wavefront(np.array([5e-7, 6e-7]), grid, phasor, jones=jones)

    coordinates = dl.Grid(n=(3, 4), d=0.07, c=(0.02, -0.03)).coordinates
    qx, qy = coordinates
    expected = amplitudes[..., None, None] * (1 + qx + 1j * qy)
    if jones_kind != "scalar":
        matrices = np.broadcast_to(matrix, (2, 2, 2))
        if jones_kind == "chromatic":
            matrices = matrices * np.array([1.0, 0.7])[:, None, None]
        expected = (
            expected[:, :, None, None, :, :] * matrices[None, :, :, :, None, None]
        )

    actual = eqx.filter_jit(lambda field: field.sample(coordinates))(wavefront)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    if jones_kind != "scalar":
        assert np.allclose(actual, wavefront.materialise().sample(coordinates))


def test_population_opd_skips_jones_axes_once():
    grid = dl.Grid(n=4, d=0.2, unit="m")
    scalar = np.ones((2, 3, 4, 4, 4), dtype=complex)
    wavefront = dl.Wavefront(5e-7, grid, scalar).materialise()
    opd = np.arange(4)[:, None, None] * np.ones((4, 4)) * 1e-8
    actual = eqx.filter_jit(lambda field: field.add_opd(opd))(wavefront)
    expected = wavefront.phasor * np.exp(
        2j * np.pi / wavefront.wavelength * opd[:, None, None]
    )
    assert actual.phasor.shape == wavefront.phasor.shape
    assert np.allclose(actual.phasor, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("operation", ["optic", "sampling"])
def test_aperture_sampling_never_pairs_with_equal_wavelength_count(operation):
    centres = np.array([[-0.15, 0.0], [0.12, 0.05]])[:, None, :]
    grid = dl.Grid(n=6, d=0.2, c=centres, unit="m")
    x, y = grid.coordinates[..., 0, :, :], grid.coordinates[..., 1, :, :]
    scales = np.array([[1.0, 2.0], [3.0, 5.0]])
    phasor = scales[..., None, None] * (1 + x + 1j * y)
    wavefront = dl.Wavefront(np.array([5e-7, 6e-7]), grid, phasor)
    assert wavefront.phasor.shape[:2] == (2, 2)

    if operation == "optic":
        actual = eqx.filter_jit(dl.Optic(transmission=CoordinateRamp()))(wavefront)
        expected = phasor * (1 + x)
        actual = actual.phasor
    else:
        coordinates = dl.Grid(n=3, d=0.1, unit="m").coordinates
        actual = eqx.filter_jit(
            lambda field: field.sample(coordinates, method="linear")
        )(wavefront)
        x, y = coordinates
        expected = scales[..., None, None] * (1 + x + 1j * y)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("polarised", [False, True])
def test_point_tuples_keep_population_and_jones_axes(polarised):
    grid = dl.Grid((5, 7), d=0.2)
    x, y = grid.coordinates
    samples = np.asarray([1.0, 2.0])[:, None, None] * (1 + x + 2j * y)
    field = dl.Wavefront(6e-7, grid, samples)
    if polarised:
        field = field.materialise()
    points = (np.asarray([0.1, -0.2, 0.3]), np.asarray([-0.1, 0.2, 0.0]))
    result = eqx.filter_jit(lambda value: value.sample(points))(field)
    expected = np.asarray([1.0, 2.0])[:, None] * (1 + points[0] + 2j * points[1])
    if polarised:
        expected = expected[:, None, None, :] * np.eye(2)[None, :, :, None]
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=2e-6)
