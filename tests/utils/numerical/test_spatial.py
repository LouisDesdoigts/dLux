"""Grid sampling and local pixel geometry for spatial expressions."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    "profile",
    [
        dl.Circle(0.7, edge=1.0),
        dl.Gaussian([0.2, 0.3]),
        dl.CorrelatedGaussian([0.2, 0.3], 0.2),
        dl.EllipticalGaussian(0.3, 0.7, 0.2),
        dl.Basis(dl.utils.Zernike(nolls=[4, 5], diameter=1.0), coeffs=[1e-8, 2e-8]),
    ],
)
def test_sample_matches_explicit_si_coordinates(profile):
    grid = dl.Grid(n=(8, 6), diam=1000.0, unit="mm")
    result = assert_jittable(lambda model: model.evaluate(grid=grid), profile)
    expected = profile.resolve(coordinates=grid.coordinates, pixel_scale=grid.d_si)
    assert result.shape == (6, 8)
    assert np.allclose(result, expected)


def test_shape_call_forwards_state_and_defers_missing_inputs():
    shape = dl.Circle(zdx.StateRef("diameter"), edge=1.0)
    grid = dl.Grid(n=8, diam=1.0, unit="m")
    assert isinstance(shape.resolve(), dl.Circle)
    result = assert_jittable(
        lambda model: model(grid, state=zdx.State(diameter=0.7)), shape
    )
    assert np.allclose(result, dl.Circle(0.7, edge=1.0)(grid))


def test_soft_shape_defers_until_coordinate_sampling_is_supplied():
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    shape = dl.Circle(zdx.Map(0.35, s=2.0), edge=1.0)
    prepared = shape.resolve(coordinates=grid.coordinates)
    assert isinstance(prepared, dl.Circle)
    assert np.allclose(prepared.diameter, 0.7)
    with pytest.raises(zdx.Unresolved, match="pixel_scale"):
        shape.evaluate(coordinates=grid.coordinates)
    result = assert_jittable(
        lambda model: model.resolve(
            coordinates=grid.coordinates, pixel_scale=grid.d_si
        ),
        prepared,
    )
    assert np.allclose(result, dl.Circle(0.7, edge=1.0).evaluate(grid=grid))


@pytest.mark.parametrize(
    "transform",
    [
        dl.AffineMap(2 * np.eye(2)),
        dl.AffineMap([[1.0, 0.4], [0.0, 0.7]]),
        dl.Affine(rotation=0.3, translation=[0.05, 0.0]),
        dl.Distortion(powers=[[2], [0]], distortion=[[0.2], [0.0]]),
    ],
)
def test_transform_paths_preserve_source_softening_scale(transform):
    grid = dl.Grid(n=(16, 12), d=0.1, unit="m")
    shape = dl.Circle(0.8, edge=1.0)
    expected = shape.resolve(
        coordinates=grid.transformed(transform), pixel_scale=grid.d_si
    )
    direct = shape.evaluate(grid=grid, transform=transform)
    dynamic = shape.set(transform=transform).evaluate(grid=grid)
    assert np.allclose(direct, expected, atol=1e-6)
    assert np.allclose(dynamic, expected, atol=1e-6)
    wavefront = dl.Wavefront(wavelength=600e-9, grid=grid)
    optic = dl.Optic(transmission=shape, transform=transform)
    assert np.allclose(optic.phasor(wavefront).real, expected, atol=1e-6)


def test_chained_affine_maps_preserve_source_softening_scale():
    grid = dl.Grid(n=(13, 11), d=[0.06, 0.09], unit="m")
    first = dl.AffineMap([[1.0, 0.4], [0.0, 1.0]])
    second = dl.AffineMap([[0.0, -2.0], [0.5, 0.0]])
    shape = dl.Circle(0.7, edge=1.0)
    chained = shape.set(transform=dl.TransformChain([first, second]))
    matrix = second.matrix @ first.matrix
    combined = dl.AffineMap(matrix)
    result = assert_jittable(lambda model: model.evaluate(grid=grid), chained)
    expected = shape.evaluate(grid=grid, transform=combined)
    assert np.allclose(result, expected, atol=1e-6)

    reference = shape.evaluate(
        coordinates=combined(grid.coordinates), pixel_scale=grid.d_si
    )
    assert np.allclose(result, reference, atol=1e-6)


def test_distortion_coordinates_and_transform_gradients():
    grid = dl.Grid(n=(9, 7), d=0.1, unit="m")
    transform = dl.Distortion(powers=[[2], [0]], distortion=[[0.2], [0.0]])
    coordinates = transform(grid.coordinates)
    x, y = grid.coordinates
    assert np.allclose(coordinates[0], x + 0.2 * x**2)
    assert np.allclose(coordinates[1], y)

    profile = dl.Gaussian([0.15, 0.2], mean=[0.04, 0.02])

    def sampled_sum(scale):
        mapping = dl.Affine().set(scale=np.array([scale, 1.0]))
        return profile.evaluate(grid=grid, transform=mapping).sum()

    derivative = jax.jit(jax.grad(sampled_sum))(1.1)
    eps = 1e-3
    finite_difference = (sampled_sum(1.1 + eps) - sampled_sum(1.1 - eps)) / (2 * eps)
    assert np.abs(derivative) > 0
    assert np.allclose(derivative, finite_difference, rtol=2e-3)


def test_dynamic_basis_solves_physical_coefficients_on_a_grid():
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    basis = dl.Basis(dl.utils.Zernike(nolls=[4], diameter=1.0), coeffs=[1e-8])
    result = basis.evaluate(grid=grid)
    assert np.allclose(basis.solve(result, grid=grid), basis.coeffs, atol=1e-12)


def test_soft_edge_transform_gradient_with_fixed_sampling():
    grid = dl.Grid(n=(16, 12), d=0.07, unit="m")
    shape = dl.Circle(0.7, edge=1.5)

    def area(scale):
        transform = dl.AffineMap().set(matrix=scale * np.eye(2))
        return shape.evaluate(grid=grid, transform=transform).sum()

    derivative = jax.jit(jax.grad(area))(1.3)
    eps = 1e-3
    finite_difference = (area(1.3 + eps) - area(1.3 - eps)) / (2 * eps)
    assert np.abs(derivative) > 0
    assert np.allclose(derivative, finite_difference, rtol=2e-3)


def test_sample_resolves_transform_state():
    grid = dl.Grid(n=(8, 6), diam=1.0, unit="m")
    profile = dl.Gaussian(0.2)
    transform = dl.Affine().set(translation=zdx.StateRef("shift"))
    state = zdx.State(shift=[0.1, -0.05])
    output = profile.evaluate(grid=grid, transform=transform, state=state)
    expected = profile.evaluate(
        grid=grid, transform=dl.Affine(translation=[0.1, -0.05])
    )
    assert np.allclose(output, expected)


@pytest.mark.parametrize("profile", [dl.Circle(0.7), dl.Gaussian(0.2)])
def test_spatial_call_conventions(profile):
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    assert isinstance(profile(), type(profile))
    expected = profile.evaluate(grid=grid)
    assert np.allclose(profile(grid), expected)
    assert np.allclose(profile(grid=grid), expected)
    assert np.allclose(profile(coordinates=grid.coordinates), expected)


def test_direct_spatial_evaluation_defers_missing_inputs():
    class ChromaticProfile(dl.Sampled):
        amplitude: object

        def __init__(self, amplitude):
            super().__init__()
            self.amplitude = amplitude

        def evaluate(self, *, grid=None, coordinates=None, wavelengths, **context):
            context = self.sampling_context(grid, coordinates=coordinates, **context)
            coordinates = context.pop("coordinates")
            if wavelengths is None:
                raise ValueError("wavelengths must be supplied numerically")
            return zdx.resolve(self.amplitude, **context) * wavelengths * coordinates[0]

    profile = ChromaticProfile(zdx.Map(x=2.0, s=3.0))
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    prepared = profile.resolve(grid=grid)
    assert isinstance(prepared, ChromaticProfile)
    assert prepared.amplitude == 6.0
    with pytest.raises(TypeError, match="wavelengths"):
        profile.evaluate(grid=grid)
    with pytest.raises(ValueError, match="numerically"):
        profile.resolve(grid=grid, wavelengths=None)
    output = assert_jittable(lambda value: value(grid, wavelengths=0.5), profile)
    assert np.allclose(output, 3 * grid.coordinates[0])


@pytest.mark.parametrize(
    "profile",
    [
        dl.Gaussian(0.2),
        dl.CorrelatedGaussian([0.2, 0.3], 0.2),
        dl.EllipticalGaussian(0.3, 0.7),
        dl.Circle(0.7, edge=1.0),
        dl.Complement(dl.Circle(0.7)),
        dl.Circle(0.7, transform=dl.Affine(translation=[0.1, 0.0])),
        dl.Gaussian(0.2, transform=dl.Affine(translation=[0.1, 0.0])),
        dl.Basis(dl.utils.Zernike(nolls=[4]), coeffs=[1.0]),
        dl.Basis(dl.utils.Zernike(nolls=[4], diameter=1.0)),
    ],
)
def test_spatial_inputs_are_exclusive(profile):
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    with pytest.raises(zdx.Unresolved):
        profile.evaluate()
    assert isinstance(profile.resolve(), type(profile))
    for operation in (profile.evaluate, profile.resolve):
        with pytest.raises(ValueError, match="not both"):
            operation(grid=grid, coordinates=grid.coordinates)


def test_optic_supplies_grid_without_duplicate_spatial_inputs():
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    wavefront = dl.Wavefront(wavelength=600e-9, grid=grid)
    optic = dl.Optic(transmission=dl.Circle(0.7))
    expected = dl.Optic(transmission=dl.Circle(0.7).evaluate(grid=grid))(wavefront)
    result = assert_jittable(lambda model: model(wavefront), optic)
    assert np.allclose(result.phasor, expected.phasor)
    # Optic owns the evaluation frame and supplies its wavefront grid.
    assert np.allclose(
        optic(wavefront, coordinates=grid.coordinates).phasor, expected.phasor
    )


def test_spatial_basis_operations_reject_ambiguous_inputs():
    grid = dl.Grid(n=(8, 8), diam=1.0, unit="m")
    basis = dl.Basis(dl.utils.Zernike(nolls=[4], diameter=1.0))
    with pytest.raises(ValueError, match="not both"):
        basis.basis.evaluate(grid=grid, coordinates=grid.coordinates)
    with pytest.raises(ValueError, match="not both"):
        basis.solve(np.zeros((8, 8)), grid=grid, coordinates=grid.coordinates)


def test_soft_aperture_refinement_on_same_physical_pupil():
    aperture = dl.Circle(0.7, edge=1.0)
    errors = []
    for n in (32, 64, 128):
        coarse_grid = dl.Grid(n=(n, n), diam=1.0, unit="m")
        fine_grid = dl.Grid(n=(2 * n, 2 * n), diam=1.0, unit="m")
        coarse = aperture.evaluate(grid=coarse_grid)
        fine = aperture.evaluate(grid=fine_grid)
        downsampled = fine.reshape(n, 2, n, 2).mean(axis=(1, 3))

        # Each grid uses its own spacing; refinement preserves the physical pupil.
        assert np.allclose(fine_grid.d_si, coarse_grid.d_si / 2)
        errors.append(float(np.abs(coarse - downsampled).mean()))
        assert np.allclose(coarse.mean(), downsampled.mean(), rtol=3e-3)

    # Differences occupy the boundary and shrink in mean as sampling improves.
    assert errors[2] < errors[1] < errors[0] < 2e-3
