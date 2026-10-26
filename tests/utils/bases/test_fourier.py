"""Fourier modes on cached grids and physical pupil coordinates."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_fourier_modes_have_known_phase_and_scaling():
    kernel = dlu.fourier_kernel_1d(3, 4, scale=2)
    assert np.allclose(kernel[:, 0], 2)
    assert np.allclose(kernel[:, 1], np.array([-2.0, 0.0, 2.0, 0.0]), atol=2e-7)
    assert np.allclose(kernel[:, 2], np.array([0.0, -2.0, 0.0, 2.0]), atol=2e-7)


def test_fourier_size_validation_tail():
    """Coverage tail: the separable Fourier expansion is two-dimensional."""
    with pytest.raises(ValueError, match="scalar or a"):
        dlu.fourier_kernels((2, 3, 4), 5)


def test_physical_fourier_period_uses_pupil_diameter():
    # The sampled grid spans two pupil diameters in each direction.
    coordinates = dl.Grid(n=16, diam=2.0, unit="m").coordinates
    fourier = dl.FourierBasis(16, (1, 3), pupil_diameter=1.0)
    modes = fourier.calculate_basis(coordinates=coordinates)

    assert modes.shape == (1, 3, 16, 16)
    cosine = modes[0, 1, 8]
    sine = modes[0, 2, 8]
    assert np.allclose(cosine[8], cosine[0], atol=1e-6)
    assert np.allclose(cosine[4], -cosine[0], atol=1e-6)
    assert np.allclose(sine[8], sine[0], atol=1e-6)
    assert np.allclose(sine[4], -sine[0], atol=1e-6)
    assert np.allclose(modes[0, 1], np.cos(2 * np.pi * coordinates[0]), atol=1e-6)


def test_physical_fourier_uses_transformed_coordinates_for_all_operations():
    grid = dl.Grid(n=(11, 13), diam=(1.8, 2.2), unit="m")
    coordinates = grid.transformed(
        dl.Affine(rotation=0.31, translation=np.array([0.12, -0.08]))
    )
    coeffs = np.array(
        [[0.1, 0.2, -0.3], [0.4, -0.2, 0.15], [-0.06, 0.12, 0.21]]
    )
    fourier = dl.FourierBasis((13, 11), (3, 3), coeffs, pupil_diameter=1.7)

    modes = assert_jittable(
        lambda model, coords: model.calculate_basis(coordinates=coords),
        fourier,
        coordinates,
    )
    assert modes.shape == (3, 3, 13, 11)
    assert np.allclose(
        modes[0, 1], np.cos(2 * np.pi * coordinates[0] / 1.7), atol=1e-6
    )
    assert np.allclose(
        modes[1, 0], np.cos(2 * np.pi * coordinates[1] / 1.7), atol=1e-6
    )

    expected = dl.Basis(modes, coeffs, axes=2).evaluate()
    output = assert_jittable(
        lambda model, coords: model.evaluate(coordinates=coords),
        fourier,
        coordinates,
    )
    assert np.allclose(output, expected, atol=1e-6)
    recovered = assert_jittable(
        lambda model, values, coords: model.solve(values, coordinates=coords),
        fourier,
        output,
        coordinates,
        atol=1e-5,
    )
    assert np.allclose(recovered, coeffs, atol=1e-5)

    batched = fourier.set(coeffs=np.stack([coeffs, 2 * coeffs]))
    batched_values = batched.evaluate(coordinates=coordinates)
    assert batched_values.shape == (2, 13, 11)
    assert np.allclose(batched_values[1], 2 * output, atol=1e-6)
    assert np.allclose(
        batched.solve(batched_values, coordinates=coordinates),
        batched.coeffs,
        atol=1e-5,
    )

    unitful = fourier.set(unit="nm")
    unitful_values = unitful.evaluate(coordinates=coordinates)
    assert np.allclose(unitful_values, output * 1e-9, rtol=1e-5, atol=1e-15)
    assert np.allclose(
        unitful.solve(unitful_values, coordinates=coordinates), coeffs, atol=1e-5
    )

    batched_unitful = batched.set(unit="nm")
    factor = dlu.unit_factor("nm")
    expected = dlu.eval_basis(modes, batched.coeffs, axes=2) * factor
    actual = assert_jittable(
        lambda model, coords: model.evaluate(coordinates=coords),
        batched_unitful,
        coordinates,
        atol=1e-14,
    )
    assert np.allclose(actual / factor, expected / factor, atol=1e-5)

    gradient = jax.jit(
        jax.grad(
            lambda values: batched_unitful.set(coeffs=values).evaluate(
                coordinates=coordinates
            )[1, 4, 5]
            / factor
        )
    )(batched_unitful.coeffs)
    expected_gradient = np.zeros_like(batched.coeffs).at[1].set(modes[:, :, 4, 5])
    assert np.allclose(gradient, expected_gradient, atol=1e-6)


def test_physical_fourier_requires_matching_coordinates():
    fourier = dl.FourierBasis((7, 9), (1, 2), pupil_diameter=1.0)
    with pytest.raises(zdx.Unresolved, match="coordinates"):
        fourier.evaluate()
    with pytest.raises(ValueError, match="coordinates"):
        fourier.calculate_basis(coordinates=np.zeros((2, 7, 8)))


def test_physical_fourier_resize_preserves_diameter_and_kernel_scales():
    coeffs = np.arange(6.0).reshape(2, 3) / 10
    fourier = dl.FourierBasis((5, 7), (2, 3), coeffs, pupil_diameter=0.8)
    scales = np.array([2.0, 3.0])
    kernels = tuple(scale * kernel for scale, kernel in zip(scales, fourier.kernels))
    scaled = fourier.set(kernels=kernels)
    resized = assert_jittable(lambda model: model.resize((9, 11)), scaled)

    assert np.allclose(resized.pupil_diameter, 0.8)
    assert np.allclose(
        np.array([kernel[0, 0] for kernel in resized.kernels]), scales
    )
    coordinates = dl.Grid(n=(11, 9), diam=1.6, unit="m").coordinates
    unscaled = dl.FourierBasis((9, 11), (2, 3), coeffs, pupil_diameter=0.8)
    assert np.allclose(
        resized.evaluate(coordinates=coordinates),
        6 * unscaled.evaluate(coordinates=coordinates),
        atol=1e-6,
    )


def test_physical_fourier_has_meaningful_coefficient_and_diameter_gradients():
    coordinates = dl.Grid(n=(9, 7), diam=1.5, unit="m").coordinates
    fourier = dl.FourierBasis(
        (7, 9), (1, 2), np.array([[0.2, 0.8]]), pupil_diameter=1.0
    )
    x = coordinates[0, 3, 5]

    def sample(model):
        return model.evaluate(coordinates=coordinates)[3, 5]

    coefficient_gradient = jax.jit(
        jax.grad(lambda coeffs: sample(fourier.set(coeffs=coeffs)))
    )(fourier.coeffs)
    assert np.allclose(
        coefficient_gradient, np.array([[1.0, np.cos(2 * np.pi * x)]]), atol=1e-6
    )

    diameter_gradient = jax.jit(
        jax.grad(lambda diameter: sample(fourier.set(pupil_diameter=diameter)))
    )(fourier.pupil_diameter)
    expected = fourier.coeffs[0, 1] * 2 * np.pi * x * np.sin(2 * np.pi * x)
    assert np.isfinite(diameter_gradient)
    assert np.abs(diameter_gradient) > 0.1
    assert np.allclose(diameter_gradient, expected, atol=1e-6)


def test_prepared_physical_fourier_is_a_fixed_geometry_snapshot():
    grid = dl.Grid(n=(11, 9), diam=(1.5, 1.3), unit="m")
    coordinates = grid.transformed(
        dl.Affine(translation=np.array([0.08, -0.04]))
    )
    coeffs = np.array(
        [[0.1, 0.2, 0.3], [-0.4, 0.2, 0.1], [0.5, -0.1, 0.3]]
    )
    fourier = dl.FourierBasis(
        (9, 11), (3, 3), coeffs, pupil_diameter=1.1, unit="nm"
    )
    factor = dlu.unit_factor("nm")

    prepared = fourier.prepare(coordinates=coordinates)
    assert isinstance(prepared, dl.FourierBasis)
    assert prepared.prepared
    assert prepared.unit == "nm"
    assert np.allclose(prepared.pupil_diameter, fourier.pupil_diameter)
    values = assert_jittable(lambda model: model.evaluate(), prepared, atol=1e-14)
    direct = fourier.evaluate(coordinates=coordinates)
    assert np.allclose(values / factor, direct / factor, atol=1e-5)
    assert np.allclose(
        prepared.calculate_basis(),
        fourier.calculate_basis(coordinates=coordinates),
        atol=1e-6,
    )
    contextual_values = assert_jittable(
        lambda model, coords: model.evaluate(coordinates=coords),
        prepared,
        coordinates,
        atol=1e-14,
    )
    assert np.allclose(contextual_values / factor, values / factor, atol=1e-6)

    batched = prepared.set(coeffs=np.stack([coeffs, 2 * coeffs]))
    batched_values = assert_jittable(
        lambda model: model.evaluate(), batched, atol=1e-14
    )
    assert batched_values.shape == (2, 9, 11)
    assert np.allclose(batched_values / factor, np.stack([direct, 2 * direct]) / factor)

    recovered = assert_jittable(lambda model, y: model.solve(y), prepared, values)
    assert np.allclose(recovered, coeffs, atol=1e-5)
    assert np.allclose(
        recovered, fourier.solve(direct, coordinates=coordinates), atol=1e-5
    )

    moved = dl.Affine(translation=np.array([0.12, -0.08]))(coordinates)
    changed = fourier.set(pupil_diameter=np.array(1.3))
    fresh = prepared.set(pupil_diameter=np.array(1.3)).prepare(coordinates=moved)
    assert np.allclose(prepared.evaluate() / factor, values / factor, atol=1e-6)
    assert np.allclose(
        prepared.set(pupil_diameter=np.array(1.3)).evaluate(coordinates=moved)
        / factor,
        values / factor,
        atol=1e-6,
    )
    assert np.allclose(
        prepared.calculate_basis(coordinates=moved),
        prepared.calculate_basis(),
        atol=1e-6,
    )
    assert np.allclose(
        prepared.solve(values, coordinates=moved), recovered, atol=1e-5
    )
    assert not np.allclose(
        prepared.evaluate() / factor,
        changed.evaluate(coordinates=moved) / factor,
        atol=1e-3,
    )
    assert np.allclose(
        fresh.evaluate() / factor,
        changed.evaluate(coordinates=moved) / factor,
        atol=1e-5,
    )

    resized = prepared.resize((13, 15))
    assert not resized.prepared
    assert np.allclose(resized.pupil_diameter, fourier.pupil_diameter)
    resized_coordinates = dl.Grid(n=(15, 13), diam=(1.5, 1.3), unit="m").coordinates
    assert np.allclose(
        resized.evaluate(coordinates=resized_coordinates) / factor,
        fourier.resize((13, 15)).evaluate(coordinates=resized_coordinates) / factor,
        atol=1e-5,
    )

    scales = (2.0, 3.0)
    scaled = fourier.set(
        kernels=tuple(scale * kernel for scale, kernel in zip(scales, fourier.kernels))
    )
    scaled_resized = scaled.prepare(coordinates=coordinates).resize((13, 15))
    assert np.allclose(
        np.array([kernel[0, 0] for kernel in scaled_resized.kernels]), np.array(scales)
    )

    def sample(model):
        return model.evaluate()[4, 6] / factor

    gradient = jax.jit(jax.grad(lambda c: sample(prepared.set(coeffs=c))))(coeffs)
    direct_gradient = jax.jit(
        jax.grad(
            lambda c: fourier.set(coeffs=c).evaluate(coordinates=coordinates)[4, 6]
            / factor
        )
    )(coeffs)
    assert np.max(np.abs(gradient)) > 0.1
    assert np.allclose(gradient, prepared.calculate_basis()[:, :, 4, 6], atol=1e-6)
    assert np.allclose(gradient, direct_gradient, atol=1e-6)


def test_prepared_physical_fourier_reuses_kernels_in_optic(monkeypatch):
    grid = dl.Grid(n=(11, 9), diam=(1.5, 1.3), unit="m")
    coeffs = np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.15], [-0.06, 0.12, 0.21]])
    fourier = dl.FourierBasis(
        grid.shape, (3, 3), coeffs, pupil_diameter=1.1, unit="nm"
    )
    prepared = fourier.prepare(coordinates=grid.coordinates)
    transmission = dl.Circle(1.1)
    wavefront = dl.Wavefront(900e-9, grid)
    opd = fourier.evaluate(coordinates=grid.coordinates)
    incident = wavefront.phasor * transmission(grid)

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("Prepared Fourier evaluation rebuilt physical modes.")

    monkeypatch.setattr(dl.FourierBasis, "_physical_modes", unexpected_rebuild)
    optic = dl.Optic(transmission=transmission, opd=prepared)
    for scale in (1.0, 2.0):
        changed = optic.set("opd.coeffs", scale * coeffs)
        output = assert_jittable(lambda model: model(wavefront).phasor, changed)
        expected = incident * np.exp(2j * np.pi * scale * opd / wavefront.wavelength)
        assert np.allclose(output, expected, rtol=1e-6, atol=1e-7)


def test_prepared_physical_fourier_requires_separable_coordinates():
    grid = dl.Grid(n=9, diam=1.5, unit="m")
    coordinates = grid.transformed(dl.Affine(rotation=0.23))
    fourier = dl.FourierBasis(9, (3, 3), pupil_diameter=1.1)

    with pytest.raises(ValueError, match="separable coordinates"):
        fourier.prepare(coordinates=coordinates)
    assert np.isfinite(fourier.evaluate(coordinates=coordinates)).all()
