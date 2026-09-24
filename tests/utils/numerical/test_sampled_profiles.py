"""Stored sampling and transforms are shared by spatial definitions."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.fixture
def grid():
    return dlu.Grid(n=(9, 7), d=0.17, unit="m")


_FACTORIES = [
    lambda **kw: dlu.Gaussian([0.4, 0.7], **kw),
    lambda **kw: dlu.CorrelatedGaussian([0.4, 0.7], 0.2, **kw),
    lambda **kw: dlu.EllipticalGaussian(0.7, 0.5, 0.4, **kw),
    lambda **kw: dlu.Circle(0.8, edge=1.2, **kw),
    lambda **kw: dlu.Rectangle(0.6, 0.9, edge=1.1, **kw),
    lambda **kw: dlu.Zernike(nolls=[4], **kw),
    lambda **kw: dlu.Zernike(nolls=[1, 2, 4], diameter=1.2, **kw),
]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_stored_sampling_matches_direct_coordinate_evaluation(factory, grid):
    transform = dlu.Affine(translation=[0.08, -0.03], rotation=0.2)
    model = factory(grid=grid, transform=transform)
    coordinates = transform(grid.coordinates)
    expected = factory().evaluate(coordinates=coordinates, pixel_scale=grid.d_si)
    result = eqx.filter_jit(lambda value: value.resolve())(model)
    assert isinstance(model, dlu.Sampled)
    assert np.allclose(result, expected, rtol=2e-5, atol=2e-6)

    # Grid values remain ordinary differentiable/updateable model leaves.
    changed = model.set("grid.d", np.array([0.11, 0.14]))
    expected = factory().evaluate(
        coordinates=transform(changed.grid.coordinates), pixel_scale=changed.grid.d_si
    )
    assert np.allclose(changed.resolve(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_runtime_sampling_and_transform_replace_stored_choices(factory, grid):
    stored = dlu.Affine(translation=[0.2, 0.1])
    replacement = dlu.Affine(translation=[-0.05, 0.08], rotation=0.3)
    target = dlu.Grid(n=(7, 5), d=0.2, unit="m")
    model = factory(grid=grid, transform=stored)
    expected = factory().evaluate(
        coordinates=replacement(target.coordinates), pixel_scale=target.d_si
    )
    assert np.allclose(
        model.resolve(grid=target, transform=replacement), expected, atol=2e-6
    )
    assert model.grid.n == grid.n
    assert np.array_equal(model.transform.translation, stored.translation)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_strict_sampling_rejects_runtime_overrides_but_allows_set(factory, grid):
    transform = dlu.Affine(translation=[0.1, 0.0])
    model = factory(grid=grid, transform=transform, strict=True)
    with pytest.raises(ValueError, match="stored grid"):
        model.resolve(grid=grid)
    with pytest.raises(ValueError, match="stored transform"):
        model.resolve(transform=dlu.Affine())
    changed = model.set(grid=grid.set(d=np.array([0.12, 0.12])), transform=dlu.Affine())
    assert np.allclose(
        changed.resolve(),
        factory().evaluate(grid=changed.grid),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("factory", _FACTORIES[:3])
def test_gaussian_log_profile_and_gradient_use_the_stored_frame(factory, grid):
    model = factory(grid=grid, transform=dlu.Affine(translation=[0.06, -0.03]))
    assert np.allclose(np.exp(model.log_profile()), model.resolve(), rtol=2e-6)

    def pixel(shift):
        profile = model.set("transform.translation", np.array([shift, -0.03]))
        return profile.resolve()[2, 3]

    derivative = jax.jit(jax.grad(pixel))(np.asarray(0.06))
    step = 1e-3
    numerical = (pixel(0.06 + step) - pixel(0.06 - step)) / (2 * step)
    assert np.abs(derivative) > 1e-3
    assert np.allclose(derivative, numerical, rtol=2e-3, atol=2e-5)


def test_correlated_covariance_constructor_retains_sampling(grid):
    profile = dlu.CorrelatedGaussian.from_cov(
        [[0.3, 0.04], [0.04, 0.2]],
        grid=grid,
        transform=dlu.Affine(rotation=0.2),
        strict=True,
    )
    assert profile.strict
    assert profile.resolve().shape == grid.shape


def test_bound_shape_complement_needs_no_duplicate_grid(grid):
    shape = dlu.Circle(0.8, grid=grid, edge=1.1, strict=True)
    complement = dlu.Complement(shape)
    assert np.allclose(complement.resolve(), 1 - shape.resolve())
    unbound = dlu.Complement(dlu.Circle(0.8))
    assert isinstance(unbound.resolve(), dlu.Complement)


def test_dynamic_zernike_evaluation_basis_and_solve_map_coordinates_once(grid):
    coefficients = np.array([0.2, -0.03, 0.04, 0.07])
    transform = dlu.Affine(translation=[0.11, -0.08], rotation=0.25)
    model = dlu.Basis(
        dlu.Zernike(
            nolls=[1, 2, 3, 4],
            diameter=1.2,
            grid=grid,
            transform=transform,
            strict=True,
        ),
        coeffs=coefficients,
    )
    explicit = dlu.Basis(dlu.Zernike(nolls=[1, 2, 3, 4], diameter=1.2)).basis.evaluate(
        coordinates=transform(grid.coordinates)
    )
    assert np.allclose(model.basis.evaluate(), explicit, atol=2e-6)
    expected = np.einsum("k,kyx->yx", coefficients, explicit)
    assert np.allclose(
        eqx.filter_jit(lambda value: value.resolve())(model), expected, atol=2e-6
    )
    assert np.allclose(model.solve(expected), coefficients, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="stored grid"):
        model.basis.evaluate(grid=grid)
