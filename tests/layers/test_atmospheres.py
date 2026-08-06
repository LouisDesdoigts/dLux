import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.ndimage import affine_transform

from dLux.layers import InfiniteAtmosphericLayer


@pytest.fixture(scope="module")
def layer():
    return InfiniteAtmosphericLayer(
        npixels=4,
        pixel_scale=0.1,
        Cn_squared=1.0,
        L0=10.0,
        velocity=np.array([0.4, -0.2]),
        stencil_length=1,
        oversampling=1,
        seed=0,
    )


def test_initial_state(layer):
    assert layer.screen.shape == (4, 4)
    assert layer.base_opd.shape == (4, 4)
    assert layer.coords.shape == (2, 4)
    assert np.all(np.isfinite(layer.screen))
    assert np.allclose(layer.center, 0)
    assert np.allclose(layer.time, 0)


def test_step_advances_state_without_mutating_original(layer):
    screen, advanced = layer.step(0.5)

    assert screen.shape == layer.screen.shape
    assert np.allclose(screen, advanced.screen)
    assert np.allclose(advanced.center, np.array([0.2, -0.1]))
    assert np.allclose(advanced.time, 0.5)
    assert np.allclose(layer.center, 0)
    assert np.allclose(layer.time, 0)


def test_immutable_model_is_excluded_from_dynamic_state(layer):
    array_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(layer) if eqx.is_array(leaf)
    ]

    assert len(array_leaves) == 10

    screen, advanced = eqx.filter_jit(lambda state: state.step(0.5))(layer)

    assert np.allclose(screen, advanced.screen)
    assert advanced._model is layer._model


def test_sampling_matches_scipy_fifth_order_affine_transform(layer):
    base = np.arange(16, dtype=float).reshape(4, 4)
    residual = np.array([0.025, -0.025])
    expected = affine_transform(
        onp.asarray(base),
        onp.ones(2),
        onp.asarray(residual / layer.pixel_scale)[::-1],
        mode="nearest",
        order=5,
    )

    sampled = layer._sample(base, residual)

    assert np.allclose(sampled, expected, rtol=2e-5, atol=2e-5)


def test_evolve_matches_repeated_steps(layer):
    _, first = layer.step(0.25)
    expected_screen, expected_layer = first.step(0.25)

    screen, evolved = layer.evolve(0.25, 2)

    assert np.allclose(screen, expected_screen)
    assert np.allclose(evolved.screen, expected_layer.screen)
    assert np.allclose(evolved.center, expected_layer.center)
    assert np.allclose(evolved.time, expected_layer.time)


def test_reset_restores_initial_state(layer):
    _, advanced = layer.step(0.5)
    screen, reset = advanced.reset()

    assert np.allclose(screen, layer.screen)
    assert np.allclose(reset.screen, layer.screen)
    assert np.allclose(reset.center, 0)
    assert np.allclose(reset.time, 0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"npixels": 1}, "npixels"),
        ({"pixel_scale": 0}, "positive"),
        ({"Cn_squared": 0}, "positive"),
        ({"L0": 0}, "positive"),
        ({"velocity": [1, 2, 3]}, "velocity"),
        ({"stencil_length": 4}, "stencil_length"),
        ({"oversampling": 0}, "oversampling"),
    ],
)
def test_invalid_parameters(overrides, message):
    parameters = {
        "npixels": 4,
        "pixel_scale": 0.1,
        "Cn_squared": 1.0,
        "L0": 10.0,
        "stencil_length": 1,
        "oversampling": 1,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        InfiniteAtmosphericLayer(**parameters)
