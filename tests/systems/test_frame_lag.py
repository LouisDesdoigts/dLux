"""Fixed frame delays keep commands in model state under JAX transforms."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl


@pytest.mark.parametrize(
    "frames, expected",
    [
        (0, [2, 2, -1, 4, 4]),
        (1, [0, 2, 2, -1, 4]),
        (3, [0, 0, 0, 2, 2]),
    ],
)
def test_frame_lag_eager_and_jit(frames, expected):
    lag = dl.FrameLag(frames)
    commands = np.array([2, 2, -1, 4, 4], dtype=float)[:, None, None]
    history = lag.initial_history(commands[0])
    assert history.shape == (frames, 1, 1)
    assert np.all(history == 0)

    eager = []
    for command in commands:
        applied, history = lag.step(history, command)
        eager.append(applied)
    eager = np.stack(eager)

    @eqx.filter_jit
    def run(model, values):
        def advance(history, command):
            applied, history = model.step(history, command)
            return history, applied

        return jax.lax.scan(advance, model.initial_history(values[0]), values)[1]

    wanted = np.array(expected, dtype=float)[:, None, None]
    assert np.array_equal(eager, wanted)
    assert np.array_equal(run(lag, commands), wanted)


@pytest.mark.parametrize("frames", [-1, 1.5, True, "1"])
def test_frame_lag_requires_non_negative_integer(frames):
    with pytest.raises(ValueError, match="non-negative integer"):
        dl.FrameLag(frames)


def test_frame_lag_rejects_incompatible_history():
    lag = dl.FrameLag(2)
    command = np.ones((2, 1))
    with pytest.raises(ValueError, match="shape"):
        lag.step(np.zeros((1, 2, 1)), command)
    with pytest.raises(ValueError, match="dtype"):
        lag.step(np.zeros((2, 2, 1), dtype=int), command)
