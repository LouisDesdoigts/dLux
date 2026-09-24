"""Compact stamping contracts and preparation diagnostics."""

import jax.numpy as np
import pytest
import dLux.utils as dlu


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda: dlu.paste(np.ones((2, 2)), [[0, 0]], (4, 4)), ValueError),
        (lambda: dlu.paste(np.ones((1, 2, 2)), [0, 0], (4, 4)), ValueError),
        (lambda: dlu.PasteSpec.from_grid(dlu.Axis(n=3), [[0, 0]], 1), TypeError),
        (lambda: dlu.PasteSpec.from_grid(dlu.Grid(n=5), [[0, 0]], 1), ValueError),
        (
            lambda: dlu.PasteSpec.from_grid(
                dlu.Grid(9, 1, c=[[0, 0], [1, 1]]), [[0, 0]], 1
            ),
            ValueError,
        ),
        (lambda: dlu.PasteSpec.from_grid(dlu.Grid(9, 1), [0, 0], 1), ValueError),
    ],
)
def test_paste_preparation_validation_tails(operation, error):
    """Coverage tails: stamp axes and complete, unbatched construction sampling."""
    with pytest.raises(error):
        operation()


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_paste_spec_extracts_batched_nonoverlapping_stamps(method):
    # A definition must be constructible before it can serve PastedBasis/builders.
    spec = dlu.PasteSpec(
        (6, 4), (2, 2), [[0, 0], [4, 2]], [[0.0, 0.0], [0.25, -0.25]], [1.0, 1.0]
    )
    assert isinstance(spec, dlu.Spec)
    assert spec.ndim == 2
    assert spec.shape == (4, 6)
    assert spec.stamp_shape == (2, 2)
    stamps = np.arange(24.0).reshape(2, 3, 2, 2)
    output = spec.paste(stamps, method)
    assert output.shape == (3, 4, 6)
    assert np.array_equal(spec.extract(output), stamps)
    local = dlu.Grid(2, 1).coordinates
    assert np.allclose(spec.coordinates[0], local)
    assert np.allclose(
        spec.coordinates[1], local + np.array([0.25, -0.25])[:, None, None]
    )


@pytest.mark.parametrize(
    "starts,offsets,message",
    [
        ([0, 0], [[0.0, 0.0]], "starts must have shape"),
        ([[0, 0]], [[0.0, 0.0], [1.0, 1.0]], "offsets must match"),
        ([[-1, 0]], [[0.0, 0.0]], "within the output"),
    ],
)
def test_paste_spec_constructor_validation_tails(starts, offsets, message):
    """Coverage tails: malformed placement metadata and out-of-bounds stamps."""
    with pytest.raises(ValueError, match=message):
        dlu.PasteSpec((4, 4), (2, 2), starts, offsets, [1.0, 1.0])


def test_paste_spec_array_shape_validation_tails():
    """Coverage tails: input stamps and extracted images must match the spec."""
    spec = dlu.PasteSpec((4, 4), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match="PasteSpec stamps"):
        spec.paste(np.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="PasteSpec output"):
        spec.extract(np.ones((3, 3)))
