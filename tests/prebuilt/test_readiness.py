"""Builder assembly, explicit coefficients and constructor validation tails."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu


def test_global_obscurations_are_applied_after_assembling_apertures():
    grid = dlu.Grid(9, 0.2, unit="m")
    primary = dlu.Circle(0.7)
    centers = np.array([[-0.4, 0.0], [0.4, 0.0]])
    obscuration = dlu.Circle(0.5)
    builder = dl.SparseApertureBuilder(
        primary, centers, global_obscurations=[obscuration], oversample=1
    )
    actual = builder(grid, normalise=False, jit=False).transmission
    x = grid.coordinates
    components = [
        primary.resolve(coordinates=dlu.Affine(translation=c)(x)) for c in centers
    ]
    expected = np.clip(sum(components), 0, 1) * (1 - obscuration.resolve(grid=grid))
    assert np.array_equal(actual, expected)


def test_sparse_builder_allocates_inspectable_zero_coefficients():
    grid = dlu.Grid(5, 0.1, unit="m")
    vectors = np.ones((2,) + grid.shape)
    basis = dlu.Basis(vectors)
    builder = dl.SparseApertureBuilder(
        dlu.Circle(0.4), [[-0.3, 0.0], [0.3, 0.0]], opd=basis
    )
    result = builder(grid, sparse=True, shared=False, jit=False)
    assert result.opd.coeffs.shape == (2, 2)
    assert np.array_equal(result.opd.coeffs, np.zeros((2, 2)))


def test_builder_validation_tails():
    grid = dlu.Grid(5, 0.1, unit="m")
    circle = dlu.Circle(0.4)
    builder = dl.ApertureBuilder(circle)
    cases = [
        (ValueError, lambda: dl.prebuilt.apertures.segmented_hex_cens(0, 1.0)),
        (TypeError, lambda: dl.ApertureBuilder(circle, obscurations=1)),
        (TypeError, lambda: dl.ApertureBuilder(circle, opd=1)),
        (TypeError, lambda: builder(1, jit=False)),
        (ValueError, lambda: builder(dlu.Grid(n=5), jit=False)),
        (ValueError, lambda: builder(grid.set(c=np.zeros((2, 2))), jit=False)),
        (TypeError, lambda: builder(grid, transform=1, jit=False)),
        (ValueError, lambda: dl.SparseApertureBuilder(circle, [0.0, 0.0])),
        (
            TypeError,
            lambda: dl.SparseApertureBuilder(
                circle, [[0.0, 0.0]], global_obscurations=1
            ),
        ),
        (
            TypeError,
            lambda: dl.SparseApertureBuilder(
                circle, [[0.0, 0.0]], global_obscurations=[1]
            ),
        ),
        (ValueError, lambda: dl.SegmentedHex(2, segment_diameter=np.array([1.0, 2.0]))),
        (
            ValueError,
            lambda: dl.ZernikeDef(order=1)(grid, coordinates=grid.coordinates),
        ),
        (
            ValueError,
            lambda: dl.ApertureBuilder(
                dlu.Spider(0.1, [0.0]), opd=dl.ZernikeDef(order=1)
            )(grid, jit=False),
        ),
        (
            ValueError,
            lambda: dl.SparseApertureBuilder(
                dlu.Spider(0.1, [0.0]), [[0.0, 0.0]], opd=dl.ZernikeDef(order=1)
            )(grid, jit=False),
        ),
    ]
    for error, call in cases:
        with pytest.raises(error):
            call()
