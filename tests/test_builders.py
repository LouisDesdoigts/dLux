"""Tests for grid-aware optical-component builders."""

import jax.numpy as np
import jax.random as jr
import pytest

import dLux as dl

from tests.helpers import assert_jittable


@pytest.mark.parametrize("mode", ["l1", "l2", "max", "rms", "p2v"])
def test_norm_contract(mode):
    basis = np.asarray(
        [
            [[0.0, 1.0], [2.0, 0.0]],
            [[0.0, -1.0], [1.0, 0.0]],
        ]
    )
    support = np.asarray([[False, True], [True, False]])

    output = dl.Norm(mode, scale=2.0)(basis, support)
    value = getattr(dl.utils, f"{mode}_norm")(
        output, mask=support, axis=(-2, -1)
    )

    assert np.allclose(value, 2.0)


def test_aperture_builder_contract():
    grid = dl.GridSpec(n=32, d=0.1, unit="m")
    builder = dl.SimpleCircular(
        diameter=2.0,
        secondary_diameter=0.4,
        opd=dl.ZernikeDef(orders=1),
        oversample=2,
    )

    transmission, basis, support = builder.build(grid, return_support=True)
    optic = builder(grid, key=jr.key(0))
    plain = dl.SimpleCircular(diameter=2.0, oversample=2)
    compiled = plain.build(grid, jit=True, return_support=True)
    assert_jittable(lambda value: value(grid), plain)

    assert transmission.shape == support.shape == (32, 32)
    assert compiled[0].shape == compiled[1].shape == (32, 32)
    assert basis.shape == (2, 32, 32)
    assert np.all(basis[:, ~support] == 0)
    assert optic.transmission.shape == (32, 32)
    assert optic.opd.coefficients.shape == (2,)
    assert optic.normalise
    assert not builder(grid, normalise=False).normalise


def test_aperture_builder_transform_and_validation():
    grid = dl.GridSpec(n=24, d=0.1, unit="m")
    builder = dl.ApertureBuilder(dl.Circle(1.0), opd=dl.ZernikeDef(nolls=1))

    centered = builder.build(grid)
    shifted = builder.build(grid, transform=dl.Affine(translation=[0.2, 0.0]))

    assert not np.allclose(centered[0], shifted[0])
    with pytest.raises(ValueError, match="exactly one"):
        dl.ZernikeDef()
    with pytest.raises(TypeError, match="primary"):
        dl.ApertureBuilder(np.ones((2, 2)))
    with pytest.raises(TypeError, match="obscurations"):
        dl.ApertureBuilder(dl.Circle(1.0), obscurations=[np.ones((2, 2))])
    with pytest.raises(ValueError, match="one of coefficients or key"):
        builder(grid, coefficients=[0.0], key=jr.key(0))


def test_sparse_builder_contract():
    grid = dl.GridSpec(n=24, d=0.1, unit="m")
    builder = dl.NRMLike(
        centers=[[-0.5, 0.0], [0.5, 0.0]],
        hole=dl.Circle(0.4),
        opd=dl.ZernikeDef(nolls=1),
        oversample=2,
    )

    global_optic = builder(grid)
    sparse_optic = builder(grid, sparse=True, coefficients=[[0.1], [-0.1]])

    assert isinstance(global_optic, dl.Optic)
    assert isinstance(sparse_optic, dl.SparseOptic)
    assert global_optic.normalise and sparse_optic.normalise
    assert sparse_optic.centers.shape == (2, 2)
    assert sparse_optic.opd.coefficients.shape == (2, 1)


def test_sparse_builder_materialisation_options():
    grid = dl.GridSpec(n=24, d=0.1, unit="m")
    builder = dl.SparseApertureBuilder(
        dl.Circle(0.4),
        centers=[[-0.5, 0.0], [0.5, 0.0]],
        opd=dl.ZernikeDef(nolls=[1, 2]),
        oversample=2,
    )

    shared = builder(grid, sparse=True, shared=True, key=jr.key(0))
    independent = builder(grid, sparse=True, key=jr.key(0))
    plain = dl.SparseApertureBuilder(
        dl.Circle(0.4), centers=builder.centers, oversample=2
    )(grid, sparse=True)

    assert shared.opd.coefficients.shape == (2,)
    assert independent.opd.coefficients.shape == (2, 2)
    assert plain.opd is None

    with pytest.raises(ValueError, match="jit is not supported"):
        builder(grid, sparse=True, jit=True)

    obscured = dl.SparseApertureBuilder(
        dl.Circle(0.4),
        centers=builder.centers,
        global_obscurations=[dl.Circle(0.1)],
    )
    with pytest.raises(ValueError, match="Global obscurations"):
        obscured(grid, sparse=True)


@pytest.mark.parametrize(
    "builder",
    [
        dl.HSTLike(oversample=1),
        dl.JWSTLike(oversample=1),
        dl.JWSTNRMLike(oversample=1),
        dl.EuclidLike(oversample=1),
        dl.SegmentedHex(nrings=2, segment_f2f=0.8, oversample=1),
    ],
)
def test_prebuilt_pupils(builder):
    grid = dl.GridSpec(n=32, d=0.3, unit="m")

    transmission = builder.build(grid)

    assert transmission.shape == (32, 32)
    assert np.all((transmission >= 0) & (transmission <= 1))
    assert 0 < transmission.sum() < transmission.size


def test_prebuilt_validation():
    with pytest.raises(ValueError, match="both be provided"):
        dl.SimpleCircular(1.0, spider_width=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        dl.SegmentedHex(nrings=2)
    with pytest.raises(TypeError, match="hole"):
        dl.NRMLike([[0.0, 0.0]], hole=np.ones((2, 2)))

    builder = dl.NRMLike(
        [[-0.2, 0.0], [0.2, 0.0]],
        hole=dl.Circle(0.2),
        opd=dl.ZernikeDef(nolls=1),
    )
    with pytest.raises(ValueError, match="leading axis"):
        builder(dl.GridSpec(16, 0.1, unit="m"), sparse=True, coefficients=[0.0])
