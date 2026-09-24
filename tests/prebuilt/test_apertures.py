"""Tests for grid-aware optical-component builders."""

import equinox as eqx
import jax.numpy as np
import jax.random as jr
import pytest
from jax import grad

import dLux as dl
import dLux.utils as dlu

from dLux.prebuilt.apertures import SparseApertureBuilder
from dLux.utils.compat.apertures import euclid_like
from tests.helpers import assert_jittable


@pytest.mark.parametrize("angles", [(0, 120, 240), (15, 80, 210)])
def test_euclid_spiders_retain_released_orientation(angles):
    grid = dl.Grid(64, diam=1.21)
    optic = dl.EuclidAper(
        grid, spider_angles=angles, pupil_oversample=3, normalise=False
    )
    reference = euclid_like(64, spider_angles=angles, oversample=3)

    # The old helper further oversamples circular edges. Compare inside the clear
    # pupil, where both implementations sample exactly the same spider geometry.
    radius = np.sqrt((grid.coordinates**2).sum(0))
    interior = (radius > 0.395 / 2 + 2 * grid.d[0]) & (
        radius < 1.21 / 2 - 2 * grid.d[0]
    )
    assert np.array_equal(optic.transmission[interior], reference[interior])


def test_aperture_builder_contract():
    grid = dl.Grid(n=32, d=0.1, unit="m")
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
    assert isinstance(basis, dlu.Basis)
    assert basis.basis.shape == (2, 32, 32)
    assert np.all(basis.basis[:, ~support] == 0)
    assert optic.transmission.shape == (32, 32)
    assert optic.opd.coeffs.shape == (2,)
    assert optic.normalise
    assert not builder(grid, normalise=False).normalise


def test_builder_keeps_complete_basis_and_uses_stored_grid():
    grid = dlu.Grid(n=8, d=0.1, unit="m")
    basis = dlu.Basis(np.ones((1, 8, 8)), coeffs=[7.0], unit="nm")
    builder = dl.ApertureBuilder(dlu.Circle(0.6), opd=basis, grid=grid, oversample=1)

    optic = builder(jit=False)
    _, built_basis = builder.build(jit=False)
    updated = builder(coeffs=[3.0], jit=False)
    compiled = assert_jittable(lambda value: value(jit=False), builder)

    assert isinstance(builder, dl.Def)
    assert built_basis is basis
    assert optic.opd is basis
    assert np.allclose(optic.opd.resolve() / 1e-9, 7.0)
    assert np.allclose(updated.opd.resolve() / 1e-9, 3.0)
    assert np.allclose(compiled.opd.resolve() / 1e-9, 7.0)
    assert updated.opd.unit == basis.unit == "nm"


def test_builder_grid_override_and_definition_units():
    stored = dlu.Grid(n=8, d=0.1, unit="m")
    override = stored.resize(10)
    builder = dl.SimpleCircular(
        diameter=0.6,
        opd=dl.ZernikeDef(nolls=1, unit="nm"),
        grid=stored,
        oversample=1,
    )

    zero = builder(jit=False)
    explicit = builder(override, coeffs=[4.0], jit=False)
    random = builder(key=jr.key(0), jit=False)

    assert zero.transmission.shape == (8, 8)
    assert explicit.transmission.shape == (10, 10)
    assert zero.opd.unit == explicit.opd.unit == random.opd.unit == "nm"
    assert np.array_equal(zero.opd.coeffs, np.zeros(1))
    assert np.array_equal(random.opd.coeffs, jr.normal(jr.key(0), (1,)))
    assert np.allclose(explicit.opd.resolve() / 1e-9, explicit.opd.basis[0] * 4.0)


def test_builder_constructor_key_and_call_override():
    grid = dl.Grid(n=24, d=0.1, unit="m")
    stored_key = jr.key(0)
    call_key = jr.key(1)
    builder = dl.SimpleCircular(
        diameter=1.0,
        opd=dl.ZernikeDef(nolls=[1, 2]),
        oversample=2,
        key=stored_key,
    )

    stored = builder(grid, jit=False)
    overridden = builder(grid, key=call_key, jit=False)
    explicit = builder(grid, coeffs=[0.1, -0.1], jit=False)
    compiled = assert_jittable(lambda value: value(grid, jit=False), builder)

    assert np.array_equal(builder.key, stored_key)
    assert np.allclose(stored.opd.coeffs, jr.normal(stored_key, (2,)))
    assert np.allclose(compiled.opd.coeffs, stored.opd.coeffs)
    assert np.allclose(overridden.opd.coeffs, jr.normal(call_key, (2,)))
    assert np.allclose(explicit.opd.coeffs, np.array([0.1, -0.1]))


def test_aperture_builder_transform_and_validation():
    grid = dl.Grid(n=24, d=0.1, unit="m")
    builder = dl.ApertureBuilder(dlu.Circle(1.0), opd=dl.ZernikeDef(nolls=1))

    centered = builder.build(grid)
    shifted = builder.build(grid, transform=dlu.Affine(translation=[0.2, 0.0]))

    assert not np.allclose(centered[0], shifted[0])
    with pytest.raises(ValueError, match="exactly one"):
        dl.ZernikeDef()
    with pytest.raises(TypeError, match="primary"):
        dl.ApertureBuilder(np.ones((2, 2)))
    with pytest.raises(TypeError, match="obscurations"):
        dl.ApertureBuilder(dlu.Circle(1.0), obscurations=[np.ones((2, 2))])
    with pytest.raises(ValueError, match="one of coeffs or key"):
        builder(grid, coeffs=[0.0], key=jr.key(0))


def test_zernike_definition_methods():
    grid = dl.Grid(n=24, d=0.1, unit="m")
    coordinates = grid.coordinates
    support = dlu.Circle(1.0).evaluate(coordinates=coordinates) > 0
    nolls = [11, 1, 4, 2, 7]
    padded = dl.ZernikeDef(nolls=nolls, method="padded")
    mapped = dl.ZernikeDef(nolls=nolls, method="mapped")

    padded_basis = padded(coordinates=coordinates, support=support, diameter=1.0).basis
    mapped_basis = mapped(coordinates=coordinates, support=support, diameter=1.0).basis
    expected = dlu.Zernike(nolls=nolls).evaluate(
        coordinates=coordinates, diameter=1.0, support=support
    )

    padded_grad = grad(
        lambda d: padded(
            coordinates=coordinates, support=support, diameter=d
        ).basis.sum()
    )(1.0)
    mapped_grad = grad(
        lambda d: mapped(
            coordinates=coordinates, support=support, diameter=d
        ).basis.sum()
    )(1.0)

    assert np.allclose(padded_basis, expected)
    assert np.allclose(mapped_basis, expected)
    assert np.allclose(padded_grad, mapped_grad)

    enlarged = padded.set(oversize=np.asarray(0.01))(
        coordinates=coordinates, support=support, diameter=1.0
    ).basis
    expected_enlarged = dlu.Zernike(nolls=nolls).evaluate(
        coordinates=coordinates, diameter=1.01, support=support
    )
    assert np.allclose(enlarged, expected_enlarged)
    assert not np.allclose(enlarged, expected)

    with pytest.raises(ValueError, match="padded.*mapped"):
        dl.ZernikeDef(nolls=1, method="invalid")(grid)

    assert len(dl.ZernikeDef(order=3).nolls) == 10
    with pytest.raises(ValueError, match="exactly one"):
        dl.ZernikeDef(order=3, orders=[2, 3])


def test_sparse_builder_contract():
    grid = dl.Grid(n=24, d=0.1, unit="m")
    builder = dl.NRMLike(
        centers=[[-0.5, 0.0], [0.5, 0.0]],
        hole=dlu.Circle(0.4),
        opd=dl.ZernikeDef(nolls=1),
        oversample=2,
    )

    global_optic = builder(grid)
    sparse_optic = builder(grid, sparse=True, coeffs=[[0.1], [-0.1]])

    assert isinstance(global_optic, dl.Optic)
    assert isinstance(sparse_optic, dl.SparseOptic)
    assert global_optic.normalise and sparse_optic.normalise
    assert sparse_optic.centers.shape == (2, 2)
    assert sparse_optic.opd.coeffs.shape == (2, 1)


def test_sparse_builder_materialisation_options():
    grid = dl.Grid(n=24, d=0.1, unit="m")
    builder = dl.SparseApertureBuilder(
        dlu.Circle(0.4),
        centers=[[-0.5, 0.0], [0.5, 0.0]],
        opd=dl.ZernikeDef(nolls=[1, 2]),
        oversample=2,
        key=jr.key(0),
    )

    shared = builder(grid, sparse=True, shared=True)
    independent = builder(grid, sparse=True)
    plain = dl.SparseApertureBuilder(
        dlu.Circle(0.4), centers=builder.centers, oversample=2
    )(grid, sparse=True)

    assert shared.opd.coeffs.shape == (2,)
    assert independent.opd.coeffs.shape == (2, 2)
    assert plain.opd is None

    eager = builder(grid, sparse=True, jit=False)
    compiled = builder(grid, sparse=True, jit=True)
    assert np.allclose(eager.transmission, compiled.transmission)
    assert np.allclose(eager.opd.basis, compiled.opd.basis)

    obscured = dl.SparseApertureBuilder(
        dlu.Circle(0.4),
        centers=builder.centers,
        global_obscurations=[dlu.Circle(0.1)],
    )
    sparse = obscured(grid, sparse=True)
    assert sparse.transmission.shape == (len(builder.centers),) + grid.shape
    with pytest.raises(ValueError, match="shared local SparseOptic"):
        builder(grid, transform=dlu.Affine(rotation=0.1), sparse=True)


def test_segmented_hex_pasted_construction():
    grid = dl.Grid(n=64, d=0.15, unit="m")
    builder = dl.SegmentedHex(
        nrings=3, segment_diameter=1.524, gap=0.007, remove_center=True, oversample=3
    )
    scatter = builder.set(paste_method="scatter")

    pasted = builder.build(grid)
    compiled = builder.build(grid, jit=True)
    parallel = scatter.build(grid, jit=True)
    global_ = SparseApertureBuilder._build(builder, grid, None)

    assert np.allclose(pasted, global_)
    assert np.allclose(compiled, global_)
    assert np.allclose(parallel, global_)

    rotated = builder.build(grid, transform=dlu.Affine(rotation=0.1))
    assert rotated.shape == pasted.shape
    assert not np.allclose(rotated, pasted)


def test_segmented_hex_rectangular_construction():
    grid = dl.Grid(n=(72, 64), d=(0.15, 0.15), unit="m")
    builder = dl.SegmentedHex(
        nrings=3,
        segment_diameter=1.524,
        gap=0.007,
        remove_center=True,
        oversample=(2, 3),
    )

    pasted = builder.build(grid, jit=True)
    dense = SparseApertureBuilder._build(builder, grid, None)

    assert pasted.shape == (64, 72)
    assert np.allclose(pasted, dense)


def test_segmented_hex_pasted_basis():
    grid = dl.Grid(n=64, d=0.15, unit="m")
    builder = dl.SegmentedHex(
        nrings=3,
        segment_diameter=1.524,
        gap=0.007,
        remove_center=True,
        opd=dl.ZernikeDef(orders=[1, 2], norm="rms", unit="nm"),
        oversample=2,
        key=jr.key(0),
    )

    optic = builder(grid, jit=True)
    rotated = builder(grid, transform=dlu.Affine(rotation=0.1))
    output = assert_jittable(lambda value: value.opd.evaluate(), optic)

    assert isinstance(optic.opd, dlu.PastedBasis)
    assert isinstance(rotated.opd, dlu.Basis)
    assert optic.opd.basis.shape[:2] == (18, 5)
    assert optic.opd.coeffs.shape == (18, 5)
    assert output.shape == (64, 64)
    assert optic.opd.unit == rotated.opd.unit == "nm"
    assert np.allclose(
        output / 1e-9,
        optic.opd.spec.paste(
            np.sum(optic.opd.basis * optic.opd.coeffs[..., None, None], axis=1)
        ),
        atol=1e-6,  # Dot versus elementwise accumulation differs by <1e-6 nm.
    )


def test_prebuilt_validation():
    with pytest.raises(ValueError, match="both be provided"):
        dl.SimpleCircular(1.0, spider_width=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        dl.SegmentedHex(nrings=2)
    with pytest.raises(ValueError, match="paste_method"):
        dl.SegmentedHex(nrings=2, segment_f2f=0.8, paste_method="invalid")
    with pytest.raises(ValueError, match="greater than zero"):
        dl.SegmentedHex(nrings=2, segment_f2f=0.0)
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        dl.SegmentedHex(nrings=2, segment_f2f=0.8, gap=-0.1)
    with pytest.raises(TypeError, match="hole"):
        dl.NRMLike([[0.0, 0.0]], hole=np.ones((2, 2)))

    builder = dl.NRMLike(
        [[-0.2, 0.0], [0.2, 0.0]],
        hole=dlu.Circle(0.2),
        opd=dl.ZernikeDef(nolls=1),
    )
    with pytest.raises(ValueError, match="leading axis"):
        builder(dl.Grid(16, 0.1, unit="m"), sparse=True, coeffs=[0.0])


@pytest.mark.parametrize(
    "constructor", [dl.HSTAper, dl.JWSTAper, dl.JWSTNRMAper, dl.EuclidAper]
)
def test_aperture_presets_construct_optics_for_imagers(constructor):
    pupil = dl.Grid(n=24, diam=9.0, unit="m")
    image = dl.Grid(n=8, d=1e-7, unit="rad")
    optic = constructor(
        pupil, pupil_oversample=1, alias=(("pupil_transmission", "transmission"),)
    )
    assert type(optic) is dl.Optic
    assert optic.grid is pupil
    assert optic.pupil_transmission is optic.transmission
    model = dl.Imager(
        pupil,
        image,
        pupil=optic,
        oversample=2,
        source=dl.Source(),
        bandpass=dlu.BandPass(6e-7, 1e-7, n=2),
    )
    assert isinstance(model, dl.Imager)
    transmission = model.pupil.transmission
    assert transmission.shape == pupil.shape
    assert np.all((transmission >= 0) & (transmission <= 1))
    assert 0 < transmission.sum() < transmission.size
    assert model.wavefront().spatial_shape == (16, 16)
    assert model.image().spatial_shape == (8, 8)


@pytest.mark.parametrize("mode", ["l1", "l2", "max", "rms", "p2v"])
def test_zernike_definition_normalisation(mode):
    grid = dl.Grid(n=24, d=0.1, unit="m")
    support = dlu.Circle(1.0).resolve(grid=grid) > 0
    basis = dl.ZernikeDef(nolls=[2, 3, 4], norm=mode, unit="nm")(
        coordinates=grid.coordinates, support=support, diameter=1.0
    ).basis
    measured = dlu.Norm(basis, mode=mode, weights=support, axis=(-2, -1)).resolve()
    assert np.allclose(measured, 1.0, rtol=2e-6)


def test_preset_focal_length_is_a_differentiable_array():
    pupil = dl.Grid(n=8, diam=3.0, unit="m")
    image = dl.Grid(n=8, d=1e-6, unit="m")
    optic = dl.HSTAper(pupil, pupil_oversample=1, focal_length=2.0)
    model = dl.Imager(
        pupil,
        image,
        pupil=optic,
        source=dl.Source(),
        bandpass=dlu.BandPass(6e-7, 1e-7, n=2),
    )

    derivative = eqx.filter_grad(lambda system: system.psf().data[..., 3, 3].sum())(
        model
    )
    assert derivative.pupil.focal_length is not None
    assert np.isfinite(derivative.pupil.focal_length)
    assert np.abs(derivative.pupil.focal_length) > 1e-6
