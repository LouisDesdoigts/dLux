"""Regular sampling shared by one-dimensional axes and Cartesian grids."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_axis_values_follow_editable_spacing_and_centre():
    axis = dlu.Axis(n=4, d=2, c=10, unit="nm")
    assert isinstance(axis, dlu.Spec)
    assert axis.n == (4,)
    assert axis.shape == (4,)
    assert np.allclose(axis.values, np.asarray([7, 9, 11, 13]))
    updated = axis.set(d=np.asarray([4.0]), c=np.asarray([20.0]))
    values = assert_jittable(lambda sampling: sampling.values, updated)
    assert np.allclose(values, np.asarray([14, 18, 22, 26]))
    assert np.allclose(axis.coordinates / 1e-9, np.asarray([[7, 9, 11, 13]]))
    assert np.allclose(axis.d_si / 1e-9, np.asarray([2]))
    assert np.allclose(axis.c_si / 1e-9, np.asarray([10]))
    assert np.allclose(axis.fov, np.asarray([8]))
    assert np.allclose(axis.cell_volume / 1e-9, 2)


def test_axis_and_grid_share_coordinate_and_gradient_contracts():
    def sample(centre, spacing):
        return dlu.Axis(n=5, d=spacing, c=centre, unit=None).values

    centre = np.asarray(2.0)
    spacing = np.asarray(0.3)
    values = jax.jit(sample)(centre, spacing)
    assert np.allclose(values, dlu.Grid(5, spacing, centre).axes[0])
    dcentre, dspacing = jax.jit(jax.jacrev(sample, argnums=(0, 1)))(centre, spacing)
    assert np.allclose(dcentre, np.ones(5))
    assert np.allclose(dspacing, np.arange(5) - 2)

    # Concrete validation does not become a callback when construction is traced.
    reconstructed = jax.jit(lambda x: dlu.Axis(x, unit=None).values)(values)
    assert np.allclose(reconstructed, values)
    jaxpr = str(jax.make_jaxpr(sample)(centre, spacing))
    assert "callback" not in jaxpr


def test_explicit_regular_values_have_one_authoritative_representation():
    axis = dlu.Axis([500.0, 550.0, 600.0], unit="nm")
    assert np.allclose(
        axis.set(c=np.asarray([700.0])).values, np.asarray([650, 700, 750])
    )
    assert "values" not in axis.__dict__
    copied = jax.tree_util.tree_map(lambda value: value, axis)
    assert eqx.tree_equal(copied, axis)
    assert np.allclose(axis.integration_weights / 1e-9, np.asarray([25, 50, 25]))
    with pytest.raises(ValueError, match="regularly spaced"):
        dlu.Axis([500.0, 550.0, 700.0], unit="nm")
    with pytest.raises(ValueError, match="regularly spaced"):
        dlu.Grid.from_axes((np.asarray([0.0, 1.0, 3.0]), np.asarray([0.0, 1.0])))
    with pytest.raises(ValueError, match="positive"):
        dlu.Axis([3.0, 2.0, 1.0])


def test_partial_axis_matches_samples_and_singletons_have_no_integration_interval():
    axis = dlu.Axis(d=0.5, c=1.0, unit=None).match_shape((3,))
    assert np.allclose(axis.values, np.asarray([0.5, 1.0, 1.5]))
    single = dlu.Axis([3.0], unit=None)
    assert np.array_equal(single.values, np.asarray([3.0]))
    assert single.integration_weights is None
    with pytest.raises(ValueError, match="one coordinate"):
        dlu.Axis(n=(2, 3), d=1)
    with pytest.raises(ValueError, match="one axis"):
        dlu.Axis(n=3, d=(1, 2))


def test_field_uses_shared_axis_sampling_properties():
    axis = dlu.Axis(n=3, d=2, c=10, unit="nm")
    field = dlu.Field(np.asarray([1.0, 2.0, 3.0]), axis)
    assert np.allclose(field.pixel_scale / 1e-9, np.asarray([2]))
    assert np.allclose(field.center / 1e-9, np.asarray([10]))
    assert np.allclose(field.diameter, np.asarray([6]))


class GeneralSampling(dlu.UnitSpec):
    """Minimal extension using UnitSpec's dimension inference."""

    n: tuple | None = None
    d: jax.Array | None = None
    c: jax.Array | None = None


@pytest.mark.parametrize(
    "kwargs,ndim",
    [
        ({"n": (2, 3, 4), "d": 0.5}, 3),
        ({"d": [1.0, 2.0, 3.0]}, 3),
        ({"c": 2.0}, 1),
        ({}, 0),
    ],
)
def test_unitspec_extension_infers_dimensions(kwargs, ndim):
    sampling = GeneralSampling(**kwargs)
    assert sampling.ndim == ndim
    if sampling.n is not None:
        assert sampling.shape == (4, 3, 2)
        assert sampling.coordinates.shape == (3, 4, 3, 2)


def test_sampling_shape_dimension_validation_tail():
    """Coverage tail: shape must describe the specification's physical dimension."""
    with pytest.raises(ValueError, match="dimension"):
        dlu.Grid().match_shape((3,))


def test_abstract_spec_attributes_allow_fields_or_properties():
    class PropertySpec(dlu.Spec):
        @property
        def n(self):
            return (3, 5)

        @property
        def ndim(self):
            return 2

    assert PropertySpec().shape == (5, 3)
    assert "n" not in PropertySpec.__dataclass_fields__
    assert "ndim" not in dlu.Grid.__dataclass_fields__
    with pytest.raises(TypeError):
        dlu.Spec()

    spec = dlu.PasteSpec((4, 4), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
    updated = assert_jittable(
        lambda value: value.set(offsets=value.offsets + 0.25), spec
    )
    assert np.allclose(updated.coordinates, spec.coordinates + 0.25)
