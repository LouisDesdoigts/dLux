import jax.numpy as np
import pytest

from dLux.coord_specs import BaseSpec, CoordSpec, PadSpec


def test_base_spec():
    assert isinstance(BaseSpec(), BaseSpec)


def test_pad_spec():
    spec = PadSpec(pad=2, crop=3, c=1.5)
    assert (spec.pad, spec.crop) == (2, 3)
    assert spec.c.shape == ()
    assert spec.c == 1.5


def test_coord_spec_properties():
    spec = CoordSpec(n=4, d=0.5, c=1.0)
    assert spec.d.shape == spec.c.shape == ()
    assert np.allclose(spec.xs, np.array([0.25, 0.75, 1.25, 1.75]))
    assert spec.fov == 2
    assert spec.extent == (0, 2)


def test_coord_spec_none_values():
    spec = CoordSpec(n=4, d=None, c=None)
    assert spec.d is None
    assert spec.c is None
    for name, message in [
        ("xs", "coordinates"),
        ("fov", "FOV"),
        ("extent", "extent"),
    ]:
        with pytest.raises(ValueError, match=message):
            getattr(spec, name)
