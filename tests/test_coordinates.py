from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest

from dLux.coordinates import Spec, PadSpec, CoordSpec


class TestSpec:
    def test_constructor(self):
        spec = Spec()
        assert isinstance(spec, Spec)


class TestPadSpec:
    def test_constructor(self):
        spec = PadSpec(pad=2, crop=3, c=1.5)

        assert spec.pad == 2
        assert spec.crop == 3
        assert spec.c == 1.5
        assert spec.c.shape == ()


class TestCoordSpec:
    def test_constructor(self):
        spec = CoordSpec(n=4, d=0.5, c=1.0)

        assert spec.n == 4
        assert spec.d == 0.5
        assert spec.c == 1.0
        assert spec.d.shape == ()
        assert spec.c.shape == ()

    def test_constructor_preserves_none(self):
        spec = CoordSpec(n=4, d=None, c=None)

        assert spec.d is None
        assert spec.c is None

    def test_xs(self):
        spec = CoordSpec(n=4, d=0.5, c=1.0)

        assert np.allclose(spec.xs, np.array([0.25, 0.75, 1.25, 1.75]))

    def test_fov(self):
        spec = CoordSpec(n=4, d=0.5)

        assert spec.fov == 2.0

    def test_extent(self):
        spec = CoordSpec(n=4, d=0.5, c=1.0)

        extent = spec.extent
        assert extent == (0.0, 2.0)

    @pytest.mark.parametrize(
        "prop_name, message",
        [
            ("xs", "d must be specified to calculate coordinates."),
            ("fov", "d must be specified to calculate FOV."),
            ("extent", "d must be specified to calculate extent."),
        ],
    )
    def test_properties_require_d(self, prop_name, message):
        spec = CoordSpec(n=4, d=None)

        with pytest.raises(ValueError, match=message):
            getattr(spec, prop_name)
