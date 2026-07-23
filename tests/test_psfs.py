from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import dLux.utils as dlu
from dLux import Affine, PSF


@pytest.fixture
def psf():
    return PSF(np.ones((16, 16)), 1 / 16)


class TestPSF:
    def test_constructor(self, psf):
        assert psf.npixels == 16
        assert psf.pixel_scale == 1 / 16

    def test_properties(self, psf):
        assert psf.ndim == 0

    def test_methods(self, psf):
        assert psf.downsample(2).npixels == 8
        assert psf.downsample(2).pixel_scale == 1 / 8
        assert np.allclose(psf.normalise().data.sum(), 1)
        assert np.allclose(psf.normalise("power", 2).data.sum(), 2)
        assert np.allclose(psf.normalise("peak", 2).data.max(), 2)
        assert isinstance(psf.convolve(np.ones((2, 2))), PSF)
        assert isinstance(psf.convolve(np.ones((2, 2)), method="fft"), PSF)
        assert isinstance(psf.rotate(np.pi), PSF)
        assert isinstance(psf.interpolate(Affine()), PSF)
        assert isinstance(psf.resize(8), PSF)
        assert isinstance(psf.flip(0), PSF)

    def test_interpolate_validation(self, psf):
        with pytest.raises(TypeError, match="transformation"):
            psf.interpolate(transformation="rotate")

    def test_normalise_validation(self, psf):
        with pytest.raises(ValueError, match="mode"):
            psf.normalise("invalid")

    def test_interpolate_matches_explicit_coordinate_mapping(self, psf):
        psf = psf.set(data=np.arange(16**2).reshape(16, 16))
        transformation = Affine(
            translation=[1 / 32, -1 / 32],
            scale=[0.9, 1.1],
        )
        coords = dlu.pixel_coords(psf.npixels, psf.npixels * psf.pixel_scale)
        expected = dlu.interp(psf.data, coords, transformation(coords))

        output = psf.interpolate(transformation)

        assert np.allclose(output.data, expected)

    def test_interpolate_fill(self, psf):
        output = psf.interpolate(Affine(translation=[10.0, 10.0]), fill=2.0)

        assert np.allclose(output.data, 2.0)

    def test_magic(self, psf):
        psf *= np.ones(1)
        assert isinstance(psf, PSF)

        psf += np.ones(1)
        assert isinstance(psf, PSF)

        psf -= np.ones(1)
        assert isinstance(psf, PSF)

        psf /= np.ones(1)
        assert isinstance(psf, PSF)

    def test_magic_with_psf_operand(self, psf):
        other = PSF(np.full((16, 16), 2.0), 1 / 16)

        added = psf + other
        assert isinstance(added, PSF)
        assert np.allclose(added.data, 3.0)

        subtracted = psf - other
        assert isinstance(subtracted, PSF)
        assert np.allclose(subtracted.data, -1.0)

    def test_magic_with_none(self, psf):
        unchanged = psf._magic_unified_op(None, "add")
        assert unchanged is psf

    def test_magic_invalid_type(self, psf):
        with pytest.raises(TypeError, match="Unsupported type"):
            psf + "invalid"

    def test_magic_invalid_operation(self, psf):
        with pytest.raises(ValueError, match="Unsupported operation"):
            psf._magic_unified_op(np.ones(1), "invalid")
