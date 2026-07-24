"""Tests for dLux.states."""

import jax.numpy as np
import pytest

import dLux as dl

from .helpers import assert_differentiable, assert_jittable


class TestPSF:
    def test_construction(self, make_psf, make_wavefront):
        psf = make_psf()
        converted = dl.PSF.from_wavefront(make_wavefront())

        assert psf.data.shape == psf.spec.shape
        assert psf.batch_ndim == 0
        assert converted.data.shape == converted.spec.shape

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf.normalise(),
            lambda psf: psf.normalise("peak", 2.0),
            lambda psf: psf.convolve(np.ones((3, 3)), method="direct"),
            lambda psf: psf.convolve(np.ones((3, 3)), method="fft"),
            lambda psf: psf.interpolate(dl.Affine(rotation=0.1)),
            lambda psf: psf.rotate(0.1),
            lambda psf: psf.scale_to(6, 0.08),
            lambda psf: psf.resize(6),
            lambda psf: psf.downsample(2),
            lambda psf: psf.flip((0, 1)),
        ],
    )
    def test_spatial_operation_contract(self, operation, make_psf):
        data = np.arange(64.0).reshape(8, 8) + 1
        assert_jittable(operation, make_psf(data=data), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf + 2.0,
            lambda psf: psf - 2.0,
            lambda psf: psf * 2.0,
            lambda psf: psf / 2.0,
        ],
    )
    def test_arithmetic_contract(self, operation, make_psf):
        assert_jittable(operation, make_psf())

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf.normalise(),
            lambda psf: psf.convolve(np.ones((3, 3))),
            lambda psf: psf.interpolate(dl.Affine(rotation=0.1)),
        ],
    )
    def test_gradients(self, operation, make_psf):
        psf = make_psf(data=np.arange(64.0).reshape(8, 8) + 1)

        def apply(data):
            return operation(psf.set(data=data)).data

        assert_differentiable(apply, psf.data)

    def test_sampling_updates(self, make_psf):
        psf = make_psf().downsample(2)

        assert psf.spec.n == (4, 4)
        assert np.allclose(psf.spec.d, 0.2)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda spec: dl.PSF(np.ones(8), spec),
            lambda spec: dl.PSF(np.ones((4, 4)), spec),
            lambda spec: dl.PSF(np.ones((8, 8)), "invalid"),
        ],
    )
    def test_construction_validation(self, constructor, make_spec):
        with pytest.raises((TypeError, ValueError)):
            constructor(make_spec())

    def test_operation_validation(self, make_psf):
        psf = make_psf()

        with pytest.raises(ValueError, match="mode"):
            psf.normalise("invalid")
        with pytest.raises(TypeError, match="transformation"):
            psf.interpolate("invalid")
        with pytest.raises(TypeError, match="Unsupported type"):
            psf + "invalid"
