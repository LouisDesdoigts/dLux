from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest

from dLux import Wavefront
from dLux.coordinates import CoordSpec, PadSpec
import dLux.layers.abcd_propagators as abcd_props


@pytest.fixture
def wavefront():
    return Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)


def test_abcd_elements():
    elements = [
        abcd_props.ABCDFreeSpace(1.0),
        abcd_props.ABCDLens(2.0),
        abcd_props.ABCDMirror(3.0),
        abcd_props.ABCDConjugatePlane(4.0),
    ]

    for element in elements:
        matrix = element.abcd
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)


def test_abcd_propagator_base_and_getattr():
    spec = CoordSpec(n=8, d=1 / 32, c=0.0)
    prop = abcd_props.MFTPropagator(
        [abcd_props.ABCDFreeSpace(1.0), abcd_props.ABCDLens(2.0)], spec
    )

    assert prop.n == 8
    assert isinstance(prop.ABCDFreeSpace, abcd_props.ABCDFreeSpace)
    assert np.allclose(prop.focal_length, np.array(2.0))
    assert prop.abcd.shape == (2, 2)

    with pytest.raises(AttributeError):
        prop.not_an_attr


def test_mft_propagator_call(wavefront):
    spec = CoordSpec(n=8, d=1 / 32, c=0.0)
    prop = abcd_props.MFTPropagator([abcd_props.ABCDFreeSpace(1.0)], spec)

    out = prop(wavefront)
    assert isinstance(out, Wavefront)
    assert out.npixels == 8
    assert np.allclose(out.pixel_scale, 1 / 32)


def test_fft_propagator_constructor_error():
    with pytest.raises(ValueError, match="can not specify d"):
        abcd_props.FFTPropagator(
            [abcd_props.ABCDFreeSpace(1.0)], CoordSpec(n=8, d=1.0, c=0.0)
        )


def test_fft_propagator_call_coordspec(wavefront):
    spec = CoordSpec(n=20, c=0.0)
    prop = abcd_props.FFTPropagator([abcd_props.ABCDConjugatePlane(1.0)], spec)

    out = prop(wavefront)
    assert isinstance(out, Wavefront)
    assert out.npixels == 20


def test_fft_propagator_call_padspec(wavefront):
    spec = PadSpec(pad=2, crop=2, c=0.0)
    prop = abcd_props.FFTPropagator([abcd_props.ABCDConjugatePlane(1.0)], spec)

    out = prop(wavefront)
    assert isinstance(out, Wavefront)
    assert out.npixels == 16


def test_asm_propagator_constructor_error():
    with pytest.raises(ValueError, match="can not specify d or c"):
        abcd_props.ASMPropagator(1.0, CoordSpec(n=8, d=1.0, c=0.0))


def test_asm_propagator_call_coordspec(wavefront):
    prop = abcd_props.ASMPropagator(1.0, CoordSpec(n=20, c=None))

    out = prop(wavefront)
    assert isinstance(out, Wavefront)
    assert out.npixels == 20


def test_asm_propagator_call_padspec(wavefront):
    prop = abcd_props.ASMPropagator(1.0, PadSpec(pad=2, crop=2, c=0.0))

    out = prop(wavefront)
    assert isinstance(out, Wavefront)
    assert out.npixels == 16


def test_asm_propagator_getattr():
    prop = abcd_props.ASMPropagator(1.0, CoordSpec(n=8, c=None))

    assert prop.n == 8
    with pytest.raises(AttributeError):
        prop.not_an_attr


def test_not_implemented_constructors():
    spec = CoordSpec(n=8, c=None)
    prop = abcd_props.ASMPropagator(1.0, spec)
    assert isinstance(prop, abcd_props.ASMPropagator)

    with pytest.raises(TypeError):
        abcd_props.Fraunhofer()

    with pytest.raises(TypeError):
        abcd_props.Fresnel()


def test_fraunhofer_init_not_implemented():
    concrete_fraunhofer = type(
        "ConcreteFraunhofer", (abcd_props.Fraunhofer,), {"__call__": lambda self, w: w}
    )

    with pytest.raises(NotImplementedError):
        concrete_fraunhofer()


def test_fresnel_init_not_implemented():
    concrete_fresnel = type(
        "ConcreteFresnel", (abcd_props.Fresnel,), {"__call__": lambda self, w: w}
    )

    with pytest.raises(NotImplementedError):
        concrete_fresnel()
