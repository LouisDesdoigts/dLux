import pytest
from jax import numpy as np, config
from dLux.layers import Optic, TransmissiveLayer, AberratedLayer, BasisLayer

config.update("jax_debug_nans", True)
from collections import OrderedDict
from dLux.utils.helpers import (
    map2array,
    list2dictionary,
    insert_layer,
    remove_layer,
    scale_layer,
)


@pytest.fixture
def tree():
    return {"a": {"b": 1, "c": 2}, "d": {"e": 3, "f": 4}}


@pytest.fixture
def fn():
    return lambda x: x + 1


@pytest.fixture
def leaf_fn():
    return lambda x: isinstance(x, int)


@pytest.fixture
def list_in():
    return [("layer", 1), ("layer", 2), 3]


@pytest.fixture
def ordered():
    return True


@pytest.fixture
def allowed_types():
    return (int, str)


@pytest.fixture
def layers():
    return OrderedDict([("layer1", 1), ("layer2", 2), ("layer3", 3)])


@pytest.fixture
def layer():
    return ("layer4", 4)


@pytest.fixture
def index():
    return 1


@pytest.fixture
def key():
    return "layer2"


@pytest.fixture
def type():
    return int


@pytest.fixture
def grid8():
    # simple nontrivial 8x8 pattern
    x = np.linspace(-1, 1, 8)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.exp(-(X**2 + Y**2))


@pytest.fixture
def basis3(grid8):
    # 3 modes, each 8x8
    return np.stack([grid8, 0.5 * grid8, grid8 + 0.1], axis=0)


def test_map2array(tree, fn, leaf_fn):
    result = map2array(fn, tree, leaf_fn)
    assert result.shape == (4,)
    result = map2array(fn, tree)
    assert result.shape == (4,)


def test_list2dictionary(list_in, ordered, allowed_types):
    result = list2dictionary(list_in, ordered, allowed_types)
    assert isinstance(result, OrderedDict)
    assert len(result) == 3
    assert list(result.keys()) == ["layer_0", "layer_1", "int"]
    assert list(result.values()) == [1, 2, 3]
    with pytest.raises(TypeError):
        list2dictionary(list_in, ordered, (float,))
    with pytest.raises(ValueError):
        list2dictionary([("a space", 1)], ordered, (int,))


def test_insert_layer(layers, layer, index, type):
    result = insert_layer(layers, layer, index, type)
    assert isinstance(result, OrderedDict)
    assert len(result) == 4
    assert list(result.keys()) == ["layer1", "layer4", "layer2", "layer3"]
    assert list(result.values()) == [1, 4, 2, 3]


def test_remove_layer(layers, key):
    result = remove_layer(layers, key)
    assert isinstance(result, OrderedDict)
    assert len(result) == 2
    assert list(result.keys()) == ["layer1", "layer3"]
    assert list(result.values()) == [1, 3]


def test_scale_layer(grid8, basis3):
    """
    Comprehensive check for `scale_layer`:

    - Unsupported types raise TypeError.
    - Optic: scales opd/phase/transmission -> (npix_out, npix_out), no NaNs.
    - TransmissiveLayer: scales transmission -> (npix_out, npix_out), no NaNs.
    - AberratedLayer: scales opd & phase -> (npix_out, npix_out), no NaNs.
    - BasisLayer: scales each mode in [m, H, W] -> (m, npix_out, npix_out), no NaNs.
    """
    npix_in = 8
    npix_out = 16
    ps_in = 1.0 / npix_in
    ps_out = 1.0 / npix_out

    # Unsupported type
    class NotALayer:
        pass

    with pytest.raises(TypeError):
        scale_layer(NotALayer(), ps_in, ps_out, npix_out)

    # Optic with all fields present
    optic = Optic(opd=grid8, phase=2 * grid8, transmission=0.7 * grid8)
    optic_s = scale_layer(optic, ps_in, ps_out, npix_out)
    assert isinstance(optic_s, Optic)
    for arr in (optic_s.opd, optic_s.phase, optic_s.transmission):
        assert arr.shape == (npix_out, npix_out)
        assert not np.isnan(arr).any()

    # TransmissiveLayer
    t = TransmissiveLayer().set("transmission", grid8)
    t_s = scale_layer(t, ps_in, ps_out, npix_out)
    assert isinstance(t_s, TransmissiveLayer)
    assert t_s.transmission.shape == (npix_out, npix_out)
    assert not np.isnan(t_s.transmission).any()

    # AberratedLayer
    a = AberratedLayer().set(["opd", "phase"], [grid8, 0.5 * grid8])
    a_s = scale_layer(a, ps_in, ps_out, npix_out)
    assert isinstance(a_s, AberratedLayer)
    assert a_s.opd.shape == (npix_out, npix_out)
    assert a_s.phase.shape == (npix_out, npix_out)
    assert not np.isnan(a_s.opd).any()
    assert not np.isnan(a_s.phase).any()

    # BasisLayer
    b = BasisLayer().set("basis", basis3)  # shape (m=3, 8, 8)
    b_s = scale_layer(b, ps_in, ps_out, npix_out)
    assert isinstance(b_s, BasisLayer)
    m, H, W = b_s.basis.shape
    assert (m, H, W) == (basis3.shape[0], npix_out, npix_out)
    assert not np.isnan(b_s.basis).any()
