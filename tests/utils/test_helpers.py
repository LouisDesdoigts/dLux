import pytest
from collections import OrderedDict
from dLux.utils.helpers import (
    map2array,
    list2dictionary,
    insert_layer,
    remove_layer,
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
