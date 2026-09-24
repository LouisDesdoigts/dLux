"""One named sequence contract for operators and bound numerical expressions."""

import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_sequential_retains_owner_and_series_resolves_runtime_input():
    layers = {"scale": zdx.Map(s=2), "offset": zdx.Map(b=1)}
    sequential = dlu.Sequential(layers)
    series = dlu.Series(layers)
    target = np.asarray([1.0, 3.0])
    assert not hasattr(series, "x")
    assert isinstance(sequential.resolve(), dlu.Sequential)
    assert isinstance(dlu.Series(layers).resolve(), dlu.Series)
    assert np.allclose(
        assert_jittable(lambda model: model.resolve(target=target), series),
        np.asarray([3, 7]),
    )
    assert np.allclose(
        sequential(np.asarray([1.0, 3.0])), series.resolve(target=target)
    )
    assert np.allclose(series(np.asarray([2.0, 4.0])), np.asarray([5, 9]))


def test_nested_sequences_debug_and_immutable_edits():
    inner = dlu.Sequential({"double": zdx.Map(s=2)})
    model = dlu.Sequential({"inner": inner, "offset": zdx.Map(b=3)})
    output, intermediate = model.debug(np.asarray(2.0))
    assert output == 7
    assert list(intermediate) == ["input", "inner", "offset"]
    changed = model.insert(("extra", zdx.Map(s=5)), 1).remove("offset")
    assert changed(np.asarray(2.0)) == 20
    assert model(np.asarray(2.0)) == 7


def test_shared_links_are_populated_at_sequence_owner():
    shared = zdx.Linked(2.0, key="scale")
    sequence = dlu.Sequential(
        {"first": zdx.Map(s=shared), "second": zdx.Map(s=shared.defer())}
    )
    assert np.allclose(
        assert_jittable(lambda model: model(np.asarray(3.0)), sequence), 12
    )


def test_empty_series_and_expression_target():
    series = dlu.Series()
    assert isinstance(series(), dlu.Series)
    assert series.resolve(target=zdx.Map(x=3.0, s=2)) == 6
    assert dlu.Sequential()(np.asarray(4.0)) == 4
    assert series.steps is series.layers


def test_generated_names_respect_explicit_suffixes():
    steps = [("gain", zdx.Map(s=2)), ("gain", zdx.Map(s=3)), ("gain_0", zdx.Map(s=4))]
    model = dlu.Sequential(steps)
    assert list(model.steps) == ["gain_1", "gain_2", "gain_0"]
    assert model(np.asarray(1.0)) == 24
    changed = model.insert(("fifth", zdx.Map(s=5)), 0, zdx.Map)
    assert changed(np.asarray(1.0)) == 120


def test_sequence_validation_tails():
    """Coverage tails: diagnostic paths for names, types and empty reductions."""
    import pytest

    cases = [
        (lambda: dlu.Sequential({"bad name": zdx.Map(s=1)}), ValueError),
        (lambda: dlu.Sequential([zdx.Module()]), TypeError),
        (lambda: dlu.Sequential([zdx.Map(s=1)], layer_type=dlu.Sequential), TypeError),
        (lambda: dlu.Combine([]), ValueError),
    ]
    for operation, error in cases:
        with pytest.raises(error):
            operation()
