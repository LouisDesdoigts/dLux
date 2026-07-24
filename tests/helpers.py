"""Shared assertions for dLux interface contracts."""

import equinox as eqx
import jax.numpy as np
import jax.tree_util as jtu


def array_leaves(tree):
    """Return paths and array leaves from a PyTree."""
    return [
        (jtu.keystr(path), leaf)
        for path, leaf in jtu.tree_leaves_with_path(tree)
        if eqx.is_array(leaf)
    ]


def assert_finite_tree(tree):
    """Assert that every array leaf contains neither NaNs nor infinities."""
    for path, leaf in array_leaves(tree):
        assert not np.any(np.isnan(leaf)), f"NaN values at {path}"
        assert np.all(np.isfinite(leaf)), f"Non-finite values at {path}"


def assert_tree_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    """Assert matching PyTree structure, metadata, and array values."""
    actual_leaves, actual_structure = jtu.tree_flatten_with_path(actual)
    expected_leaves, expected_structure = jtu.tree_flatten_with_path(expected)
    assert actual_structure == expected_structure, "PyTree structures do not match."

    for (path, actual_leaf), (_, expected_leaf) in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        path = jtu.keystr(path)
        if eqx.is_array(actual_leaf):
            assert np.allclose(
                actual_leaf,
                expected_leaf,
                rtol=rtol,
                atol=atol,
            ), f"Array values differ at {path}"
        else:
            assert actual_leaf == expected_leaf, f"Metadata differs at {path}"


def assert_jittable(function, *args, rtol=1e-6, atol=1e-6, **kwargs):
    """Assert eager and filtered-JIT execution agree and remain finite."""
    eager = function(*args, **kwargs)
    compiled = eqx.filter_jit(function)(*args, **kwargs)
    assert_finite_tree(eager)
    assert_finite_tree(compiled)
    assert_tree_allclose(eager, compiled, rtol=rtol, atol=atol)
    return eager


def assert_differentiable(
    function,
    target,
    *args,
    rtol=1e-6,
    atol=1e-6,
    **kwargs,
):
    """Assert filtered Grad and JIT-Grad agree and remain finite."""

    def loss(differentiable):
        output = function(differentiable, *args, **kwargs)
        leaves = [
            leaf
            for _, leaf in array_leaves(output)
            if np.issubdtype(leaf.dtype, np.inexact)
        ]
        if not leaves:
            raise ValueError("Differentiation requires an inexact array output.")
        return sum(np.real(np.vdot(leaf, leaf)) for leaf in leaves)

    gradient = eqx.filter_grad(loss)
    eager = gradient(target)
    compiled = eqx.filter_jit(gradient)(target)
    assert_finite_tree(eager)
    assert_finite_tree(compiled)
    assert_tree_allclose(eager, compiled, rtol=rtol, atol=atol)
    return eager
