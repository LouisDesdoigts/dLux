"""Parametric paraxial matrices, composition and compiled numerical contracts."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu


def test_matrix_validation_distinguishes_arrays_from_physical_lct_inputs():
    matrix = np.array([[2, 0], [0, 1]])
    assert np.array_equal(dlu.ABCD.validate(matrix), matrix)
    with pytest.raises(ValueError, match="unimodular"):
        dlu.ABCD.validate(matrix, unimodular=True)

    # The same public check keeps parameter batches and derivatives under JIT.
    def validate_distances(distance):
        matrix = dlu.ABCD(free_space=distance).resolve()
        return dlu.ABCD.validate(matrix, unimodular=True)

    distance = np.array([0.2, -0.3])
    actual = jax.jit(validate_distances)(distance)
    assert actual.shape == (2, 2, 2)
    assert np.array_equal(actual[..., 0, 1], distance)
    gradient = jax.grad(lambda d: validate_distances(d)[..., 0, 1].sum())(distance)
    assert np.array_equal(gradient, np.ones_like(distance))
    assert not jax.make_jaxpr(validate_distances)(distance).effects


@pytest.mark.parametrize(
    "definition,expected",
    [
        ({"lens": 2.0}, [[1.0, 0.0], [-0.5, 1.0]]),
        ({"radius": 2.0}, [[1.0, 0.0], [-1.0, 1.0]]),
        ({"power": 2.0}, [[1.0, 0.0], [-2.0, 1.0]]),
        ({"free_space": 2.0}, [[1.0, 2.0], [0.0, 1.0]]),
        ({"fraunhofer": 2.0}, [[0.0, 2.0], [-0.5, 0.0]]),
    ],
)
def test_physical_definitions_resolve_to_their_paraxial_matrices(definition, expected):
    expression = dlu.ABCD(**definition)
    matrix = expression.resolve()
    assert expression.matrix is None
    assert np.allclose(matrix, np.array(expected))
    assert np.allclose(expression.determinant, 1.0)
    assert np.allclose(
        eqx.filter_jit(lambda value: value.resolve())(expression), matrix
    )


def test_direct_matrix_and_mixed_chain_use_physical_application_order():
    space = dlu.ABCD(free_space=2.0)
    lens = np.array([[1.0, 0.0], [-1.0 / 3.0, 1.0]])
    chain = dlu.ABCDChain({"space": space, "lens": lens})
    expected = lens @ np.array([[1.0, 2.0], [0.0, 1.0]])
    assert np.array_equal(dlu.ABCD(lens).resolve(), lens)
    assert np.allclose(chain.resolve(), expected)
    compiled = eqx.filter_jit(lambda value: value.resolve())(chain)
    assert np.allclose(compiled, expected)
    assert np.allclose(space.then(lens).resolve(), expected)
    assert np.allclose(chain.then(dlu.ABCD.identity()).resolve(), expected)
    assert space.is_free_space()
    assert dlu.ABCD(lens).is_surface()
    assert dlu.ABCD(expected).is_unimodular()


def test_chain_then_preserves_named_parameters_and_application_order():
    chain = dlu.ABCDChain(
        {"space": dlu.ABCD(free_space=2.0), "lens": dlu.ABCD(lens=3.0)},
        alias=(("distance", "elements.space.free_space"),),
    )
    extended = chain.then(("exit", dlu.ABCD(free_space=0.5)))

    assert tuple(chain.elements) == ("space", "lens")
    assert tuple(extended.elements) == ("space", "lens", "exit")
    assert np.allclose(extended.get("distance"), 2.0)
    expected = np.array([[1.0, 0.5], [0.0, 1.0]]) @ chain.resolve()
    assert np.allclose(extended.resolve(), expected)


def test_repeated_then_builds_a_flat_chain_that_resolves_under_jit():
    chain = dlu.ABCDChain({"entrance": dlu.ABCD(free_space=0.001)})
    for _ in range(105):
        chain = chain.then(dlu.ABCD(free_space=0.001))

    assert len(chain.elements) == 106
    assert all(isinstance(element, dlu.ABCD) for element in chain.elements.values())
    expected = np.array([[1.0, 0.106], [0.0, 1.0]])
    assert np.allclose(chain.resolve(), expected)
    assert np.allclose(eqx.filter_jit(lambda value: value.resolve())(chain), expected)


@pytest.mark.parametrize("definition", [{}, {"lens": 2.0, "free_space": 1.0}])
def test_constructor_requires_one_definition(definition):
    with pytest.raises(ValueError, match="exactly one"):
        dlu.ABCD(**definition)


@pytest.mark.parametrize("parameter", ["lens", "radius", "fraunhofer"])
@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf")])
def test_physical_constructors_reject_invalid_concrete_scales(parameter, invalid):
    with pytest.raises(ValueError, match="finite|non-zero"):
        dlu.ABCD(**{parameter: invalid})


def test_abcd_optical_reversal_restores_rays_in_reversed_longitudinal_frame():
    distance = np.array([0.1, -0.2])
    focal_length = np.array([0.7, 0.9])
    translation = dlu.ABCD(free_space=distance)
    lens = dlu.ABCD(lens=focal_length)
    system = translation.then(lens)
    ray = np.array([0.003, 0.02])
    propagated = np.einsum("...ij,j->...i", system.resolve(), ray)
    expected = np.stack(
        [
            ray[0] + distance * ray[1],
            ray[1] - (ray[0] + distance * ray[1]) / focal_length,
        ],
        axis=-1,
    )
    recovered = np.einsum(
        "...ij,...j->...i", system.inverse().resolve(), propagated * np.array([1, -1])
    )
    assert np.allclose(propagated, expected)
    assert np.allclose(
        recovered, np.broadcast_to(ray * np.array([1, -1]), recovered.shape)
    )
    assert np.allclose(dlu.ABCD(system.resolve()).determinant, 1.0)


def test_runtime_context_resolves_physical_parameters_without_mutating_definitions():
    class Distance(zdx.Expression):
        def evaluate(self, distance, **context):
            return distance

    class OpticalModel(zdx.Module):
        matrix: jax.Array | zdx.Expression

    system = dlu.ABCDChain(
        {
            "space": dlu.ABCD(free_space=Distance()),
            "lens": dlu.ABCD(lens=zdx.Map(x=2.0, s=0.5)),
        }
    )
    realised = OpticalModel(system).resolve(distance=0.2)
    assert isinstance(realised, OpticalModel)
    assert np.allclose(realised.matrix, np.array([[1.0, 0.2], [-1.0, 0.8]]))

    assert isinstance(system.space.free_space, Distance)


def test_abcd_jit_has_no_runtime_checks_and_keeps_physical_derivatives():
    def traced_system(focal_length):
        matrix = dlu.ABCD(free_space=0.1).then(dlu.ABCD(lens=focal_length))
        return matrix.inverse().resolve()

    focal_length = np.array(0.7)
    jaxpr = jax.make_jaxpr(traced_system)(focal_length)
    assert not jaxpr.effects
    assert "callback" not in str(jaxpr).lower()
    actual = jax.jit(traced_system)(focal_length)
    assert np.allclose(actual, traced_system(focal_length))
    derivative = jax.jacfwd(traced_system)(focal_length)
    expected = np.array([[0.1 / focal_length**2, 0.0], [1 / focal_length**2, 0.0]])
    assert np.allclose(derivative, expected, rtol=2e-6, atol=2e-6)


def test_abcd_rejects_bad_shapes_and_nonfinite_concrete_matrices():
    for value in (np.zeros((2, 3)), np.array([[1, np.inf], [0, 1]])):
        with pytest.raises(ValueError):
            dlu.ABCD(value)
    with pytest.raises(ValueError, match="non-zero"):
        dlu.ABCD(np.zeros((2, 2))).inverse()
