"""Numerical Jones expressions, composition, rotation and Stokes conversion."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx
import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_factories_return_definition_matrix_or_array():
    definition = dlu.Jones.HalfWavePlate(angle=0.2)
    matrix = dlu.Jones.HalfWavePlate(angle=0.2, matrix=True)
    array = dlu.Jones.half_wave_plate(angle=0.2)
    assert isinstance(definition, dlu.Jones)
    assert isinstance(matrix, dlu.JonesMatrix)
    assert np.allclose(definition.resolve(), array)
    assert np.allclose(matrix.resolve(), array)
    assert np.allclose(array.conj().T @ array, np.eye(2), atol=1e-6)


def test_linear_polariser_and_uniform_rotation():
    horizontal = dlu.Jones.HorizontalPolariser()
    vertical = horizontal.rotate(np.pi / 2)
    assert np.allclose(horizontal.resolve(), np.asarray([[1, 0], [0, 0]]), atol=1e-6)
    assert np.allclose(vertical.resolve(), np.asarray([[0, 0], [0, 1]]), atol=1e-6)
    matrix = dlu.JonesMatrix(horizontal.resolve()).rotate(np.pi / 2)
    assert np.allclose(matrix.resolve(), vertical.resolve(), atol=1e-6)
    with pytest.raises(ValueError, match="None"):
        horizontal.rotate(None)
    with pytest.raises(ValueError, match="uniform"):
        dlu.Jones.HalfWavePlate(angle=np.zeros((3, 3))).rotate(0.2)


def test_composition_order_chain_and_polarisation_power():
    polariser = dlu.Jones.LinearPolariser(angle=0.3)
    plate = dlu.Jones.HalfWavePlate(angle=0.1)
    product = polariser.then(plate)
    chain = dlu.JonesChain({"polariser": polariser, "plate": plate})
    expected = plate.resolve() @ polariser.resolve()
    assert isinstance(product, dlu.JonesMatrix)
    assert np.allclose(product.resolve(), expected)
    assert np.allclose(assert_jittable(lambda model: model.resolve(), chain), expected)
    assert np.allclose(plate.apply(polariser.resolve()), expected)
    assert np.allclose(dlu.Jones.Identity().to_stokes(), np.asarray([1, 0, 0, 0]))
    assert np.allclose(polariser.to_stokes()[0], 0.5)


def test_spatial_broadcast_and_angle_gradients():
    angles = np.asarray([[0.1, 0.2], [0.4, 0.6]])
    definition = dlu.Jones.LinearPolariser(angles)
    output = assert_jittable(lambda model: model.resolve(), definition)
    expected = np.stack(
        [dlu.Jones.linear_polariser(a) for a in angles.reshape(-1)]
    ).reshape(2, 2, 2, 2)
    assert np.allclose(output, expected)
    derivative = jax.grad(lambda a: definition.set(angle=a).resolve()[..., 0, 0].sum())(
        angles
    )
    assert np.allclose(derivative, -np.sin(2 * angles), atol=1e-6)


def test_jones_defers_until_required_context():
    definition = dlu.Jones(angle=zdx.StateRef("angle"))
    assert isinstance(definition.resolve(), dlu.Jones)
    result = definition.resolve(state=zdx.State(angle=0.2))
    assert np.allclose(result, dlu.Jones.linear_polariser(0.2))


def test_chain_reuses_sequence_execution_after_partial_resolution():
    chain = dlu.JonesChain(
        {
            "polariser": dlu.Jones.LinearPolariser(0.1),
            "plate": dlu.Jones.HalfWavePlate(zdx.StateRef("angle")),
        }
    )
    partial = chain.resolve()
    assert isinstance(partial, dlu.JonesChain)
    state = zdx.State(angle=0.3)
    expected = dlu.Jones.half_wave_plate(0.3) @ dlu.Jones.linear_polariser(0.1)
    actual = assert_jittable(lambda model: model.resolve(state=state), partial)
    assert np.allclose(actual, expected)
    result, steps = partial.debug(state=state)
    assert list(steps) == ["input", "polariser", "plate"]
    assert np.allclose(steps["polariser"], dlu.Jones.linear_polariser(0.1))
    assert np.allclose(result, expected)


def test_chain_empty_and_spatial_input_use_matrix_multiplication():
    assert np.array_equal(dlu.JonesChain().resolve(), np.eye(2))
    source = dlu.Jones.linear_polariser(np.asarray([[0.0, 0.1], [0.2, 0.3]]))
    chain = dlu.JonesChain([dlu.Jones.HalfWavePlate(0.4)])
    expected = dlu.Jones.half_wave_plate(0.4) @ source
    assert np.allclose(chain(source), expected)
    assert np.allclose(
        assert_jittable(lambda model, value: model.apply(value), chain, source),
        expected,
    )
    assert not hasattr(chain, "x")


def test_leading_batches_compose_and_convert_stokes_like_scalar_elements():
    angles = np.asarray([[0.1, 0.4, 0.7], [-0.2, 0.3, 0.8]])
    polariser = dlu.Jones.LinearPolariser(angles)
    plate = dlu.Jones.HalfWavePlate(0.2)
    chain = dlu.JonesChain([polariser, plate])
    result, mueller, stokes = assert_jittable(
        lambda model: (model.resolve(), model.mueller(), model.to_stokes()), chain
    )
    scalar = [
        dlu.Jones.LinearPolariser(angle).then(plate) for angle in angles.reshape(-1)
    ]
    assert result.shape == (2, 3, 2, 2)
    assert mueller.shape == (2, 3, 4, 4)
    assert stokes.shape == (2, 3, 4)
    assert np.allclose(
        result, np.stack([x.resolve() for x in scalar]).reshape(result.shape)
    )
    assert np.allclose(
        stokes, np.stack([x.to_stokes() for x in scalar]).reshape(stokes.shape)
    )


@pytest.mark.parametrize(
    "name, parameters, helper, expected",
    [
        ("LinearPolariser", {"angle": 0.0}, "linear_polariser", [[1, 0], [0, 0]]),
        ("HorizontalPolariser", {}, "horizontal_polariser", [[1, 0], [0, 0]]),
        ("VerticalPolariser", {}, "vertical_polariser", [[0, 0], [0, 1]]),
        ("Retarder", {"retardance": 0.0}, "retarder", [[1, 0], [0, 1]]),
        ("HalfWavePlate", {}, "half_wave_plate", [[1, 0], [0, -1]]),
        ("QuarterWavePlate", {}, "quarter_wave_plate", [[1, 0], [0, 1j]]),
        ("Identity", {}, "identity", [[1, 0], [0, 1]]),
    ],
)
def test_named_factory_forms_have_known_matrices(name, parameters, helper, expected):
    factory = getattr(dlu.Jones, name)
    definition = factory(**parameters)
    materialised = factory(**parameters, matrix=True)
    array = getattr(dlu.Jones, helper)(**parameters)

    assert isinstance(definition, dlu.Jones)
    assert isinstance(materialised, dlu.JonesMatrix)
    expected = np.asarray(expected, dtype=complex)
    assert np.allclose(definition(), expected, atol=1e-6)
    assert np.allclose(materialised(), expected, atol=1e-6)
    assert np.allclose(array, expected, atol=1e-6)


@pytest.mark.parametrize(
    "name, helper, sign",
    [("RHCPolariser", "rhc_polariser", 1), ("LHCPolariser", "lhc_polariser", -1)],
)
def test_circular_polarisers_are_orthogonal_projectors(name, helper, sign):
    response = getattr(dlu.Jones, name)()
    matrix = getattr(dlu.Jones, helper)()
    passed = np.array([1.0, sign * 1j]) / np.sqrt(2)
    blocked = np.conj(passed)

    assert isinstance(response, dlu.JonesMatrix)
    assert np.array_equal(response.resolve(), matrix)
    assert np.allclose(matrix @ passed, passed)
    assert np.allclose(matrix @ blocked, 0)
    assert np.allclose(matrix @ matrix, matrix)
    assert np.allclose(response.to_stokes(), np.array([0.5, 0, 0, -sign * 0.5]))


def test_mueller_and_stokes_match_direct_complex_field_evaluation():
    response = dlu.Jones.Retarder(np.array([0.2, 0.8, 1.3]), angle=0.4)
    field = np.array([0.8 + 0.1j, 0.3 - 0.2j])

    def stokes_from_field(value):
        x, y = value[..., 0], value[..., 1]
        cross = x * np.conj(y)
        return np.stack(
            (
                abs(x) ** 2 + abs(y) ** 2,
                abs(x) ** 2 - abs(y) ** 2,
                2 * cross.real,
                2 * cross.imag,
            ),
            -1,
        )

    incoming = stokes_from_field(field)
    expected = stokes_from_field(response.resolve() @ field)
    result = assert_jittable(lambda model: model.to_stokes(incoming), response)
    assert np.allclose(result, expected, atol=2e-6)
    assert np.allclose(result[..., 0], incoming[0])
    assert np.allclose(dlu.Jones.Identity().mueller(), np.eye(4))


def test_retarder_parameters_broadcast_and_differentiate_independently():
    angles = np.array([0.1, 0.4])[:, None]
    retardance = np.array([0.2, 0.7, 1.2])[None, :]
    model = dlu.Jones(angle=angles, retardance=retardance)
    matrix = assert_jittable(lambda value: value.resolve(), model)
    expected = np.stack(
        [
            np.stack([dlu.Jones.retarder(r, a) for r in retardance[0]])
            for a in angles[:, 0]
        ]
    )
    assert matrix.shape == (2, 3, 2, 2)
    assert np.allclose(matrix, expected)
    assert np.allclose(
        np.swapaxes(matrix.conj(), -1, -2) @ matrix, np.eye(2), atol=1e-6
    )

    derivative = jax.jit(jax.grad(lambda r: dlu.Jones.retarder(r, 0.3)[0, 0].real))(0.7)
    assert np.allclose(derivative, -np.sin(0.7) * np.sin(0.3) ** 2, atol=1e-6)


def test_wrapped_expressions_and_factories_resolve_supplied_context():
    state = zdx.State(angle=0.3, matrix=np.array([[1.0, 0], [0, 0]]))
    wrapped = dlu.JonesMatrix(zdx.StateRef("matrix"))
    assert isinstance(wrapped.resolve(), dlu.JonesMatrix)
    assert np.array_equal(wrapped.resolve(state=state), state.matrix)
    plate = dlu.Jones.QuarterWavePlate(zdx.StateRef("angle"), matrix=True, state=state)
    assert np.allclose(plate.resolve(), dlu.Jones.quarter_wave_plate(0.3))


def test_uniform_rotation_preserves_retardance_and_matches_matrix_rotation():
    original = dlu.Jones.HalfWavePlate(0.2)
    rotated = assert_jittable(lambda value: value.rotate(0.3), original)
    matrix = dlu.JonesMatrix(original.resolve()).rotate(0.3)
    assert np.allclose(rotated.angle, 0.5)
    assert np.array_equal(rotated.retardance, original.retardance)
    assert np.allclose(rotated.resolve(), matrix.resolve(), atol=1e-6)
    assert np.allclose(original.angle, 0.2)
    assert np.allclose(rotated.rotate(-0.3).resolve(), original.resolve(), atol=1e-6)


def test_named_arrays_chain_and_then_apply_in_physical_order():
    polariser = dlu.Jones.linear_polariser(0.1)
    plate = dlu.Jones.quarter_wave_plate(0.3)
    initial = np.array([[0.7, 0.2j], [0.1, 0.3]])
    chain = dlu.JonesChain([("polariser", polariser), ("plate", plate)])
    expected = plate @ polariser @ initial
    result, steps = assert_jittable(lambda model: model.debug(initial), chain)
    assert list(steps) == ["input", "polariser", "plate"]
    assert np.allclose(steps["polariser"], polariser @ initial)
    assert np.allclose(result, expected)
    combined = dlu.JonesMatrix(polariser).then(plate)
    assert np.allclose(combined(initial), expected)


def test_rotation_requires_context_before_checking_uniformity():
    model = dlu.Jones(zdx.StateRef("angle"))
    with pytest.raises(zdx.Unresolved, match="context"):
        model.rotate(0.1)
    assert np.allclose(model.rotate(0.1, state=zdx.State(angle=0.2)).angle, 0.3)


def test_composition_populates_links_across_elements():
    shared = zdx.Linked(0.3, key="angle")
    polariser = dlu.Jones.LinearPolariser(shared)
    retarder = dlu.Jones.Retarder(shared.defer(), angle=0.2)
    chain = dlu.JonesChain({"polariser": polariser, "retarder": retarder})
    expected = dlu.Jones.retarder(0.3, 0.2) @ dlu.Jones.linear_polariser(0.3)

    result = assert_jittable(lambda model: model.resolve(), chain)
    assert np.allclose(result, expected)
    assert np.allclose(polariser.then(retarder).resolve(), expected)
    assert isinstance(chain.layers["polariser"].angle, zdx.Linked)


@pytest.mark.parametrize(
    "operation, message",
    [
        (lambda: dlu.Jones(angle=None), "angle cannot be None"),
        (lambda: dlu.Jones.Retarder(None), "requires a numerical retardance"),
        (lambda: dlu.JonesMatrix(np.eye(3)), "shape"),
        (lambda: dlu.JonesMatrix(np.eye(2)).rotate(None), "angle cannot be None"),
        (lambda: dlu.JonesMatrix(np.ones((3, 2, 2))).rotate(0.2), "uniform"),
        (lambda: dlu.JonesMatrix(np.eye(2)).rotate(np.array([0.1, 0.2])), "scalar"),
        (lambda: dlu.Jones.rotate_matrix(np.eye(2), None), "angle cannot be None"),
    ],
)
def test_jones_constructor_and_rotation_validation_tails(operation, message):
    """Coverage tails: useful errors for malformed definitions and rotations."""
    with pytest.raises(ValueError, match=message):
        operation()
