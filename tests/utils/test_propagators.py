"""Numerical propagation, shared operations and direct Wavefront inheritance."""

import inspect

import jax
import jax.numpy as np
import pytest
import zodiax as zdx
from abcdLux import asm, lct

import dLux as dl
import dLux.utils as dlu
from dLux.utils import propagation as prop
from tests.helpers import assert_jittable, assert_tree_allclose


def _field(grid):
    """A nonuniform complex field exposing phase and axis-order mistakes."""
    values = np.arange(np.prod(np.asarray(grid.n))).reshape(grid.n[::-1])
    return (1 + values / values.size) * np.exp(0.17j * values)


@pytest.mark.parametrize("batches", [2, 3])
def test_chirp_fusion_preserves_sequential_prefix_batches(batches):
    wavelengths, nx, ny = 2, 4, 3
    first = prop.Chirp(
        tuple(
            np.exp(0.17j * np.arange(wavelengths * n).reshape(wavelengths, n))
            for n in (nx, ny)
        )
    )
    second = prop.Chirp(
        tuple(
            np.exp(
                0.29j
                * np.arange(wavelengths * batches * n).reshape(wavelengths, batches, n)
            )
            for n in (nx, ny)
        )
    )
    # The trailing leading axes represent Jones components, shared by both chirps.
    field = np.ones((wavelengths, batches, 2, 2, ny, nx), dtype=complex)

    for initial, following in ((first, second), (second, first)):
        expected = following(initial(field))
        actual = assert_jittable(
            lambda a, b, value: a.then(b)(value), initial, following, field
        )
        assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("reverse", [False, True])
def test_direct_focal_recipes_match_abcdlux_complex_field(reverse):
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.01, -0.02), "m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), (0.02, 0.01), "m")
    field = _field(source)
    for recipe in (
        prop.Propagator(target, focal_length=0.7, inverse=reverse),
        prop.Propagator(target, defocus=0.04, focal_length=0.7, inverse=reverse),
    ):
        actual, output = recipe(field, source, 5e-4, grid_out=True)
        expected = lct.lct_prop(field, source.axes, target.axes, 5e-4, recipe.matrix)
        assert_tree_allclose(output, target)
        assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("shape", [(7, 6), (6, 7), (5, 7)])
@pytest.mark.parametrize("reverse", [False, True])
def test_fft_matches_mft_at_realised_sampling(shape, reverse):
    source = dl.GridSpec(shape, (0.02, 0.03), (0.01, -0.02), "m")
    request = dl.FFTSpec(pad=2, crop=2, c=(0.001, -0.002))
    fft = prop.FFTPropagator(request, defocus=0.04, focal_length=0.7, inverse=reverse)
    field = _field(source)
    actual, output = fft(field, source, 5e-4, grid_out=True)
    mft = prop.Propagator(output, defocus=0.04, focal_length=0.7, inverse=reverse)
    expected, _ = mft(field, source, 5e-4, grid_out=True)
    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("distance", [-0.1, 0.0, 0.1])
def test_native_free_space_matches_abcdlux(distance):
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.01, -0.02), "m")
    field = _field(source)
    recipe = prop.FreeSpace(distance)
    actual, output = recipe(field, source, 5e-4, grid_out=True)
    expected = asm.asm_prop(
        field, source.axes, 5e-4, distance, npad=source.n, crop=True
    )
    assert_tree_allclose(output, source)
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_spectral_native_free_space_matches_angular_spectrum():
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.01, -0.02), "m")
    field = _field(source)
    spectral = prop.SpectralPropagator(prop.ABCD.free_space(0.1), dl.FFTSpec(), source)
    actual, _ = spectral(field, source, 5e-4, grid_out=True)
    expected, _ = prop.FreeSpace(0.1)(field, source, 5e-4, grid_out=True)
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("distance", [-1e-5, 0.0, 1e-5])
def test_spectral_negative_magnification_preserves_abcdlux_phase(distance):
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.0, 0.0), "m")
    frequency = source.frequency()
    matrix = prop.ABCD(np.asarray([[-1.0, distance], [0.0, -1.0]]))
    recipe = prop.SpectralPropagator(matrix, frequency, source)
    field = _field(source)
    actual, _ = recipe(field, source, 5e-4, grid_out=True)
    expected = lct.lct_prop_2ft(
        field, source.axes, source.axes, frequency.axes, 5e-4, matrix.matrix
    )
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_propagation_is_jittable_and_retains_physical_derivatives():
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.01, -0.02), "m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), (0.02, 0.01), "m")
    field = _field(source)
    recipe = prop.Propagator(target, defocus=0.04, focal_length=0.7)

    def prepared_output(distance):
        model = recipe.set(defocus=distance)
        return model(field, source, 5e-4, grid_out=True)[0]

    actual = assert_jittable(prepared_output, np.asarray(0.04), rtol=2e-5, atol=2e-5)
    expected = recipe(field, source, 5e-4, grid_out=True)[0]
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    derivative = jax.jacfwd(prepared_output)(np.asarray(0.04))
    step = 1e-5
    difference = (prepared_output(0.04 + step) - prepared_output(0.04 - step)) / (
        2 * step
    )
    assert np.max(np.abs(derivative)) > 0.1
    assert np.allclose(derivative, difference, rtol=3e-3, atol=3e-3)


def test_zero_defocus_matches_fraunhofer_with_live_defocus_derivative():
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    field = _field(source)
    expected, _ = prop.Propagator(target, 0.7)(field, source, 5e-4, grid_out=True)
    model = prop.Propagator(target, focal_length=0.7, defocus=0.0)
    actual, _ = model(field, source, 5e-4, grid_out=True)
    derivative = jax.jacfwd(
        lambda z: model.set(defocus=z)(field, source, 5e-4, grid_out=True)[0]
    )(0.0)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(derivative)) > 0.1


@pytest.mark.parametrize("polarised", [False, True])
@pytest.mark.parametrize("populations", [(2, 3), (2, 2)])
@pytest.mark.parametrize("kind", ["direct", "fft", "spectral", "space"])
def test_layer_batch_values_match_independent_scalar_propagation(
    polarised, populations, kind
):
    wavelengths, batches = populations
    grid = dl.GridSpec(
        (5, 6),
        (0.02, 0.03),
        np.stack((np.linspace(-0.01, 0.01, batches), np.zeros(batches)), axis=-1),
        "m",
    )
    lam = np.linspace(4e-4, 6e-4, wavelengths)
    wavefront_type = dl.PolarisedWavefront if polarised else dl.Wavefront
    wavefront = wavefront_type(lam, grid)
    target = dl.GridSpec((7, 4), (0.01, 0.02), unit="m")
    layer = {
        "direct": dl.Propagator(target, defocus=0.04, focal_length=0.7),
        "fft": dl.FFTPropagator(defocus=0.04, focal_length=0.7),
        "spectral": dl.SpectralPropagator(
            dl.ABCD.free_space(0.1), dl.FFTSpec(), target
        ),
        "space": dl.FreeSpace(0.1),
    }[kind]
    output = assert_jittable(layer, wavefront, rtol=2e-5, atol=2e-5)
    for w in range(wavelengths):
        for b in range(batches):
            local_grid = grid.set(c=grid.c[b])
            local = wavefront_type(lam[w], local_grid, wavefront.phasor[w, b])
            expected = layer(local)
            assert np.allclose(
                output.phasor[w, b], expected.phasor, rtol=2e-5, atol=2e-5
            )


@pytest.mark.parametrize("kind", ["direct", "fft", "spectral", "space", "lct"])
def test_array_default_and_opt_in_grid(kind):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    field = _field(source)
    definitions = {
        "direct": prop.Propagator(target, focal_length=0.7),
        "fft": prop.FFTPropagator(focal_length=0.7),
        "spectral": prop.SpectralPropagator(
            prop.ABCD.free_space(0.1), dl.FFTSpec(), target
        ),
        "space": prop.FreeSpace(0.1),
        "lct": prop.LCT(prop.ABCD.free_space(0.1), target),
    }
    definition = definitions[kind]
    output = assert_jittable(definition, field, source, 5e-4)
    returned, grid = assert_jittable(
        lambda x: definition(x, source, 5e-4, grid_out=True), field
    )
    assert isinstance(output, jax.Array)
    assert np.allclose(output, returned)
    assert returned.shape == grid.shape


@pytest.mark.parametrize("kind", ["direct", "fft", "spectral", "space"])
@pytest.mark.parametrize("polarised", [False, True])
def test_layer_inherits_numerics_and_preserves_metadata(kind, polarised):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    constructors = {
        "direct": (dl.Propagator, prop.Propagator, (target,), {"focal_length": 0.7}),
        "fft": (dl.FFTPropagator, prop.FFTPropagator, (), {"focal_length": 0.7}),
        "spectral": (
            dl.SpectralPropagator,
            prop.SpectralPropagator,
            (prop.ABCD.free_space(0.1), dl.FFTSpec(), target),
            {},
        ),
        "space": (dl.FreeSpace, prop.FreeSpace, (0.1,), {}),
    }
    layer_type, numeric_type, args, kwargs = constructors[kind]
    layer, numeric = layer_type(*args, **kwargs), numeric_type(*args, **kwargs)
    wavefront_type = dl.PolarisedWavefront if polarised else dl.Wavefront
    wavefront = wavefront_type(
        np.asarray([5e-4, 6e-4]), source, weight=np.array([0.3, 0.7])
    )
    result = assert_jittable(layer, wavefront)
    expected, grid = numeric(
        wavefront.phasor, source, wavefront.wavelength, grid_out=True
    )
    assert isinstance(layer, numeric_type)
    assert np.allclose(result.phasor, expected, rtol=2e-5, atol=2e-5)
    assert_tree_allclose(result.grid, grid)
    assert_tree_allclose(result.weight, wavefront.weight)
    assert_tree_allclose(result.wavelength, wavefront.wavelength)
    assert type(result) is type(wavefront)
    assert "model" not in type(layer).__dataclass_fields__


@pytest.mark.parametrize("layer", [False, True])
def test_expression_resolution_and_missing_context(layer):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    field = _field(source)
    wavefront = dl.Wavefront(5e-4, source, field)
    cls = dl.FreeSpace if layer else prop.FreeSpace
    definition = cls(zdx.StateRef("distance"))
    args = (wavefront,) if layer else (field, source, 5e-4)
    with pytest.raises(ValueError, match="Unresolved.*distance"):
        definition(*args)
    for distance in (0.1, 0.2):
        actual = assert_jittable(
            lambda p: p(*args, state=zdx.State(distance=distance)), definition
        )
        expected = cls(distance)(*args)
        assert_tree_allclose(actual, expected)
    assert isinstance(definition.distance, zdx.Expression)


def test_linked_parameters_resolve_at_common_sequence_scope():
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    wavefront = dl.Wavefront(5e-4, source, _field(source))
    shared = zdx.Linked(0.7, key="focal_length")
    definition = dl.Propagator(target, focal_length=shared.defer())
    with pytest.raises((ValueError, KeyError)):
        definition(wavefront)
    system = dl.Sequential(
        [("lens", dl.Optic(amplitude=shared)), ("focus", definition)]
    )
    changed = system.set("lens.amplitude.value", 0.8)
    expected = dl.Propagator(target, 0.8)(dl.Optic(amplitude=0.8)(wavefront))
    assert_tree_allclose(assert_jittable(changed, wavefront), expected)


def test_named_sequence_is_shared_by_arrays_and_layers():
    assert dl.Sequential is dlu.Sequential
    field = np.arange(12.0).reshape(3, 4).astype(complex)
    fft = prop.FFT((4, 3))
    inverse = prop.FFT((4, 3), inverse=True)
    sequence = dl.Sequential([("forward", fft), ("inverse", inverse)])
    assert not isinstance(fft, (zdx.Transform, zdx.Expression))
    assert np.allclose(assert_jittable(sequence, field), field, rtol=1e-6, atol=1e-6)
    result, states = sequence.debug(field)
    assert tuple(states) == ("input", "forward", "inverse")
    assert np.allclose(result, field, rtol=1e-6, atol=1e-6)
    assert_tree_allclose(sequence.remove("inverse").forward, fft)
    nested = dl.Sequential([("inner", sequence)])
    assert nested.inner.forward is fft
    assert np.allclose(nested(field), field, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "focal_length,unit", [(None, "rad"), (None, "mas"), (0.7, "m"), (0.7, "mm")]
)
def test_focal_units_checked_at_construction_and_after_updates(focal_length, unit):
    target = dl.GridSpec((5, 8), 0.01, unit=unit)
    layer = dl.Propagator(target, focal_length=focal_length)
    wrong = "m" if focal_length is None else "rad"
    with pytest.raises(ValueError):
        dl.Propagator(target.set(unit=wrong), focal_length=focal_length)
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    with pytest.raises(ValueError):
        layer.set("grid_out.unit", wrong)(dl.Wavefront(5e-4, source))


def test_custom_abcd_excludes_conjugate_parameters():
    target = dl.GridSpec((5, 8), 0.01, unit="m")
    for cls, grid in ((prop.Propagator, target), (prop.FFTPropagator, dl.FFTSpec())):
        for kwargs in ({"focal_length": 0.7}, {"defocus": 0.0}):
            with pytest.raises(ValueError, match="exclusive"):
                cls(grid, abcd=prop.ABCD.free_space(0.1), **kwargs)


def test_serialisation_preserves_inherited_physical_definition(tmp_path):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    layer = dl.Propagator(target, focal_length=0.7, defocus=0.04)
    path = tmp_path / "propagator.dlux"
    zdx.save(path, layer)
    loaded = zdx.load(path)
    assert type(loaded) is type(layer)
    wavefront = dl.Wavefront(5e-4, source, _field(source))
    assert_tree_allclose(loaded(wavefront), layer(wavefront))
    assert_tree_allclose(
        loaded.set(defocus=0.05)(wavefront), layer.set(defocus=0.05)(wavefront)
    )


@pytest.mark.parametrize("kind", ["direct", "fft", "spectral"])
def test_state_matrix_batches_keep_their_matrix_axes(kind):
    source = dl.GridSpec(6, 0.02, unit="m")
    target = dl.GridSpec(5, 0.01, unit="m")
    state = zdx.State(matrix=dl.ABCD.free_space(np.array([0.1, 0.2])).matrix)
    matrix = state.ref("matrix")
    constructors = {
        "direct": lambda abcd: dl.Propagator(target, abcd=abcd),
        "fft": lambda abcd: dl.FFTPropagator(abcd=abcd),
        "spectral": lambda abcd: dl.SpectralPropagator(abcd, dl.FFTSpec(), target),
    }
    layer = constructors[kind](matrix)
    wavelengths = np.array([5e-4, 6e-4])
    wavefront = dl.Wavefront(wavelengths, source)
    output = assert_jittable(lambda wf: layer(wf, state=state), wavefront)
    for i in range(2):
        expected = constructors[kind](state.matrix[i])(
            dl.Wavefront(wavelengths[i], source)
        )
        assert np.allclose(output.phasor[i], expected.phasor, rtol=2e-5, atol=2e-5)
    with pytest.raises(ValueError, match="Unresolved"):
        layer(wavefront)


@pytest.mark.parametrize("distance", [0.1, -0.1])
def test_custom_fft_matches_lct_with_explicit_counts_and_signed_distance(distance):
    source = dl.GridSpec((7, 6), (0.02, 0.03), (0.01, -0.02), "m")
    request = dl.FFTSpec(n=(10, 9), c=(0.001, -0.002))
    abcd = dl.ABCD.free_space(distance)
    definition = prop.FFTPropagator(request, abcd=abcd)
    field = _field(source)
    actual, output = definition(field, source, 5e-4, grid_out=True)
    expected = prop.LCT(abcd, output)(field, source, 5e-4)
    assert output.n == (10, 9)
    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_singular_factorisations_report_their_required_coefficient():
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    field = _field(source)
    with pytest.raises(ValueError, match="ABCD.b"):
        prop.LCT(dl.ABCD.identity(), source)(field, source, 5e-4)
    with pytest.raises(ValueError, match="ABCD.a"):
        prop.SpectralPropagator(dl.ABCD.fraunhofer(0.7), dl.FFTSpec(), source)(
            field, source, 5e-4
        )


def test_unresolved_output_sampling_names_the_missing_context():
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    state = zdx.State(spacing=0.01)
    target = dl.GridSpec((5, 8), d=state.ref("spacing"), unit="rad")
    wavefront = dl.Wavefront(5e-4, source, _field(source))
    numerical = prop.Propagator(target)
    layer = dl.Propagator(target)
    for call in (
        lambda **context: numerical(wavefront.phasor, source, 5e-4, **context),
        lambda **context: layer(wavefront, **context).phasor,
    ):
        with pytest.raises(ValueError, match="Unresolved.*grid_out"):
            call()
        actual = assert_jittable(lambda: call(state=state), rtol=2e-5, atol=2e-5)
        expected = prop.Propagator(target.set(d=0.01))(wavefront.phasor, source, 5e-4)
        assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("cls", [prop.Propagator, prop.FFTPropagator])
def test_default_inverse_is_hidden_and_explicit_reverse_is_printed(cls):
    target = dl.GridSpec((5, 6), 0.01, unit="m")
    grid = target if cls is prop.Propagator else dl.FFTSpec()
    assert inspect.signature(cls).parameters["inverse"].default is False
    forward = cls(grid, focal_length=0.7)
    reverse = cls(grid, focal_length=0.7, inverse=True)
    custom = cls(grid, abcd=prop.ABCD.free_space(0.1), inverse=True)
    assert forward.inverse is False
    assert reverse.inverse is True
    assert custom.inverse is None
    assert "inverse=" not in repr(forward)
    assert "inverse=True" in repr(reverse)
    assert "_sampling" not in repr(forward)


@pytest.mark.parametrize("cls", [prop.Propagator, prop.FFTPropagator])
@pytest.mark.parametrize("distance", [-0.1, 0.1])
def test_custom_matrix_owns_direction_independently_of_inverse(cls, distance):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    grid = target if cls is prop.Propagator else dl.FFTSpec()
    abcd = prop.ABCD.free_space(distance)
    field = _field(source)
    expected = cls(grid, abcd=abcd)(field, source, 5e-4)
    for inverse in (None, False, True):
        definition = cls(grid, abcd=abcd, inverse=inverse)
        assert definition.inverse is None
        assert np.array_equal(definition.matrix, abcd.matrix)
        assert np.allclose(definition(field, source, 5e-4), expected)


@pytest.mark.parametrize("shape", [(7,), (7, 6)])
def test_custom_fft_mixed_sign_matrix_batches_match_independent_lct_under_jit(shape):
    source = dl.GridSpec(shape, 0.02, unit="m")
    field = _field(source)
    fields = np.stack([field, field * np.exp(0.3j)])
    distances = np.asarray([-0.1, 0.2])
    request = dl.FFTSpec(pad=2, crop=2)

    def propagate(distance):
        definition = prop.FFTPropagator(request, abcd=prop.ABCD.free_space(distance))
        return definition(fields, source, 5e-4, grid_out=True)

    actual, output = assert_jittable(propagate, distances, rtol=3e-5, atol=3e-5)
    for index, distance in enumerate(distances):
        local_grid = output.set(d=output.d[index], c=output.c[index])
        expected = prop.LCT(prop.ABCD.free_space(distance), local_grid)(
            fields[index], source, 5e-4
        )
        assert np.allclose(actual[index], expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("kind", ["direct", "fft", "space"])
def test_prepared_operations_reuse_kernels_and_keep_parameter_derivatives(kind):
    source = dl.GridSpec((7, 6), (0.02, 0.03), unit="m")
    target = dl.GridSpec((5, 8), (0.01, 0.02), unit="m")
    constructors = {
        "direct": lambda distance: prop.Propagator(target, 0.7, defocus=distance),
        "fft": lambda distance: prop.FFTPropagator(dl.FFTSpec(pad=2), 0.7, distance),
        "space": prop.FreeSpace,
    }
    definition = constructors[kind](0.04)
    prepared, output_grid = definition.prepare(source, 5e-4, grid_out=True)
    assert isinstance(prepared, prop.MFT if kind == "direct" else dlu.Sequential)
    field = _field(source)
    for incident in (field, field * np.exp(0.3j)):
        actual = assert_jittable(prepared, incident, rtol=2e-5, atol=2e-5)
        expected, grid = definition(incident, source, 5e-4, grid_out=True)
        assert_tree_allclose(grid, output_grid)
        assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def prepared_output(distance):
        return constructors[kind](distance).prepare(source, 5e-4)(field)

    derivative = jax.jit(jax.jacfwd(prepared_output))(0.04)
    step = 1e-4
    difference = (prepared_output(0.04 + step) - prepared_output(0.04 - step)) / (
        2 * step
    )
    assert np.max(np.abs(derivative)) > 1e-3
    assert np.allclose(derivative, difference, rtol=5e-3, atol=5e-3)
