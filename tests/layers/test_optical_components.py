"""Optic inherits independent effects with flat leaves and one sampling context."""

from dataclasses import fields

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_differentiable, assert_jittable, assert_tree_allclose


class CoordinateRamp(zdx.Expression):
    """Use raw SI coordinates and the sampling step without a Sampled base."""

    def evaluate(self, *, coordinates, pixel_scale, **context):
        x = coordinates[..., 0, :, :]
        return 0.5 + 0.02 * x / pixel_scale[..., 0, None, None]


@pytest.mark.parametrize(
    "component, optic",
    [
        (lambda x: dl.BaseOptic(transmission=x), lambda x: dl.Optic(transmission=x)),
        (lambda x: dl.ComplexLayer(phase=x), lambda x: dl.Optic(phase=x)),
        (
            lambda x: dl.RefractiveLayer(thickness=1e-7, n=x),
            lambda x: dl.Optic(thickness=1e-7, n=x),
        ),
        (
            lambda x: dl.JonesLayer(dlu.Jones.HalfWavePlate(angle=x)),
            lambda x: dl.Optic(jones=dlu.Jones.HalfWavePlate(angle=x)),
        ),
    ],
)
@pytest.mark.parametrize("sampled", [False, True])
def test_standalone_components_share_optic_sampling_context(
    component, optic, sampled, make_wavefront
):
    grid = dlu.Grid(n=4, d=2.0, unit="mm")
    wavefront = make_wavefront(wavelength=np.array([5e-7, 7e-7]), grid=grid)
    expression = dlu.Circle(diameter=0.006, edge=1.0) if sampled else CoordinateRamp()
    coordinates, pixel_scale = grid.coordinates, grid.d_si
    values = expression.resolve(coordinates=coordinates, pixel_scale=pixel_scale)
    layer = component(expression)
    expected = component(values)(wavefront)

    standalone = assert_jittable(layer, wavefront)
    combined = assert_jittable(optic(expression), wavefront)
    assert_tree_allclose(standalone, expected, rtol=2e-6)
    assert_tree_allclose(combined, expected, rtol=2e-6)


def test_components_match_combined_scalar_phase(make_wavefront):
    wavefront = make_wavefront(wavelength=np.array([5e-7, 7e-7]))
    scalar = dl.BaseOptic(transmission=0.8, opd=2e-8)
    complex = dl.ComplexLayer(amplitude=0.7, phase=0.2)
    refractive = dl.RefractiveLayer(thickness=1e-7, n=1.5 + 0.02j)
    optic = dl.Optic(
        transmission=0.8,
        opd=2e-8,
        amplitude=0.7,
        phase=0.2,
        thickness=1e-7,
        n=1.5 + 0.02j,
    )

    actual = assert_jittable(optic, wavefront)
    expected_phase = 0.2 + wavefront.wavenumber[:, None, None] * (
        2e-8 + (0.5 + 0.02j) * 1e-7
    )
    expected = wavefront.phasor * 0.8 * 0.7 * np.exp(1j * expected_phase)
    assert np.allclose(actual.phasor, expected, rtol=2e-6)
    separate = refractive(complex(scalar(wavefront)))
    assert_tree_allclose(actual, separate, rtol=2e-6)


@pytest.mark.parametrize("spatial", [False, True])
def test_normalisation_precedes_jones_losses(spatial, make_wavefront):
    wavefront = make_wavefront(wavelength=np.array([5e-7, 7e-7]))
    angle = np.zeros(wavefront.grid.shape) if spatial else 0.0
    optic = dl.Optic(
        transmission=0.4,
        amplitude=0.3,
        normalise=True,
        jones=dlu.Jones.LinearPolariser(angle),
    )
    output = assert_jittable(optic, wavefront)
    assert np.allclose(output.power, 0.5, rtol=2e-6)


def test_common_owner_populates_linked_optical_leaves(make_wavefront):
    shared = zdx.Linked(0.5, key="amplitude")
    optic = dl.Optic(transmission=shared, amplitude=shared.defer())
    wavefront = make_wavefront()
    actual = assert_jittable(optic, wavefront)
    assert np.allclose(actual.phasor, wavefront.phasor * 0.25)


def test_local_application_uses_supplied_frame_once(make_wavefront):
    wavefront = make_wavefront()
    optic = dl.Optic(
        transmission=dlu.Circle(diameter=0.3),
        transform=dlu.Affine(translation=[0.2, 0.0]),
    )
    context = optic.context(wavefront)
    prepared = optic.resolve_local(**context)
    ordinary = assert_jittable(optic, wavefront)
    local = assert_jittable(lambda: optic.apply_local(wavefront, **context))
    reused = assert_jittable(lambda: prepared.apply_scalar(wavefront))
    assert_tree_allclose(ordinary, local)
    assert_tree_allclose(ordinary, reused)


def test_flat_paths_remain_immutable_and_differentiable(make_wavefront):
    wavefront = make_wavefront()
    optic = dl.Optic(amplitude=0.7, phase=0.2)
    updated = optic.set("amplitude", np.asarray(0.4))
    assert np.isclose(optic.amplitude, 0.7)
    assert np.isclose(assert_jittable(updated, wavefront).power, 0.4**2)
    derivative = jax.jit(
        jax.grad(lambda value: optic.set("amplitude", value)(wavefront).power)
    )(np.asarray(0.7))
    assert np.allclose(derivative, 1.4, rtol=2e-6)


def test_optic_inherits_effects_and_stores_flat_leaves():
    optic = dl.Optic(
        transmission=0.8,
        opd=2e-8,
        amplitude=0.7,
        phase=0.2,
        thickness=1e-7,
        n=1.5,
        jones=dlu.Jones.HalfWavePlate(),
    )
    for base in (
        dl.BaseOptic,
        dl.ComplexLayer,
        dl.RefractiveLayer,
        dl.JonesLayer,
        dl.PropagatorLayer,
    ):
        assert isinstance(optic, base)

    leaf_names = {field.name for field in fields(optic)}
    assert {
        "transmission",
        "opd",
        "normalise",
        "amplitude",
        "phase",
        "thickness",
        "n",
        "jones",
        "focal_length",
        "defocus",
        "abcd",
        "inverse",
        "grid_out",
    } <= leaf_names
    assert leaf_names.isdisjoint(
        {"scalar", "complex", "refractive", "polarisation", "propagator"}
    )


def test_chromatic_jones_component_retains_compact_matrices(make_wavefront):
    wavefront = make_wavefront(wavelength=np.array([5e-7, 7e-7]))
    matrices = np.stack(
        [dlu.Jones.half_wave_plate(0.2), dlu.Jones.half_wave_plate(0.4)]
    )

    class ChromaticPlate(zdx.Expression):
        def evaluate(self, wavelength, **context):
            angle = 0.2 + (wavelength - 5e-7) * 1e6
            return dlu.Jones.half_wave_plate(angle)

    # Layer parameters are shared; wavelength-dependent values resolve per sample.
    layer = dl.JonesLayer(ChromaticPlate())
    output = assert_jittable(layer, wavefront)
    assert output.jones.shape == (2, 2, 2)
    assert output.phasor.shape == wavefront.phasor.shape
    assert np.allclose(output.jones, matrices)
    assert_tree_allclose(output, dl.Optic(jones=ChromaticPlate())(wavefront))


@pytest.mark.parametrize("custom_abcd", [False, True])
@pytest.mark.parametrize("polarised", [False, True])
def test_optic_relay_matches_local_effects_then_explicit_propagation(
    custom_abcd, polarised
):
    grid = dlu.Grid((6, 4), (0.02, 0.03), unit="m")
    target = dlu.Grid((5, 4), (0.01, 0.02), unit="m")
    wavelengths = np.array([5e-4, 6e-4])
    field = np.ones((3, 2, 4, 6), complex)
    wavefront = dl.Wavefront(wavelengths, grid, field)
    relay = (
        {"abcd": dlu.ABCD(free_space=0.1)}
        if custom_abcd
        else {"focal_length": 0.7, "defocus": 0.04}
    )
    optic = dl.Optic(
        transmission=0.6,
        phase=0.2,
        normalise=True,
        jones=dlu.Jones.LinearPolariser(0.2) if polarised else None,
        grid_out=target,
        **relay,
    )
    local = optic.set(grid_out=None)(wavefront)
    expected = local.propagate(dlu.Propagator(target, **relay))
    actual = assert_jittable(optic, wavefront, rtol=2e-5, atol=2e-5)
    assert_tree_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert actual.phasor.shape == (3, 2, 4, 5)
    assert_tree_allclose(actual.grid, target)
    assert np.allclose(local.power, 0.5 if polarised else 1.0)
    assert optic.inverse is (None if custom_abcd else False)


def test_optic_without_output_grid_keeps_relay_parameters_inactive(make_wavefront):
    wavefront = make_wavefront()
    optic = dl.Optic(transmission=0.5, phase=0.2, focal_length=2.0, defocus=0.01)
    actual = assert_jittable(optic, wavefront)
    expected = wavefront.set(phasor=wavefront.phasor * 0.5 * np.exp(0.2j))
    assert_tree_allclose(actual, expected)
    with pytest.raises(ValueError, match="grid_out"):
        optic.prepare(wavefront.grid, wavefront.wavelength)


def test_optic_prepare_needs_only_relay_context():
    grid = dlu.Grid(6, 0.02, unit="m")
    target = dlu.Grid(5, 0.01, unit="m")
    wavefront = dl.Wavefront(5e-4, grid)
    optic = dl.Optic(
        phase=zdx.StateRef("unavailable_phase"),
        grid_out=target,
        focal_length=0.7,
    )
    actual = assert_jittable(
        lambda model: model.prepare(grid, wavefront.wavelength)(wavefront.phasor),
        optic,
        rtol=2e-5,
        atol=2e-5,
    )
    expected = dlu.Propagator(target, focal_length=0.7)(wavefront.phasor, grid, 5e-4)
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert isinstance(optic.phase, zdx.Expression)


@pytest.mark.parametrize("parameter", ["focal_length", "defocus"])
def test_optic_relay_updates_are_flat_immutable_and_differentiable(parameter):
    grid = dlu.Grid(6, 0.02, unit="m")
    target = dlu.Grid(5, 0.01, unit="m")
    wavefront = dl.Wavefront(5e-4, grid)
    optic = dl.Optic(phase=0.2, grid_out=target, focal_length=0.7, defocus=0.04)
    original = getattr(optic, parameter)
    updated = optic.set(parameter, original + 0.01)
    assert np.isclose(getattr(optic, parameter), original)
    assert not np.allclose(optic(wavefront).phasor, updated(wavefront).phasor)
    assert_differentiable(
        lambda value: optic.set(parameter, value)(wavefront).phasor,
        original,
        rtol=2e-5,
        atol=2e-5,
    )


def test_mono_jones_layer_does_not_infer_a_chromatic_matrix_axis(make_wavefront):
    matrices = np.broadcast_to(np.eye(2), (3, 2, 2))
    with pytest.raises(ValueError, match="local Jones"):
        dl.JonesLayer(matrices)(make_wavefront())
