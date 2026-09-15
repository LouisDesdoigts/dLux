"""Layer-owned context and interchangeable array/expression operands."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
from tests.helpers import assert_jittable, assert_tree_allclose


class ScaledJones(zdx.Expression):
    scale: object

    def evaluate(self, **context):
        return zdx.resolve(self.scale, **context) * np.eye(2)


@pytest.mark.parametrize("layer_type", [dl.TransmissiveLayer, dl.Optic])
def test_transmission_is_real_for_arrays_expressions_and_updates(
    layer_type, make_wavefront
):
    wavefront = make_wavefront()
    with pytest.raises(TypeError, match="transmission must be real"):
        layer_type(transmission=np.array(1j))
    layer = layer_type(transmission=zdx.StateRef("transmission"))
    state = zdx.State(transmission=0.4)
    output = assert_jittable(lambda model: model(wavefront, state=state), layer)
    assert_tree_allclose(output, layer_type(transmission=0.4)(wavefront))
    for model, context in [
        (layer, {"state": zdx.State(transmission=np.array(1j))}),
        (layer_type(transmission=1.0).set(transmission=np.array(1j)), {}),
    ]:
        with pytest.raises(TypeError, match="transmission must be real"):
            model(wavefront, **context)


@pytest.mark.parametrize("layer_type", [dl.SparseOptic, dl.SparseDynamicOptic])
def test_sparse_polynomial_selects_one_coefficient_vector_per_aperture(
    layer_type, make_wavefront
):
    wavefront = make_wavefront()
    coefficients = np.array([[0.1], [0.3]])
    polynomial = dl.Polynomial(degree=0, ndim=2, coeffs=zdx.StateRef("coeffs"))
    optic = layer_type([[-0.2, 0.0], [0.2, 0.0]], phase=polynomial)
    evaluate = lambda coeffs: optic(wavefront, state=zdx.State(coeffs=coeffs)).phasor
    result = assert_jittable(evaluate, coefficients)
    expected = np.exp(1j * coefficients[:, 0, None, None]) * wavefront.phasor
    assert result.shape == (2,) + wavefront.phasor.shape
    assert np.allclose(result, expected)
    derivative = jax.jit(jax.grad(lambda coeffs: evaluate(coeffs).real.sum()))(
        coefficients
    )
    expected_derivative = -np.sin(coefficients) * wavefront.phasor.real.sum()
    assert np.allclose(derivative, expected_derivative, atol=1e-6)


def test_optic_resolves_links_between_scalar_and_jones_terms(make_wavefront):
    wavefront = make_wavefront()
    shared = zdx.Linked(zdx.StateRef("transmission"), key="transmission")
    optic = dl.Optic(transmission=shared, jones=ScaledJones(shared.defer()))
    state = zdx.State(transmission=0.5)
    result = assert_jittable(lambda model: model(wavefront, state=state), optic)
    expected = dl.Optic(transmission=0.5, jones=0.5 * np.eye(2))(wavefront)
    assert_tree_allclose(result, expected)
    assert np.allclose(optic.phasor(wavefront, state=state), 0.5)


@pytest.mark.parametrize("uniform", [False, True])
def test_polarising_optics_resolve_jones_and_orientation(uniform, make_wavefront):
    wavefront = make_wavefront(wavelength=np.array([500e-9, 700e-9]))
    matrix = np.array([[1.0, 0.0], [0.0, 0.3]])
    state = zdx.State(jones=matrix, angle=0.2)
    if uniform:
        optic = dl.UniformPolarisingOptic(zdx.StateRef("jones"), zdx.StateRef("angle"))
        reference = dl.UniformPolarisingOptic(matrix, 0.2)
    else:
        optic = dl.PolarisingOptic(zdx.StateRef("jones"))
        reference = dl.PolarisingOptic(matrix)
    output = assert_jittable(lambda model: model(wavefront, state=state), optic)
    assert_tree_allclose(output, reference(wavefront))


@pytest.mark.parametrize("field_type", ["wavefront", "intensity"])
def test_normalise_resolves_target_value(field_type, make_wavefront):
    target = make_wavefront()
    mode = "power"
    if field_type == "intensity":
        target = target.to_intensity()
    layer = dl.Normalise(mode, zdx.StateRef("target_power"))
    output = assert_jittable(
        lambda model: model(target, state=zdx.State(target_power=2.0)), layer
    )
    assert_tree_allclose(output, dl.Normalise(mode, 2.0)(target))


def test_detector_soft_shape_receives_grid_sampling(make_psf):
    intensity = make_psf()
    shape = dl.Circle(0.6, edge=1.0)
    response = dl.Sensitivity(shape)
    output = assert_jittable(response, intensity)
    assert np.allclose(
        output.data, intensity.data * shape.evaluate(grid=intensity.grid)
    )
    assert set(response.context(intensity)) == {"grid"}


def test_specialised_contexts_keep_their_query_meaning(make_psf, make_wavefront):
    intensity = make_psf(data=np.arange(64.0).reshape(8, 8))
    gain = dl.Gain(dl.Polynomial(1, [1.0, 0.1]))
    result = assert_jittable(gain, intensity)
    assert np.allclose(result.data, intensity.data * (1 + 0.1 * intensity.data))
    assert set(gain.context(intensity)) == {"coordinates"}

    dynamic = dl.DynamicOptic(transmission=dl.Circle(0.6, edge=1.0))
    context = dynamic.context(make_wavefront())
    assert set(context) == {"coordinates", "pixel_scale", "wavelengths"}


def test_jitter_profile_uses_local_pixels_and_explicit_incident_data(make_psf):
    class SignalWidth(zdx.Expression):
        def evaluate(self, *, data, **context):
            return np.mean(data) / 4

    intensity = make_psf(data=np.full((8, 8), 2.0))
    jitter = dl.Jitter(dl.Gaussian(SignalWidth()), kernel_size=5, oversample=2)
    kernel = assert_jittable(lambda model: model.kernel(intensity), jitter)
    assert np.allclose(
        kernel, dl.Jitter(dl.Gaussian(0.5), kernel_size=5, oversample=2).kernel()
    )
