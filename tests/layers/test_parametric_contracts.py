"""Layer-owned context and interchangeable array/expression operands."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable, assert_tree_allclose


class ScaledJones(zdx.Expression):
    scale: object

    def evaluate(self, **context):
        return zdx.resolve(self.scale, **context) * np.eye(2)


@pytest.mark.parametrize("layer_type", [dl.Optic])
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
        (layer_type(transmission=1.0).set("transmission", np.array(1j)), {}),
    ]:
        with pytest.raises(TypeError, match="transmission must be real"):
            model(wavefront, **context)


@pytest.mark.parametrize("layer_type", [dl.SparseOptic])
def test_sparse_polynomial_selects_one_coefficient_vector_per_aperture(
    layer_type, make_wavefront
):
    wavefront = make_wavefront()
    coefficients = np.array([[0.1], [0.3]])
    polynomial = dlu.Polynomial(degree=0, ndim=2, coeffs=zdx.StateRef("coeffs"))
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


@pytest.mark.parametrize("field_type", ["wavefront", "intensity"])
def test_normalise_resolves_target_value(field_type, make_wavefront):
    target = make_wavefront()
    mode = "power"
    if field_type == "intensity":
        target = target.to_intensity()
    layer = dl.NormaliseLayer(mode, zdx.StateRef("target_power"))
    output = assert_jittable(
        lambda model: model(target, state=zdx.State(target_power=2.0)), layer
    )
    assert_tree_allclose(output, dl.NormaliseLayer(mode, 2.0)(target))


def test_detector_soft_shape_receives_grid_sampling(make_psf):
    intensity = make_psf()
    shape = dlu.Circle(0.6, edge=1.0)
    response = dl.SensitivityLayer(shape)
    output = assert_jittable(response, intensity)
    assert np.allclose(
        output.data, intensity.data * shape.evaluate(grid=intensity.grid)
    )
    assert set(response.context(intensity)) == {"grid"}


def test_specialised_contexts_keep_their_query_meaning(make_psf, make_wavefront):
    intensity = make_psf(data=np.arange(64.0).reshape(8, 8))
    gain = dl.GainLayer(dlu.Polynomial(1, [1.0, 0.1]))
    result = assert_jittable(gain, intensity)
    assert np.allclose(result.data, intensity.data * (1 + 0.1 * intensity.data))
    assert set(gain.context(intensity)) == {"coordinates"}

    dynamic = dl.Optic(transmission=dlu.Circle(0.6, edge=1.0))
    context = dynamic.context(make_wavefront())
    assert set(context) == {"coordinates", "pixel_scale", "wavelength", "wavelengths"}


def test_sparse_coordinate_maps_resolve_state_parameters(make_wavefront):
    wavefront = make_wavefront()
    shifts = np.array([[0.05, 0.0], [-0.03, 0.0]])
    transform = dlu.Affine(translation=zdx.StateRef("shifts"))
    optic = dl.SparseOptic(
        [[-0.2, 0.0], [0.2, 0.0]], transmission=dlu.Circle(0.4), transform=transform
    )
    state = zdx.State(shifts=shifts)
    result = assert_jittable(lambda x: optic(x, state=state), wavefront)
    reference = optic.set(transform=dlu.Affine(translation=shifts))(wavefront)
    assert_tree_allclose(result, reference)


def test_sparse_invalid_coefficients_and_jones_axes(make_wavefront):
    """Conditional tails for explicitly unbound or incorrectly batched leaves."""
    field = make_wavefront()
    centers = [[-0.2, 0.0], [0.2, 0.0]]
    basis = dlu.Basis(np.ones((1, 8, 8)), empty=True)
    # Selection preserves an unbound definition; evaluating it still needs coefficients.
    selected, batched = dl.SparseOptic._slice(basis, 0, 2)
    assert selected is basis and not batched
    with pytest.raises(ValueError, match="coeffs"):
        dl.SparseOptic(centers, opd=basis.set(coeffs=np.ones((3, 1))))(field)
    with pytest.raises(ValueError, match="Jones aperture"):
        dl.SparseOptic(centers, jones=np.broadcast_to(np.eye(2), (3, 2, 2)))(field)
