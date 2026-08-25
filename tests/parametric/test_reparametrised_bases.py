"""Integration tests for reparametrised coefficient leaves."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class _CoordinateCoeffs(dl.Parametric):
    """Coefficient value requiring local coordinate context."""

    @property
    def shape(self):
        """Return the single physical coefficient shape."""
        return (1,)

    def evaluate(self, *, coordinates, **context):
        """Return one coefficient derived from local coordinates."""
        return np.asarray([1e-8 * (coordinates[0, 0, 0] + 1)])


def _reparametrise(coeffs):
    """Represent an array exactly through an identity latent map."""
    coeffs = np.asarray(coeffs, dtype=float)
    origin = np.zeros_like(coeffs)
    matrix = np.eye(coeffs.size, dtype=coeffs.dtype)
    return dl.Reparametrisation(origin, matrix, coeffs.reshape(-1))


def _explicit_case():
    vectors = np.arange(24.0).reshape(2, 3, 4) / 24
    coeffs = np.asarray([0.2, -0.1])
    return (
        dl.Basis(vectors, coeffs),
        dl.Basis(vectors, _reparametrise(coeffs)),
        {},
    )


def _fourier_case():
    coeffs = np.linspace(-0.2, 0.3, 12).reshape(3, 4)
    return (
        dl.FourierBasis((6, 5), (3, 4), coeffs),
        dl.FourierBasis((6, 5), (3, 4), _reparametrise(coeffs)),
        {},
    )


def _pasted_case():
    spec = dl.PasteSpec(
        n=(6, 4),
        shape=(2, 2),
        starts=[[0, 0], [4, 2]],
        offsets=np.zeros((2, 2)),
        d=np.ones(2),
    )
    modes = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    vectors = np.broadcast_to(modes, (2, 2, 2, 2))
    coeffs = np.asarray([[0.1, 0.2], [-0.3, 0.4]])
    return (
        dl.PastedBasis(vectors, spec, coeffs),
        dl.PastedBasis(vectors, spec, _reparametrise(coeffs)),
        {},
    )


def _spline_case():
    coeffs = np.linspace(-0.2, 0.3, 6).reshape(3, 2)
    return (
        dl.SplineBasis((6, 5), (2, 3), coeffs, method="linear"),
        dl.SplineBasis((6, 5), (2, 3), _reparametrise(coeffs), method="linear"),
        {},
    )


def _spectral_polynomial_case():
    coeffs = np.asarray([0.2, -0.1])
    context = {"wavelengths": np.linspace(500e-9, 700e-9, 5)}
    return (
        dl.SpectralPolynomial(degree=2, coeffs=coeffs, normalise=False),
        dl.SpectralPolynomial(degree=2, coeffs=_reparametrise(coeffs), normalise=False),
        context,
    )


def _spectral_basis_case():
    vectors = np.stack((np.ones(5), np.linspace(-0.5, 0.5, 5)))
    coeffs = np.asarray([1.0, 0.2])
    return (
        dl.SpectralBasis(vectors, coeffs, normalise=False),
        dl.SpectralBasis(vectors, _reparametrise(coeffs), normalise=False),
        {},
    )


def _polynomial_case():
    coeffs = np.asarray([0.7, -0.2, 0.1])
    context = {"variables": np.linspace(-1.0, 1.0, 7)}
    return (
        dl.Polynomial(degree=2, coeffs=coeffs),
        dl.Polynomial(degree=2, coeffs=_reparametrise(coeffs)),
        context,
    )


def _dynamic_zernike_case():
    coeffs = np.asarray([0.2, -0.1])
    grid = dl.GridSpec(n=8, diam=1.0, unit="m").broadcast(2)
    context = {"coordinates": grid.coordinates}
    return (
        dl.DynamicZernikeBasis(js=[1, 4], coeffs=coeffs, diameter=0.8),
        dl.DynamicZernikeBasis(js=[1, 4], coeffs=_reparametrise(coeffs), diameter=0.8),
        context,
    )


_CASES = {
    "basis": _explicit_case,
    "fourier": _fourier_case,
    "pasted": _pasted_case,
    "spline": _spline_case,
    "spectral-polynomial": _spectral_polynomial_case,
    "spectral-basis": _spectral_basis_case,
    "polynomial": _polynomial_case,
    "dynamic-zernike": _dynamic_zernike_case,
}


@pytest.fixture(params=_CASES, ids=_CASES)
def basis_pair(request):
    """Return equivalent array- and latent-coefficient bases with context."""
    return _CASES[request.param]()


def test_reparametrised_basis_evaluation_jit_and_gradient(basis_pair):
    array_basis, reparametrised, context = basis_pair

    expected = array_basis.evaluate(**context)
    output = assert_jittable(
        lambda model: model.evaluate(**context),
        reparametrised,
        rtol=1e-5,
        atol=1e-5,
    )
    gradient = assert_differentiable(
        lambda latent: reparametrised.set("latent", latent).evaluate(**context),
        reparametrised.coeffs.latent,
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert np.any(np.abs(gradient) > 0)


def test_coefficient_aliases_retain_parametrisation_identity():
    coeffs = _reparametrise([0.2, -0.1])
    basis = dl.Basis(np.arange(24.0).reshape(2, 3, 4), coeffs)

    assert basis.coeffs is coeffs
    assert basis.c is coeffs
    assert basis.alpha is coeffs
    with pytest.warns(DeprecationWarning):
        assert basis.coefficients is coeffs


def test_nested_latent_paths_support_get_set_and_add():
    coeffs = _reparametrise([0.2, -0.1])
    basis = dl.Basis(np.arange(24.0).reshape(2, 3, 4), coeffs)
    latent = np.asarray([-0.4, 0.3])
    delta = np.asarray([0.05, -0.02])

    raised = basis.set("latent", latent)
    qualified = basis.set("coeffs.latent", latent)
    raised_added = basis.add("latent", delta)
    qualified_added = basis.add("coeffs.latent", delta)

    assert np.array_equal(basis.get("latent"), coeffs.latent)
    assert np.array_equal(basis.get("coeffs.latent"), coeffs.latent)
    assert np.array_equal(raised.coeffs.latent, latent)
    assert np.array_equal(qualified.coeffs.latent, latent)
    assert np.allclose(raised_added.coeffs.latent, coeffs.latent + delta)
    assert np.allclose(qualified_added.coeffs.latent, coeffs.latent + delta)
    assert np.array_equal(basis.coeffs.latent, coeffs.latent)


def test_setting_coeffs_replaces_parametrisation_without_mutation():
    vectors = np.arange(24.0).reshape(2, 3, 4)
    coeffs = _reparametrise([0.2, -0.1])
    basis = dl.Basis(vectors, coeffs)
    replacement = np.asarray([-0.3, 0.4])

    replaced = basis.set("coeffs", replacement)
    expected = dl.Basis(vectors, replacement)

    assert isinstance(basis.coeffs, dl.Reparametrisation)
    assert basis.coeffs is coeffs
    assert np.array_equal(basis.coeffs.latent, np.asarray([0.2, -0.1]))
    assert np.array_equal(replaced.coeffs, replacement)
    assert np.allclose(replaced.evaluate(), expected.evaluate())


def test_single_mode_reparametrisation_preserves_paired_vectorisation():
    vectors = np.arange(12.0).reshape(1, 3, 2, 2)
    latent = np.asarray([[1.0], [2.0], [3.0]])
    coeffs = dl.Reparametrisation(np.zeros(1), np.ones((1, 1)), latent)
    basis = dl.Basis(vectors, coeffs, shape=(1,))

    output = assert_jittable(lambda model: model.evaluate(), basis)
    expected = latent[:, 0, None, None] * vectors[0]

    assert output.shape == (3, 2, 2)
    assert np.allclose(output, expected)


def test_selection_is_a_settable_coefficient_leaf():
    origin = np.arange(6.0).reshape(2, 3) / 10
    mask = np.asarray([[True, False, True], [False, True, False]])
    selection = dl.Selection(origin, mask)
    basis = dl.FourierBasis((6, 5), (2, 3), selection)
    latent = np.asarray([0.2, -0.1, 0.3])

    updated = basis.set("latent", latent)
    expected = basis.set("coeffs", selection.to_coeffs(latent))
    output = assert_jittable(lambda model: model.evaluate(), updated)

    assert isinstance(basis.coeffs, dl.Selection)
    assert np.array_equal(updated.get("coeffs.latent"), latent)
    assert np.allclose(output, expected.evaluate())


def test_optical_system_raised_latent_propagation():
    grid = dl.GridSpec(n=10, diam=1.0, unit="m")
    coeffs = _reparametrise(np.asarray([2e-8, -1e-8]))
    opd = dl.DynamicZernikeBasis(js=[4, 5], coeffs=coeffs, diameter=0.8)
    system = dl.OpticalSystem([("pupil", dl.Optic(opd=opd))], grid)
    delta = np.asarray([1e-8, -2e-8])

    baseline = assert_jittable(
        lambda model: model.propagate_mono(1e-6, return_wf=True), system
    )
    updated = system.add("latent", delta)
    qualified = system.set("pupil.opd.coeffs.latent", coeffs.latent + delta)
    output = updated.propagate_mono(1e-6, return_wf=True)

    assert np.array_equal(system.get("latent"), coeffs.latent)
    assert np.array_equal(system.get("pupil.opd.coeffs.latent"), coeffs.latent)
    assert np.array_equal(updated.get("latent"), coeffs.latent + delta)
    assert np.array_equal(qualified.get("latent"), coeffs.latent + delta)
    qualified_output = qualified.propagate_mono(1e-6, return_wf=True)
    assert np.allclose(output.phasor, qualified_output.phasor)
    assert not np.allclose(output.phasor, baseline.phasor)


def test_sparse_optic_resolves_local_reparametrised_coeffs():
    grid = dl.GridSpec(n=12, d=0.04, unit="m")
    wavefront = dl.Wavefront(1e-6, grid)
    centers = np.asarray([[-0.15, 0.0], [0.15, 0.0]])
    values = np.asarray([1e-8, -2e-8])
    array_opd = dl.DynamicZernikeBasis(js=[4], coeffs=values, diameter=0.2)
    latent_opd = dl.DynamicZernikeBasis(
        js=[4], coeffs=_reparametrise(values), diameter=0.2
    )
    transmission = dl.Circle(0.18, edge=0.01)
    array_optic = dl.SparseOptic(centers, transmission, array_opd)
    latent_optic = dl.SparseOptic(centers, transmission, latent_opd)

    expected = array_optic(wavefront)
    output = assert_jittable(latent_optic, wavefront, rtol=1e-5, atol=1e-5)
    gradient = assert_differentiable(
        lambda latent: latent_optic.set("latent", latent)(wavefront),
        latent_opd.coeffs.latent,
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.allclose(output.phasor, expected.phasor, rtol=1e-5, atol=1e-5)
    assert np.any(np.abs(gradient) > 0)


def test_sparse_optic_defers_context_dependent_coefficients():
    grid = dl.GridSpec(n=8, d=0.04, unit="m")
    wavefront = dl.Wavefront(1e-6, grid)
    centers = np.asarray([[-0.1, 0.0], [0.1, 0.0]])
    opd = dl.DynamicZernikeBasis(js=[4], coeffs=_CoordinateCoeffs(), diameter=0.2)
    optic = dl.SparseOptic(centers, dl.Circle(0.18, edge=0.01), opd)

    output = assert_jittable(optic, wavefront)

    assert output.phasor.shape[0] == len(centers)
