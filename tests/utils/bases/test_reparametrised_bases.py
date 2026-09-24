"""Integration tests for reparametrised coefficient leaves."""

import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


class _CoordinateCoeffs(zdx.Expression):
    """Coefficient value requiring local coordinate context."""

    @property
    def shape(self):
        """Return the single physical coefficient shape."""
        return (1,)

    def evaluate(self, *, coordinates, **context):
        """Return one coefficient derived from local coordinates."""
        return np.asarray([1e-8 * (coordinates[0, 0, 0] + 1)])


def _reparametrise(coeffs):
    """Represent native coefficient axes through a public Zodiax transform."""
    return zdx.Map(np.asarray(coeffs), s=1.0, alias={"latent": "x"})


def _explicit_case():
    vectors = np.arange(24.0).reshape(2, 3, 4) / 24
    coeffs = np.asarray([0.2, -0.1])
    return (
        dl.Basis(vectors, coeffs),
        dl.Basis(vectors, _reparametrise(coeffs), axes=1),
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
        stamp_shape=(2, 2),
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


def _spectral_polynomial_case():
    coeffs = np.asarray([0.2, -0.1])
    context = {"coordinates": np.linspace(500e-9, 700e-9, 5)}
    mapping = zdx.Map(s=1e7, b=-6.0)
    return (
        dlu.Polynomial(degrees=[1, 2], coeffs=coeffs, transform=mapping),
        dlu.Polynomial(
            degrees=[1, 2], coeffs=_reparametrise(coeffs), transform=mapping
        ),
        context,
    )


def _spectral_basis_case():
    vectors = np.stack((np.ones(5), np.linspace(-0.5, 0.5, 5)))
    coeffs = np.asarray([1.0, 0.2])
    return (
        dlu.Basis(vectors, coeffs),
        dlu.Basis(vectors, _reparametrise(coeffs), axes=1),
        {},
    )


def _polynomial_case():
    coeffs = np.asarray([0.7, -0.2, 0.1])
    context = {"coordinates": np.linspace(-1.0, 1.0, 7)}
    return (
        dl.Polynomial(degree=2, coeffs=coeffs),
        dl.Polynomial(degree=2, coeffs=_reparametrise(coeffs)),
        context,
    )


def _dynamic_zernike_case():
    coeffs = np.asarray([0.2, -0.1])
    grid = dl.Grid(n=8, diam=1.0, unit="m")
    context = {"coordinates": grid.coordinates}
    return (
        dl.Basis(dl.utils.Zernike(nolls=[1, 4], diameter=0.8), coeffs=coeffs),
        dl.Basis(
            dl.utils.Zernike(nolls=[1, 4], diameter=0.8), coeffs=_reparametrise(coeffs)
        ),
        context,
    )


_CASES = {
    "basis": _explicit_case,
    "fourier": _fourier_case,
    "pasted": _pasted_case,
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


def test_stored_coefficients_retain_parametrisation_identity():
    coeffs = _reparametrise([0.2, -0.1])
    basis = dl.Basis(np.arange(24.0).reshape(2, 3, 4), coeffs, axes=1)

    assert basis.coeffs is coeffs


def test_nested_latent_paths_support_get_set_and_add():
    coeffs = _reparametrise([0.2, -0.1])
    basis = dl.Basis(np.arange(24.0).reshape(2, 3, 4), coeffs, axes=1)
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
    basis = dl.Basis(vectors, coeffs, axes=1)
    replacement = np.asarray([-0.3, 0.4])

    replaced = basis.set("coeffs", replacement)
    expected = dl.Basis(vectors, replacement)

    assert isinstance(basis.coeffs, zdx.Transform)
    assert basis.coeffs is coeffs
    assert np.array_equal(basis.coeffs.latent, np.asarray([0.2, -0.1]))
    assert np.array_equal(replaced.coeffs, replacement)
    assert np.allclose(replaced.evaluate(), expected.evaluate())


def test_single_mode_reparametrisation_retains_independent_batches():
    vectors = np.arange(12.0).reshape(1, 3, 2, 2)
    coeffs = _reparametrise(np.asarray([[1.0], [2.0], [3.0]]))
    basis = dl.Basis(vectors, coeffs, axes=1)
    output = assert_jittable(lambda model: model.resolve(), basis)
    expected = coeffs.resolve()[:, 0, None, None, None] * vectors[0]
    assert output.shape == (3, 3, 2, 2)
    assert np.allclose(output, expected)


def test_optical_system_raised_latent_propagation():
    grid = dl.Grid(n=10, diam=1.0, unit="m")
    coeffs = _reparametrise(np.asarray([2e-8, -1e-8]))
    opd = dl.Basis(dl.utils.Zernike(nolls=[4, 5], diameter=0.8), coeffs=coeffs)
    system = dl.OpticalSystem([("pupil", dl.Optic(opd=opd))])
    wavefront = dl.Wavefront(wavelength=1e-6, grid=grid)
    delta = np.asarray([1e-8, -2e-8])

    baseline = assert_jittable(lambda model: model(wavefront), system)
    updated = system.add("latent", delta)
    qualified = system.set("pupil.opd.coeffs.latent", coeffs.latent + delta)
    output = updated(wavefront)

    assert np.array_equal(system.get("latent"), coeffs.latent)
    assert np.array_equal(system.get("pupil.opd.coeffs.latent"), coeffs.latent)
    assert np.array_equal(updated.get("latent"), coeffs.latent + delta)
    assert np.array_equal(qualified.get("latent"), coeffs.latent + delta)
    qualified_output = qualified(wavefront)
    assert np.allclose(output.phasor, qualified_output.phasor)
    assert not np.allclose(output.phasor, baseline.phasor)


def test_sparse_optic_resolves_local_reparametrised_coeffs():
    grid = dl.Grid(n=12, d=0.04, unit="m")
    wavefront = dl.Wavefront(1e-6, grid)
    centers = np.asarray([[-0.15, 0.0], [0.15, 0.0]])
    values = np.asarray([[1e-8], [-2e-8]])
    array_opd = dl.Basis(dl.utils.Zernike(nolls=[4], diameter=0.2), coeffs=values)
    latent_opd = dl.Basis(
        dl.utils.Zernike(nolls=[4], diameter=0.2), coeffs=_reparametrise(values)
    )
    transmission = dl.Circle(0.18, edge=0.01)
    array_optic = dl.SparseOptic(centers, transmission, array_opd)
    latent_optic = dl.SparseOptic(centers, transmission, latent_opd)

    expected = array_optic(wavefront)
    output = assert_jittable(latent_optic, wavefront, rtol=1e-5, atol=1e-5)
    # Total field power is invariant to this phase parameter. Its real component
    # has meaningful coefficient sensitivity, without cancellation of phase gradients.
    gradient = assert_differentiable(
        lambda latent: latent_optic.set("latent", latent)(wavefront).phasor.real,
        latent_opd.coeffs.latent,
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.allclose(output.phasor, expected.phasor, rtol=1e-5, atol=1e-5)
    assert np.any(np.abs(gradient) > 0)


def test_sparse_optic_defers_context_dependent_coefficients():
    grid = dl.Grid(n=8, d=0.04, unit="m")
    wavefront = dl.Wavefront(1e-6, grid)
    centers = np.asarray([[-0.1, 0.0], [0.1, 0.0]])
    opd = dl.Basis(
        dl.utils.Zernike(nolls=[4], diameter=0.2), coeffs=_CoordinateCoeffs()
    )
    optic = dl.SparseOptic(centers, dl.Circle(0.18, edge=0.01), opd)

    output = assert_jittable(optic, wavefront)

    assert output.phasor.shape[0] == len(centers)


def test_basis_solve_returns_physical_coefficients_without_reversing_transform():
    latent = np.array([0.2, -0.1])
    coefficients = zdx.Map(latent, s=np.array([2.0, 4.0]), b=np.array([1.0, 2.0]))
    basis = dl.Basis(np.eye(2), coeffs=coefficients, axes=1)

    values = assert_jittable(lambda model: model.evaluate(), basis)
    recovered = assert_jittable(lambda model: model.solve(values), basis)
    assert np.allclose(recovered, np.array([1.4, 1.6]))
    assert np.allclose(coefficients(recovered, inverse=True), latent)
    assert basis.coeffs is coefficients
    assert not hasattr(type(basis), "apply")
