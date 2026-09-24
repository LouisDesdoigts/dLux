"""Definitions are consumed at construction; numerical models own evaluation."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu


@pytest.mark.parametrize("dynamic", [False, True])
def test_optic_constructs_unitful_basis_in_its_local_frame(dynamic):
    grid = dlu.Grid(n=8, diam=1.0, unit="m")
    transform = dlu.Affine(translation=[0.08, -0.03], rotation=0.2)
    aperture = dlu.Circle(0.8, edge=1.0)
    definition = dl.ZernikeDef(
        nolls=[1, 4], coeffs=[10.0, -3.0], diameter=0.8, unit="nm", norm="rms"
    )
    optic = dl.Optic(
        transmission=aperture,
        opd=definition,
        grid=grid,
        transform=transform,
        dynamic=dynamic,
    )
    assert type(optic.opd) is dlu.Basis
    assert isinstance(optic.opd.basis, dlu.Zernike) == dynamic
    assert not isinstance(optic.opd, dl.Def)
    assert np.array_equal(optic.opd.coeffs, np.array([10.0, -3.0]))

    coordinates = grid.transformed(transform)
    transmission = aperture.resolve(coordinates=coordinates, pixel_scale=grid.d_si)
    basis = definition(coordinates=coordinates, support=transmission != 0)
    wavefront = dl.Wavefront(600e-9, grid)
    expected = dl.Optic(transmission=transmission, opd=basis)(wavefront)
    result = eqx.filter_jit(lambda model: model(wavefront))(optic)
    assert np.allclose(result.phasor, expected.phasor, rtol=2e-6, atol=2e-7)

    def real_field(coeffs):
        return optic.set("opd.coeffs", coeffs)(wavefront).phasor.real[3, 4]

    gradient = jax.jit(jax.grad(real_field))(optic.opd.coeffs)
    wavenumber = 2 * np.pi / wavefront.wavelength
    amplitude = wavefront.amplitude[3, 4] * transmission[3, 4]
    phase = wavenumber * basis.resolve()[3, 4]
    reference = -amplitude * np.sin(phase) * wavenumber * basis.basis[:, 3, 4] * 1e-9
    assert np.allclose(gradient, reference, rtol=2e-5, atol=1e-9)


def test_missing_optic_grid_builds_dynamic_with_fit_ready_coefficients():
    optic = dl.Optic(opd=dl.ZernikeDef(order=2, unit="nm"))
    assert isinstance(optic.opd, dlu.Basis)
    assert np.array_equal(optic.opd.coeffs, np.zeros(6))
    assert isinstance(optic.resolve().opd, dlu.Basis)


def test_definition_sampling_uses_optic_transform_and_aperture_support():
    grid = dlu.Grid(n=8, diam=1.0, unit="m")
    definition = dl.ZernikeDef(
        nolls=[2, 4],
        coeffs=[10.0, 4.0],
        diameter=1.0,
        grid=grid,
        norm="rms",
        unit="nm",
    )
    aperture = dlu.Circle(0.6)
    transform = dlu.Affine(translation=[0.05, -0.1])
    cached = dl.Optic(transmission=aperture, opd=definition, transform=transform)
    dynamic = dl.Optic(
        transmission=aperture, opd=definition, transform=transform, dynamic=True
    )
    wavefront = dl.Wavefront(600e-9, grid)
    assert np.allclose(cached(wavefront).phasor, dynamic(wavefront).phasor)


def test_dynamic_option_retains_already_constructed_models():
    basis = dlu.Basis(np.ones((2, 6, 6)), coeffs=[1.0, 2.0], unit="nm")
    optic = dl.Optic(opd=basis, dynamic=True)
    assert optic.opd is basis
    assert np.allclose(optic.opd.resolve(), 3e-9, rtol=1e-6, atol=0)


def test_definition_resolve_retains_factory():
    definition = dl.ZernikeDef(nolls=[1, 4], coeffs=zdx.Map(x=[2.0, 3.0], s=2.0))
    resolved = definition.resolve()
    assert isinstance(resolved, dl.Def)
    assert not isinstance(resolved, zdx.Expression)
    assert np.array_equal(resolved.coeffs, np.array([4.0, 6.0]))


def test_sparse_definitions_use_runtime_local_coordinates():
    grid = dlu.Grid(n=8, diam=0.4, unit="m")
    centers = np.array([[-0.3, 0.0], [0.3, 0.0]])
    definition = dl.ZernikeDef(
        nolls=[1, 4], coeffs=[[1.0, 2.0], [3.0, 4.0]], diameter=0.4, unit="nm"
    )
    optic = dl.SparseOptic(centers, transmission=dlu.Circle(0.4), opd=definition)
    assert isinstance(optic.opd, dlu.Basis)
    expected = dl.SparseOptic(centers, transmission=dlu.Circle(0.4), opd=definition())
    wavefront = dl.Wavefront(600e-9, grid)
    result = eqx.filter_jit(lambda model: model(wavefront))(optic)
    assert np.allclose(result.phasor, expected(wavefront).phasor)
    with pytest.raises(ValueError, match="construct a local Basis"):
        dl.SparseOptic(centers, opd=definition, grid=grid)
    dynamic = dl.SparseOptic(centers, opd=definition, grid=grid, dynamic=True)
    assert isinstance(dynamic.opd, dlu.Basis)
