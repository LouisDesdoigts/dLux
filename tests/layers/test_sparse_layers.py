"""Tests for dLux.layers.sparse_layers."""

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx
from jax import Array

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class LocalCoefficients(dl.Parametric):
    """Shared or aperture-vectorised coefficients requiring local context."""

    scale: Array
    weights: Array

    def evaluate(self, *, coordinates, state, **context):
        return self.weights * (self.scale * coordinates[0, 0, 0] + state.time)


@pytest.fixture
def centers():
    return np.array([[-0.2, 0.0], [0.2, 0.0]])


@pytest.fixture
def wavefront(make_wavefront, make_grid):
    return make_wavefront(grid=make_grid(n=16, d=0.05))


@pytest.mark.parametrize("optic_type", [dl.SparseOptic, dl.SparseDynamicOptic])
@pytest.mark.parametrize("method", ["__call__", "localise", "phasor"])
@pytest.mark.parametrize("jones", [np.eye(2), zdx.StateRef("matrix")])
def test_sparse_rejects_local_jones(optic_type, method, jones, centers, wavefront):
    optic = optic_type(centers).set(jones=jones)
    apply = getattr(optic, method)
    for evaluate in (apply, eqx.filter_jit(apply)):
        with pytest.raises(ValueError, match="do not support local Jones terms"):
            evaluate(wavefront)


@pytest.mark.parametrize("dynamic", [False, True])
def test_sparse_optic_contract(dynamic, centers, wavefront):
    optic_type = dl.SparseDynamicOptic if dynamic else dl.SparseOptic
    optic = optic_type(centers, transmission=dl.Circle(0.2, edge=0.01))

    output = assert_jittable(optic, wavefront)
    local = assert_jittable(optic.localise, wavefront)
    assert optic.n_apertures == len(centers)
    assert np.allclose(output.phasor, local.phasor)
    assert output.phasor.shape == local.phasor.shape
    assert local.phasor.shape[0] == optic.n_apertures


def test_interfere(centers, wavefront):
    optic = dl.SparseOptic(centers, transmission=dl.Circle(0.2, edge=0.01))
    local = optic.localise(wavefront)
    output = assert_jittable(dl.Interfere(), local)

    assert np.allclose(output.phasor, local.phasor.sum(0))
    assert output.phasor.shape == wavefront.phasor.shape
    assert output.grid.c.shape == wavefront.grid.c.shape


def test_shared_and_local_coeffs(centers, wavefront):
    shared = dl.DynamicZernikeBasis(js=[4], coeffs=[1e-7], diameter=0.2)
    local = dl.DynamicZernikeBasis(
        js=[4], coeffs=np.array([[1e-7], [2e-7]]), diameter=0.2
    )
    common = {"centers": centers, "transmission": dl.Circle(0.2, edge=0.01)}
    shared_optic = dl.SparseDynamicOptic(opd=shared, **common)
    local_optic = dl.SparseDynamicOptic(opd=local, **common)

    assert_jittable(shared_optic, wavefront)
    assert_jittable(local_optic, wavefront)
    assert not np.allclose(
        shared_optic(wavefront).phasor, local_optic(wavefront).phasor
    )
    assert_differentiable(
        lambda coeffs: np.real(local_optic.set("opd.coeffs", coeffs)(wavefront).phasor),
        local.coeffs,
    )


def test_shared_and_local_distortions(centers, wavefront):
    shared = dl.Distortion(order=2, shift_invariant=True)
    local = shared.set(
        distortion=np.stack([shared.distortion, shared.distortion.at[0, 0].set(0.01)])
    )
    common = {"centers": centers, "transmission": dl.Circle(0.2, edge=0.01)}

    assert_jittable(dl.SparseDynamicOptic(transformation=shared, **common), wavefront)
    assert_jittable(dl.SparseDynamicOptic(transformation=local, **common), wavefront)


def test_local_affine_and_mismatched_transform(centers, wavefront):
    common = {"centers": centers, "transmission": dl.Circle(0.2, edge=0.01)}
    affine = dl.Affine(translation=np.array([[0.0, 0.0], [0.01, -0.01]]))
    assert_jittable(dl.SparseDynamicOptic(transformation=affine, **common), wavefront)

    distortion = dl.Distortion(order=2)
    distortion = distortion.set(
        distortion=np.stack([distortion.distortion] * (len(centers) + 1))
    )
    with pytest.raises(ValueError, match="distortion"):
        dl.SparseDynamicOptic(transformation=distortion, **common)(wavefront)


def test_local_affine_map(centers, wavefront):
    matrix = np.broadcast_to(np.eye(2), (len(centers), 2, 2))
    offset = np.asarray(((0.0, 0.0), (0.01, -0.01)))
    optic = dl.SparseDynamicOptic(
        centers,
        transmission=dl.Circle(0.2, edge=0.01),
        transformation=dl.AffineMap(matrix, offset),
    )

    output = assert_jittable(optic, wavefront)
    assert output.phasor.shape[0] == len(centers)


def test_center_gradients(centers, wavefront):
    optic = dl.SparseDynamicOptic(centers, transmission=dl.Circle(0.2, edge=0.01))

    assert_differentiable(lambda value: optic.set(centers=value)(wavefront), centers)


@pytest.mark.parametrize("basis_type", ["explicit", "implicit"])
@pytest.mark.parametrize("frame", ["plain", "global", "local"])
@pytest.mark.parametrize("per_aperture", [False, True])
@pytest.mark.parametrize("operation", ["phasor", "localise"])
def test_sparse_coefficients_use_local_context(
    basis_type, frame, per_aperture, operation, centers, wavefront
):
    weights = np.asarray([0.4, -0.2])
    if per_aperture:
        weights = np.stack((weights, -2 * weights))
    coeffs = LocalCoefficients(scale=np.asarray(0.7), weights=weights)
    if basis_type == "explicit":
        modes = np.stack((np.ones(wavefront.shape), wavefront.coordinates[0]))
        phase = dl.Basis(modes, coeffs=coeffs, axes=1)
    else:
        phase = dl.DynamicZernikeBasis(js=[1, 4], coeffs=coeffs, diameter=1.0)

    matrices = np.asarray([[1.2, 0.1], [0.0, 0.8]])
    offsets = np.asarray([0.07, -0.03])
    if frame == "local":
        matrices = np.stack((matrices, 0.7 * matrices))
        offsets = np.stack((offsets, -2 * offsets))
    if frame == "plain":
        optic = dl.SparseOptic(centers, phase=phase)
    else:
        optic = dl.SparseDynamicOptic(
            centers, phase=phase, transformation=dl.AffineMap(matrices, offsets)
        )
    state = zdx.State(time=0.13)

    # Caller coordinates must not replace the authoritative aperture frame.
    context = {
        "state": state,
        "coordinates": np.full_like(wavefront.coordinates, 99.0),
        "coords": np.full_like(wavefront.coordinates, -99.0),
    }

    def evaluate(scale):
        model = optic.set("phase.coeffs.scale", scale)
        result = getattr(model, operation)(wavefront, **context)
        return result.phasor if operation == "localise" else result

    def reference(scale):
        phasors = []
        for index, center in enumerate(centers):
            incident = wavefront
            if operation == "localise":
                incident = wavefront.set(c=center / wavefront.grid.scale)
            coordinates = incident.coordinates
            if frame == "global":
                coordinates = np.einsum("ij,jxy->ixy", matrices, coordinates)
                coordinates = coordinates + offsets[:, None, None]
            coordinates = coordinates - center[:, None, None]
            if frame == "local":
                coordinates = np.einsum("ij,jxy->ixy", matrices[index], coordinates)
                coordinates = coordinates + offsets[index, :, None, None]

            selected = weights[index] if per_aperture else weights
            values = selected * (scale * coordinates[0, 0, 0] + state.time)
            resolved = phase.set(coeffs=values).resolve(coordinates=coordinates)
            phasor = np.exp(1j * resolved)
            if operation == "localise":
                phasor = incident.phasor * phasor
            phasors.append(phasor)
        return np.stack(phasors) if operation == "localise" else sum(phasors)

    output = assert_jittable(evaluate, coeffs.scale, rtol=1e-5, atol=1e-6)
    assert np.allclose(output, reference(coeffs.scale), rtol=1e-5, atol=1e-6)
    assert optic.phase.coeffs is coeffs

    if frame == "local" and per_aperture and basis_type == "implicit":
        gradient = assert_differentiable(evaluate, coeffs.scale)
        expected = assert_differentiable(reference, coeffs.scale)
        assert np.allclose(gradient, expected, rtol=1e-5, atol=1e-6)
        assert np.abs(gradient) > 0


def test_sparse_phasor_and_coordinate_sources(centers, wavefront):
    transmission = dl.Circle(0.2, edge=0.01)
    optic = dl.SparseOptic(centers, transmission=transmission)
    coordinates = wavefront.coordinates
    grid = wavefront.grid
    array_optic = dl.SparseDynamicOptic(
        centers, transmission=transmission, coordinates=coordinates
    )
    spec_optic = dl.SparseDynamicOptic(
        centers, transmission=transmission, coordinates=grid
    )

    phasor = assert_jittable(optic.phasor, wavefront)
    array_output = assert_jittable(array_optic, wavefront)
    spec_output = assert_jittable(spec_optic, wavefront)

    expected = sum(
        transmission.evaluate(
            coordinates=wavefront.coordinates - center[:, None, None],
            pixel_scale=wavefront.pixel_scale,
        )
        for center in centers
    )
    assert np.allclose(phasor, expected)
    assert np.allclose(array_output.phasor, spec_output.phasor)


@pytest.mark.parametrize("polarised", [False, True])
@pytest.mark.parametrize(
    "layer",
    [
        dl.Fraunhofer(dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")),
        dl.Fraunhofer(dl.ResizeSpec(pad=(2, 3)), method="fft"),
        dl.FreeSpace(0.1, dl.ResizeSpec(pad=(2, 3)), crop=False),
    ],
)
def test_sparse_propagation_dimensions(layer, polarised, make_grid, make_wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    grid = make_grid(n=(8, 6), d=(0.05, 0.06))
    wavefront = make_wavefront(
        wavelength=np.asarray([1e-6, 1.1e-6]), grid=grid, polarised=polarised
    )
    optic = dl.SparseOptic(centers, transmission=dl.Circle(0.2, edge=0.01))

    local = optic.localise(wavefront)
    propagated = assert_jittable(layer, local, rtol=1e-5, atol=1e-5)
    combined = assert_jittable(dl.Interfere(), propagated)

    assert local.phasor.shape[:2] == (2, 3)
    assert propagated.phasor.shape[:2] == (2, 3)
    assert np.allclose(combined.phasor, propagated.phasor.sum(1))
    assert combined.phasor.shape[0] == 2
    assert combined.grid.d.ndim <= 2
    assert combined.grid.c is None or combined.grid.c.ndim <= 2


def test_sparse_propagation_gradient(wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    optic = dl.SparseDynamicOptic(centers, transmission=dl.Circle(0.2, edge=0.01))
    layer = dl.Fraunhofer(dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad"))

    def propagate(value):
        local = optic.set(centers=value).localise(wavefront)
        return dl.Interfere()(layer(local))

    assert_differentiable(propagate, centers, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("polarise_input", [False, True])
def test_sparse_optical_system_contract(polarise_input, centers, make_grid):
    spec_in = make_grid(n=(8, 6), d=(0.05, 0.06))
    spec_out = dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")
    polarisation = dl.PolarisationLayer(dl.LinearPolariser(0.2))
    layers = []
    if polarise_input:
        layers.append(("InputPolarisation", polarisation))
    layers += [
        ("SparseOptic", dl.SparseOptic(centers, transmission=dl.Circle(0.2))),
        ("OutputPolarisation", polarisation),
        ("Fraunhofer", dl.Fraunhofer(spec_out)),
        ("Interfere", dl.Interfere()),
    ]
    system = dl.OpticalSystem(layers, spec_in)
    wavelengths = np.asarray([1e-6, 1.1e-6])

    output = system(system.initialise_wavefront(wavelengths))
    _, states = system.debug(system.initialise_wavefront(wavelengths))

    assert output.phasor.shape == (2, 2, 2, 8, 6)
    assert states["SparseOptic"].batch_ndim == 2
    assert states["SparseOptic"].phasor.shape[:2] == (2, len(centers))
    assert states["OutputPolarisation"].batch_ndim == 2
    assert np.allclose(states["Interfere"].phasor, states["Fraunhofer"].phasor.sum(1))


def test_center_validation():
    with pytest.raises(ValueError, match="centers"):
        dl.SparseOptic([0.0, 1.0])


@pytest.mark.parametrize("apertures", [2, 3])
@pytest.mark.parametrize("polarised", [False, True])
def test_interference_collapses_shared_aperture_metadata_axis(apertures, polarised):
    wavelengths = np.asarray([5e-4, 6e-4])
    spacing = np.asarray([[0.02, 0.03], [0.025, 0.035]])[:, None, :]
    centers = np.asarray([[0.001, -0.002], [0.003, -0.001]])[:, None, :]
    grid = dl.GridSpec((5, 6), spacing, centers, "m")
    intrinsic = (2, 2) if polarised else ()
    field = np.ones((2, apertures) + intrinsic + (6, 5), dtype=complex)
    wavefront_type = dl.PolarisedWavefront if polarised else dl.Wavefront
    wavefront = wavefront_type(wavelengths, grid, field)
    output = assert_jittable(dl.Interfere(), wavefront)
    assert np.allclose(output.phasor, field.sum(axis=1))
    assert output.grid.d.shape == (2, 2)
    assert output.grid.c.shape == (2, 2)
    assert np.allclose(output.grid.d, spacing[:, 0])
    assert np.allclose(output.grid.c, centers[:, 0])
