"""Tests for dLux.layers.sparse_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def centers():
    return np.array([[-0.2, 0.0], [0.2, 0.0]])


@pytest.fixture
def wavefront(make_wavefront, make_spec):
    return make_wavefront(spec=make_spec(n=16, d=0.05))


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
    assert output.spec.c.shape == wavefront.spec.c.shape


def test_shared_and_local_coeffs(centers, wavefront):
    shared = dl.DynamicZernikeBasis(js=[4], coeffs=[1e-7], diameter=0.2)
    local = dl.DynamicZernikeBasis(
        js=[4], coeffs=np.array([1e-7, 2e-7]), diameter=0.2
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
        lambda coeffs: np.real(
            local_optic.set("opd.coeffs", coeffs)(wavefront).phasor
        ),
        local.coeffs,
    )


def test_shared_and_local_distortions(centers, wavefront):
    shared = dl.DistortCoords(order=2, shift_invariant=True)
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

    distortion = dl.DistortCoords(order=2)
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


def test_sparse_phasor_and_coordinate_sources(centers, wavefront):
    transmission = dl.Circle(0.2, edge=0.01)
    optic = dl.SparseOptic(centers, transmission=transmission)
    coordinates = wavefront.coordinates
    spec = wavefront.spec
    array_optic = dl.SparseDynamicOptic(
        centers, transmission=transmission, coordinates=coordinates
    )
    spec_optic = dl.SparseDynamicOptic(
        centers, transmission=transmission, coordinates=spec
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
def test_sparse_propagation_dimensions(layer, polarised, make_spec, make_wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    spec = make_spec(n=(8, 6), d=(0.05, 0.06))
    wavefront = make_wavefront(
        wavelength=np.asarray([1e-6, 1.1e-6]), spec=spec, polarised=polarised
    )
    optic = dl.SparseOptic(centers, transmission=dl.Circle(0.2, edge=0.01))

    local = optic.localise(wavefront)
    propagated = assert_jittable(layer, local, rtol=1e-5, atol=1e-5)
    combined = assert_jittable(dl.Interfere(), propagated)

    assert local.phasor.shape[:2] == (2, 3)
    assert propagated.phasor.shape[:2] == (2, 3)
    assert np.allclose(combined.phasor, propagated.phasor.sum(1))
    assert combined.phasor.shape[0] == 2
    assert combined.spec.d.ndim <= 2
    assert combined.spec.c is None or combined.spec.c.ndim <= 2


def test_sparse_propagation_gradient(wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    optic = dl.SparseDynamicOptic(centers, transmission=dl.Circle(0.2, edge=0.01))
    layer = dl.Fraunhofer(dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad"))

    def propagate(value):
        local = optic.set(centers=value).localise(wavefront)
        return dl.Interfere()(layer(local))

    assert_differentiable(propagate, centers, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("polarise_input", [False, True])
def test_sparse_optical_system_contract(polarise_input, centers, make_spec):
    spec_in = make_spec(n=(8, 6), d=(0.05, 0.06))
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

    output = system.propagate(wavelengths, return_wf=True)
    _, states = system.debug(system.initialise_wavefront(wavelengths))

    assert output.phasor.shape == (2, 2, 2, 8, 6)
    assert states["SparseOptic"].batch_ndim == 2
    assert states["SparseOptic"].phasor.shape[:2] == (2, len(centers))
    assert states["OutputPolarisation"].batch_ndim == 2
    assert np.allclose(states["Interfere"].phasor, states["Fraunhofer"].phasor.sum(1))


def test_center_validation():
    with pytest.raises(ValueError, match="centers"):
        dl.SparseOptic([0.0, 1.0])
