"""Tests for dLux.layers.sparse_layers."""

from dataclasses import fields

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx
from jax import Array

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable, assert_tree_allclose


class LocalCoefficients(zdx.Expression):
    """Shared or aperture-vectorised coefficients requiring local context."""

    scale: Array
    weights: Array

    def evaluate(self, *, coordinates, state, **context):
        return self.weights * (self.scale * coordinates[0, 0, 0] + state.time)


class RampBasis(dlu.ParametricBasis):
    """User-defined piston and x-ramp, independent of built-in basis families."""

    @property
    def coeffs_ndim(self):
        return 1

    def evaluate(self, *, coordinates, **context):
        self = self.resolve_children(coordinates=coordinates, **context)
        return self.coeffs[0] + self.coeffs[1] * coordinates[0]

    def solve(self, samples, *, coordinates, **context):
        modes = np.stack((np.ones_like(coordinates[0]), coordinates[0]))
        return dlu.solve_basis(samples, modes, axes=1)


@pytest.fixture
def centers():
    return np.array([[-0.2, 0.0], [0.2, 0.0]])


@pytest.fixture
def wavefront(make_wavefront, make_grid):
    return make_wavefront(grid=make_grid(n=16, d=0.05))


@pytest.mark.parametrize("chromatic", [False, True])
def test_sparse_composes_local_jones_after_pupil_normalisation(chromatic):
    centers = np.array([[-0.2, 0.0], [0.2, 0.0]])
    grid = dlu.Grid(n=6, d=0.05, unit="m")
    wavelength = np.array([0.8e-6, 1.1e-6]) if chromatic else 0.8e-6
    wavefront = dl.Wavefront(wavelength, grid)
    angles = np.array([0.0, 0.3])
    optic = dl.SparseOptic(
        centers,
        transmission=np.stack([np.ones(grid.shape), 0.5 * np.ones(grid.shape)]),
        normalise=True,
        jones=dlu.Jones.LinearPolariser(angles[:, None, None]),
    )
    output = assert_jittable(optic, wavefront)
    assert output.jones is True
    assert output.phasor.shape[0] == 2
    # A linear polariser removes half the unpolarised input power. The two
    # apertures keep their relative 4:1 intensity after joint normalisation.
    assert np.allclose(output.power.sum(0), 0.5, atol=2e-6)
    assert np.allclose(output.power[0] / output.power[1], 4.0)
    expected = []
    for angle, amplitude in zip(angles, [1.0, 0.5]):
        field = wavefront * (amplitude / np.sqrt(1.25))
        expected.append(
            field.apply_jones(dlu.Jones.linear_polariser(angle)).materialise().phasor
        )
    expected = np.stack(expected, axis=0)
    assert np.allclose(output.phasor, expected, atol=2e-6)


def test_sparse_jones_scalar_phasor_guidance(centers, wavefront):
    optic = dl.SparseOptic(centers, jones=dlu.Jones.HalfWavePlate())
    with pytest.raises(ValueError, match="use localise or apply"):
        optic.phasor(wavefront)


def test_sparse_inherits_flat_scalar_and_refractive_effects(centers, wavefront):
    thickness = np.stack(
        [np.full(wavefront.spatial_shape, value) for value in [2e-7, 4e-7]]
    )
    optic = dl.SparseOptic(
        centers,
        transmission=0.6,
        opd=1e-8,
        amplitude=0.8,
        phase=0.2,
        thickness=thickness,
        n=1.5,
    )
    for base in (dl.BaseOptic, dl.ComplexLayer, dl.RefractiveLayer, dl.JonesLayer):
        assert isinstance(optic, base)
    leaf_names = {field.name for field in fields(optic)}
    assert {
        "centers",
        "transmission",
        "opd",
        "amplitude",
        "phase",
        "thickness",
        "n",
    } <= leaf_names
    assert leaf_names.isdisjoint({"scalar", "complex", "refractive", "polarisation"})
    actual = assert_jittable(optic, wavefront)
    expected = []
    for samples in thickness:
        layer = dl.Optic(
            transmission=0.6,
            opd=1e-8,
            amplitude=0.8,
            phase=0.2,
            thickness=samples,
            n=1.5,
        )
        expected.append(layer(wavefront).phasor)
    assert np.allclose(actual.phasor, np.stack(expected), atol=2e-6)


def test_sparse_optic_contract(centers, wavefront):
    optic_type = dl.SparseOptic
    optic = optic_type(centers, transmission=dlu.Circle(0.2, edge=0.01))

    output = assert_jittable(optic, wavefront)
    local = assert_jittable(optic.localise, wavefront)
    assert optic.n_apertures == len(centers)
    assert np.allclose(output.phasor, local.phasor)
    assert output.phasor.shape == local.phasor.shape
    assert local.phasor.shape[0] == optic.n_apertures


@pytest.mark.parametrize("representation", ["array", "expression", "basis"])
def test_sampled_sparse_axis_is_aperture_when_wavelength_count_matches(representation):
    centers = np.asarray([[-0.2, 0.0], [0.2, 0.0]])
    grid = dlu.Grid((5, 4), 0.1, unit="m")
    wavelengths = np.asarray([0.8e-6, 1.1e-6])
    wavefront = dl.Wavefront(wavelengths, grid)
    x, y = grid.coordinates
    transmission = np.stack((0.7 + 0.2 * x, 0.4 + 0.1 * y))
    opd = np.stack((2e-8 * x, -3e-8 * y))
    phase = np.stack((0.1 + y, -0.2 + x))

    def wrap(value):
        if representation == "expression":
            return zdx.Map(value)
        if representation == "basis":
            return dlu.Basis(value[None], coeffs=[1.0])
        return value

    optic = dl.SparseOptic(centers, wrap(transmission), wrap(opd), wrap(phase))
    independent = []
    for index in range(len(centers)):
        local = dl.Optic(transmission[index], opd[index], phase[index])(wavefront)
        independent.append(local.phasor)
    expected = np.stack(independent, axis=0)

    local = assert_jittable(optic.localise, wavefront)
    assert local.phasor.shape == (2, 2, 4, 5)
    assert np.allclose(local.phasor, expected, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        optic.phasor(wavefront) * wavefront.phasor, expected.sum(0), atol=1e-6
    )


@pytest.mark.parametrize("name", ["transmission", "opd", "phase"])
def test_sparse_sampled_aperture_count_must_match(name, centers, wavefront):
    values = np.ones((len(centers) + 1,) + wavefront.spatial_shape)
    optic = dl.SparseOptic(centers, **{name: values})
    with pytest.raises(ValueError, match=f"{name}.*aperture axis"):
        eqx.filter_jit(optic.localise)(wavefront)


def test_coherent_sum_requires_common_grid(centers, wavefront):
    local = dl.SparseOptic(centers, transmission=dlu.Circle(0.2)).localise(wavefront)
    with pytest.raises(ValueError, match="common spatial grid"):
        local.coherent_sum()


def test_shared_and_local_coeffs(centers, wavefront):
    shared = dlu.Basis(dlu.Zernike(nolls=[4], diameter=0.2), coeffs=[1e-7])
    local = dlu.Basis(
        dlu.Zernike(nolls=[4], diameter=0.2), coeffs=np.array([[1e-7], [2e-7]])
    )
    common = {"centers": centers, "transmission": dlu.Circle(0.2, edge=0.01)}
    shared_optic = dl.SparseOptic(opd=shared, **common)
    local_optic = dl.SparseOptic(opd=local, **common)

    assert_jittable(shared_optic, wavefront)
    assert_jittable(local_optic, wavefront)
    assert not np.allclose(
        shared_optic(wavefront).phasor, local_optic(wavefront).phasor
    )
    assert_differentiable(
        lambda coeffs: np.real(local_optic.set("opd.coeffs", coeffs)(wavefront).phasor),
        local.coeffs,
    )


@pytest.mark.parametrize("per_aperture", [False, True])
def test_sparse_accepts_custom_basis_contract(per_aperture, centers, wavefront):
    coefficients = np.asarray([0.2, -0.3])
    if per_aperture:
        coefficients = np.stack((coefficients, -2 * coefficients))
    phase = RampBasis(coeffs=zdx.Map(coefficients))
    optic = dl.SparseOptic(centers, phase=phase)
    expected = []
    for i, center in enumerate(centers):
        selected = coefficients[i] if per_aperture else coefficients
        x = wavefront.coordinates[0] - center[0]
        expected.append(np.exp(1j * (selected[0] + selected[1] * x)))

    actual = assert_jittable(optic.phasor, wavefront)
    assert np.allclose(actual, sum(expected), atol=1e-6)
    gradient = assert_differentiable(
        lambda c: optic.set("phase.coeffs.x", c).phasor(wavefront).real,
        coefficients,
    )
    assert np.any(np.abs(gradient) > 0)


def test_sparse_pasted_stamp_axis_is_intrinsic(centers):
    grid = dlu.Grid((6, 4), 1.0)
    wavefront = dl.Wavefront(1e-6, grid)
    spec = dlu.PasteSpec((6, 4), (2, 2), [[0, 0], [4, 2]], np.zeros((2, 2)), np.ones(2))
    phase = dlu.PastedBasis(np.ones((2, 1, 2, 2)), spec, [[0.2], [-0.3]])
    optic = dl.SparseOptic(centers, phase=phase)
    actual = assert_jittable(optic.phasor, wavefront)
    expected = len(centers) * np.exp(1j * phase.resolve())
    assert phase.coeffs_ndim == 2
    assert np.allclose(actual, expected, atol=1e-6)


def test_shared_and_local_distortions(centers, wavefront):
    shared = dlu.Distortion(orders=(2,))
    local = shared.set(
        distortion=np.stack([shared.distortion, shared.distortion.at[0, 0].set(0.01)])
    )
    common = {"centers": centers, "transmission": dlu.Circle(0.2, edge=0.01)}

    assert_jittable(dl.SparseOptic(transform=shared, **common), wavefront)
    assert_jittable(dl.SparseOptic(transform=local, **common), wavefront)


def test_local_affine_and_mismatched_transform(centers, wavefront):
    common = {"centers": centers, "transmission": dlu.Circle(0.2, edge=0.01)}
    affine = dlu.Affine(translation=np.array([[0.0, 0.0], [0.01, -0.01]]))
    assert_jittable(dl.SparseOptic(transform=affine, **common), wavefront)

    distortion = dlu.Distortion(order=2)
    distortion = distortion.set(
        distortion=np.stack([distortion.distortion] * (len(centers) + 1))
    )
    with pytest.raises(ValueError, match="distortion"):
        dl.SparseOptic(transform=distortion, **common)(wavefront)


def test_local_affine_map(centers, wavefront):
    matrix = np.broadcast_to(np.eye(2), (len(centers), 2, 2))
    offset = np.asarray(((0.0, 0.0), (0.01, -0.01)))
    optic = dl.SparseOptic(
        centers,
        transmission=dlu.Circle(0.2, edge=0.01),
        transform=dlu.AffineMap(matrix, offset),
    )

    output = assert_jittable(optic, wavefront)
    assert output.phasor.shape[0] == len(centers)


def test_center_gradients(centers, wavefront):
    optic = dl.SparseOptic(centers, transmission=dlu.Circle(0.2, edge=0.01))

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
        phase = dlu.Basis(modes, coeffs=coeffs, axes=1)
    else:
        phase = dlu.Basis(dlu.Zernike(nolls=[1, 4], diameter=1.0), coeffs=coeffs)

    matrices = np.asarray([[1.2, 0.1], [0.0, 0.8]])
    offsets = np.asarray([0.07, -0.03])
    if frame == "local":
        matrices = np.stack((matrices, 0.7 * matrices))
        offsets = np.stack((offsets, -2 * offsets))
    if frame == "plain":
        optic = dl.SparseOptic(centers, phase=phase)
    else:
        optic = dl.SparseOptic(
            centers, phase=phase, transform=dlu.AffineMap(matrices, offsets)
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
                incident = wavefront.set(
                    c=center / dlu.unit_factor(wavefront.grid.unit)
                )
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
    transmission = dlu.Circle(0.2, edge=0.01)
    optic = dl.SparseOptic(centers, transmission=transmission)
    explicit = optic.set(grid=wavefront.grid)
    expected = sum(
        transmission.resolve(
            coordinates=wavefront.coordinates - center[:, None, None],
            pixel_scale=wavefront.pixel_scale,
        )
        for center in centers
    )
    assert np.allclose(assert_jittable(optic.phasor, wavefront), expected)
    samples = np.stack(
        [
            transmission.resolve(
                coordinates=wavefront.coordinates - center[:, None, None],
                pixel_scale=wavefront.pixel_scale,
            )
            for center in centers
        ]
    )
    assert np.allclose(explicit(wavefront).phasor, samples * wavefront.phasor)


@pytest.mark.parametrize("polarised", [False, True])
@pytest.mark.parametrize(
    "layer",
    [
        dl.PropagatorLayer(dl.Grid(n=(6, 8), d=(2e-07, 3e-07), unit="rad")),
        dlu.FFTPropagator(dlu.ResizeSpec(pad=(2, 3))),
        dlu.FreeSpace(0.1, dlu.ResizeSpec(pad=(2, 3))),
    ],
)
def test_sparse_propagation_dimensions(layer, polarised, make_grid, make_wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    grid = make_grid(n=(8, 6), d=(0.05, 0.06))
    wavefront = make_wavefront(
        wavelength=np.asarray([1e-6, 1.1e-6]), grid=grid, polarised=polarised
    )
    optic = dl.SparseOptic(centers, transmission=dlu.Circle(0.2, edge=0.01))

    local = optic.localise(wavefront)
    propagated = assert_jittable(
        lambda field: field.propagate(layer), local, rtol=1e-5, atol=1e-5
    )
    if isinstance(layer, dlu.FreeSpace):
        # Free space retains separate patch centres; their fields cannot yet sum.
        assert propagated.phasor.shape[:2] == (3, 2)
        with pytest.raises(ValueError, match="common spatial grid"):
            propagated.coherent_sum()
        return
    combined = assert_jittable((lambda value: value.coherent_sum()), propagated)

    assert local.phasor.shape[:2] == (3, 2)
    assert propagated.phasor.shape[:2] == (3, 2)
    assert np.allclose(combined.phasor, propagated.phasor.sum(0))
    assert combined.phasor.shape[0] == 2
    assert combined.grid.d.ndim <= 2
    assert combined.grid.c is None or combined.grid.c.ndim <= 2


def test_sparse_propagation_gradient(wavefront):
    centers = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    optic = dl.SparseOptic(centers, transmission=dlu.Circle(0.2, edge=0.01))
    layer = dl.PropagatorLayer(dl.Grid(n=(6, 8), d=(2e-07, 3e-07), unit="rad"))

    def propagate(value):
        local = optic.set(centers=value).localise(wavefront)
        return (lambda value: value.coherent_sum())(layer(local))

    assert_differentiable(propagate, centers, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("polarise_input", [False, True])
def test_sparse_optical_system_contract(polarise_input, centers, make_grid):
    entrance = make_grid(n=(8, 6), d=(0.05, 0.06))
    output = dl.Grid(n=(6, 8), d=(2e-7, 3e-7), unit="rad")
    wavefront = dl.Wavefront(np.array([1e-6, 1.1e-6]), entrance)
    polarisation = dl.Optic(jones=dlu.Jones.LinearPolariser(0.2))
    if polarise_input:
        wavefront = polarisation(wavefront)
    system = dl.OpticalSystem(
        [
            ("apertures", dl.SparseOptic(centers, transmission=dlu.Circle(0.2))),
            ("polarisation", polarisation),
            ("propagation", dl.PropagatorLayer(output)),
        ]
    )
    propagated = assert_jittable(system, wavefront)
    combined = propagated.coherent_sum()
    assert combined.intensity.shape == (2, 8, 6)
    assert np.allclose(
        combined.materialise().phasor, propagated.materialise().phasor.sum(0)
    )


def test_sparse_optional_relay_follows_local_effects_and_retains_aperture_axes():
    entrance = dlu.Grid(6, d=0.02, unit="m")
    target = dlu.Grid(5, d=0.01, unit="m")
    wavefront = dl.Wavefront(np.array([5e-4, 6e-4]), entrance)
    optic = dl.SparseOptic(
        [[-0.1, 0.0], [0.1, 0.0]],
        transmission=dlu.Circle(0.1),
        normalise=True,
        jones=dlu.Jones.LinearPolariser(0.2),
        grid_out=target,
        focal_length=0.7,
    )
    local = optic.localise(wavefront, normalise=True)
    expected = local.propagate(dlu.Propagator(target, focal_length=0.7))
    actual = assert_jittable(optic, wavefront, rtol=2e-5, atol=2e-5)
    assert actual.phasor.shape == (2, 2, 2, 2, 5, 5)
    assert local.spatial_shape == entrance.shape
    assert np.allclose(local.power.sum(0), 0.5, atol=2e-6)
    assert_tree_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_center_validation():
    with pytest.raises(ValueError, match="centers"):
        dl.SparseOptic([0.0, 1.0])


@pytest.mark.parametrize("apertures", [2, 3])
@pytest.mark.parametrize("polarised", [False, True])
def test_interference_collapses_shared_aperture_metadata_axis(apertures, polarised):
    wavelengths = np.asarray([5e-4, 6e-4])
    spacing = np.asarray([[0.02, 0.03], [0.025, 0.035]])[None, :, :]
    centers = np.asarray([[0.001, -0.002], [0.003, -0.001]])[None, :, :]
    grid = dl.Grid((5, 6), spacing, centers, "m")
    intrinsic = (2, 2) if polarised else ()
    field = np.ones((apertures, 2) + intrinsic + (6, 5), dtype=complex)
    wavefront = dl.Wavefront(
        wavelengths, grid, field, jones=True if polarised else None
    )
    output = assert_jittable((lambda value: value.coherent_sum()), wavefront)
    assert np.allclose(output.phasor, field.sum(axis=0))
    assert output.grid.d.shape == (2, 2)
    assert output.grid.c.shape == (2, 2)
    assert np.allclose(output.grid.d, spacing[0])
    assert np.allclose(output.grid.c, centers[0])
