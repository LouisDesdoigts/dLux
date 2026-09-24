"""Numerical propagation adapts to fields without owning a second relay model."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable, assert_tree_allclose


@pytest.fixture
def grids():
    return dlu.Grid(4, d=0.1, unit="m"), dlu.Grid(4, d=1e-6, unit="rad")


def test_imager_uses_flat_defocus_on_its_oversampled_image_grid(grids):
    pupil_grid, image_grid = grids
    pupil = dl.Optic(transmission=0.8, defocus=1e-6)
    imager = dl.Imager(pupil_grid, image_grid, pupil, oversample=2)
    wavefront = dl.Wavefront(np.array([5e-7, 7e-7]), pupil_grid)
    expected = pupil(wavefront).propagate(
        dlu.Propagator(image_grid.oversample(2), defocus=1e-6)
    )
    actual = assert_jittable(lambda model: model.apply(wavefront), imager)
    assert_tree_allclose(actual, expected, rtol=2e-6)
    assert imager.propagator.grid_out.n == (8, 8)
    assert np.allclose(imager.propagator.defocus, pupil.defocus)


def test_imager_cartesian_abcd_does_not_require_focal_length(grids):
    pupil_grid, _ = grids
    image_grid = dlu.Grid(4, d=1e-6, unit="m")
    matrix = dlu.ABCD(fraunhofer=0.3)
    pupil = dl.Optic(phase=0.2, abcd=matrix)
    imager = dl.Imager(pupil_grid, image_grid, pupil)
    wavefront = dl.Wavefront(np.array([5e-7, 7e-7]), pupil_grid, jones=True)
    expected = pupil(wavefront).propagate(dlu.Propagator(image_grid, abcd=matrix))
    actual = assert_jittable(lambda model: model.apply(wavefront), imager)
    assert_tree_allclose(actual, expected, rtol=2e-6)
    assert imager.propagator.focal_length is None


@pytest.mark.parametrize("sparse", [False, True])
def test_imager_rejects_local_propagation_at_construction_and_after_update(
    grids, sparse
):
    pupil_grid, image_grid = grids
    pupil = dl.SparseOptic([[0.0, 0.0]]) if sparse else dl.Optic()
    active = pupil.set(grid_out=image_grid)
    with pytest.raises(TypeError, match="grid-preserving|owns propagation"):
        dl.Imager(pupil_grid, image_grid, active)
    imager = dl.Imager(pupil_grid, image_grid, pupil).set(pupil=active)
    with pytest.raises(TypeError, match="grid-preserving|owns propagation"):
        imager.apply(dl.Wavefront(5e-7, pupil_grid))


def test_imager_requires_forward_relay(grids):
    pupil_grid, _ = grids
    image_grid = dlu.Grid(4, d=1e-6, unit="m")
    with pytest.raises(ValueError, match="forward"):
        dl.Imager(pupil_grid, image_grid, dl.Optic(focal_length=0.3, inverse=True))


def test_imager_rejects_a_bare_propagator_without_output_sampling(grids):
    pupil_grid, image_grid = grids
    with pytest.raises(TypeError, match="grid-preserving"):
        dl.Imager(pupil_grid, image_grid, dl.PropagatorLayer())


def test_imager_sequence_uses_one_relay_owner_and_reports_ambiguity(grids):
    pupil_grid, image_grid = grids
    default = dlu.Sequential([dl.Optic(), dl.Optic(transmission=0.8)])
    assert dl.Imager(pupil_grid, image_grid, default).propagator.defocus is None

    one_relay = dlu.Sequential(
        {"aperture": dl.Optic(), "relay": dl.Optic(defocus=1e-6)}
    )
    model = dl.Imager(pupil_grid, image_grid, one_relay)
    assert np.allclose(model.propagator.defocus, 1e-6)

    ambiguous = one_relay.set("layers.aperture.defocus", np.asarray(2e-6))
    with pytest.raises(ValueError, match="multiple relay definitions"):
        dl.Imager(pupil_grid, image_grid, ambiguous)
    explicit = dlu.Sequential(
        ambiguous.layers, alias=(("defocus", "layers.relay.defocus"),)
    )
    assert np.allclose(
        dl.Imager(pupil_grid, image_grid, explicit).propagator.defocus, 1e-6
    )


def test_optical_system_adapts_nested_numerical_steps_for_apply_and_debug(grids):
    pupil_grid, _ = grids
    wavefront = dl.Wavefront(np.array([5e-7, 7e-7]), pupil_grid, jones=True)
    optic = dl.Optic(phase=0.2)
    free_space = dlu.FreeSpace(0.2)
    fft = dlu.FFTPropagator()
    system = dl.OpticalSystem(
        {"pupil": optic, "space": dlu.Sequential([free_space]), "focus": fft}
    )
    expected = optic(wavefront).propagate(free_space).propagate(fft)
    actual = assert_jittable(system, wavefront, rtol=2e-6)
    debug_output, stages = system.debug(wavefront)
    assert_tree_allclose(actual, expected, rtol=2e-6)
    assert_tree_allclose(debug_output, expected, rtol=2e-6)
    assert tuple(stages) == ("input", "pupil", "space", "focus")

    inserted = dl.OpticalSystem([optic]).insert_layer(free_space, 1)
    assert_tree_allclose(inserted(wavefront), optic(wavefront).propagate(free_space))


@pytest.mark.parametrize("plane", ["pupil", "focal", "lyot"])
def test_coronagraph_rejects_active_propagation_in_updated_local_planes(grids, plane):
    pupil_grid, image_grid = grids
    model = dl.Coronagraph(
        pupil_grid,
        image_grid,
        image_grid,
        pupil=dl.Optic(),
        focal=dl.Optic(),
        lyot=dl.Optic(),
    ).set(f"{plane}.grid_out", image_grid)
    with pytest.raises(TypeError, match="grid-preserving"):
        model.apply(dl.Wavefront(5e-7, pupil_grid))
