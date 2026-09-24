"""Imager owns source, wavelength and aperture roles at its array boundaries."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


@pytest.fixture
def model():
    return dl.Imager(
        dl.Grid(n=6, diam=1, unit="m"),
        dl.Grid(n=8, d=1e-7, unit="rad"),
        source=dl.Source(positions=[[0, 0], [1e-7, 0], [0, -1e-7]], flux=[1, 2, 3]),
        bandpass=dlu.BandPass(6e-7, 2e-7, n=3),
    )


@pytest.mark.parametrize(
    "grid,error,message",
    [
        (None, TypeError, "must be a Grid"),
        (dl.Grid(6, unit="m"), ValueError, "must define"),
        (dl.Grid(6, d=1, unit="1/m"), ValueError, "length or angular"),
        (dl.Grid(6, d=1, unit="rad"), ValueError, "physical length"),
    ],
)
def test_input_grid_validation_tails(model, grid, error, message):
    with pytest.raises(error, match=message):
        dl.Imager(grid, model.image_grid)
    if grid is not None:
        with pytest.raises(error, match=message):
            model.psf(pupil_grid=grid)


def test_stage_input_validation_tails(model):
    # These check helpful public boundary errors, rather than numerical coverage.
    with pytest.raises(TypeError, match="Wavefront"):
        model.apply(np.ones((6, 6)))
    with pytest.raises(ValueError, match="offset must have shape"):
        model.initialise_wavefront(6e-7, offset=[1, 2, 3])
    with pytest.raises(ValueError, match="wavelength must be scalar"):
        model.propagate_mono([5e-7, 6e-7])
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        model.propagate([])
    with pytest.raises(ValueError, match="weights must be real"):
        model.propagate([5e-7, 6e-7], weights=[1])
    with pytest.raises(ValueError, match="Supply pupil or optics"):
        model.psf(pupil=dl.Optic(), optics=dl.Optic())
    with pytest.raises(TypeError, match="individual sources"):
        model.intensity([dl.Source()], spectrum=dl.Spectrum())
    with pytest.raises(TypeError, match="channel names must be strings"):
        model.psf(source={1: dl.Source()})


@pytest.mark.parametrize(
    "name,value",
    [
        ("filter", dlu.Spectrum()),
        ("bandpass", dlu.Filter()),
        ("bandpass", dlu.Spectrum()),
    ],
)
def test_wrong_photometric_module_has_a_named_error(model, name, value):
    with pytest.raises(TypeError, match=name):
        dl.Imager(model.pupil_grid, model.image_grid, **{name: value})
    with pytest.raises(TypeError, match=name):
        model.intensity(**{name: value})


def test_spectrum_override_promotes_values_and_rejects_responses(model):
    expected = model.intensity(spectrum=dlu.Spectrum([1, 2, 3])).data
    actual = model.intensity(spectrum=zdx.Map(x=[1, 2, 3])).data
    assert np.allclose(actual, expected)
    with pytest.raises(TypeError, match="spectrum"):
        model.intensity(spectrum=dlu.Filter())


@pytest.mark.parametrize("name", ["filter", "qe"])
@pytest.mark.parametrize("shape", [(1, 3), (3, 3)])
def test_responses_are_shared_curves_even_when_source_and_wavelength_counts_match(
    model, name, shape
):
    # Three source components and three wavelengths must not accidentally make a
    # (3,3) response valid as a per-source filter or QE definition.
    response = dlu.Filter(zdx.Map(x=np.ones(shape)))
    if name == "filter":
        model = model.set(filter=response)
    else:
        model = model.set(detector=dl.Detector(qe=response))
    with pytest.raises(ValueError, match="one shared wavelength curve"):
        model.intensity()


def test_shared_response_and_root_scene_links_keep_their_roles(model):
    shared = zdx.Linked(np.asarray(2.0), key="brightness")
    scene = dl.Scene(
        {
            "primary": dl.Source(flux=shared),
            "group": dl.Scene({"companion": dl.Source(flux=shared.defer())}),
        }
    )
    linked = model.set(source=scene, filter=dlu.Filter([0.2, 0.5, 0.8]))
    expected = model.intensity(
        dl.Source(flux=4), filter=dlu.Filter([0.2, 0.5, 0.8])
    ).data
    output = assert_jittable(lambda system: system.intensity().data, linked)
    assert np.allclose(output, expected)
    gradient = jax.jit(
        jax.grad(
            lambda value: (
                linked.set("source.sources.primary.flux.value", value)
                .intensity()
                .data.sum()
            )
        )
    )(np.asarray(2.0))
    assert np.allclose(gradient, expected.sum() / 2, rtol=2e-5)


@pytest.mark.parametrize(
    "source,message",
    [
        (dl.Source().set(positions=zdx.Map(x=np.zeros((0, 2)))), "non-empty shape"),
        (dl.Source().set(positions=zdx.Map(x=np.zeros((2, 3)))), "non-empty shape"),
        (
            dl.Source(positions=np.zeros((3, 2))).set(flux=zdx.Map(x=[1])),
            "flux must resolve",
        ),
        (dl.Source().set(flux=zdx.Map(x=[[1]])), "flux must resolve"),
    ],
)
def test_resolved_source_shapes_have_owner_errors(model, source, message):
    with pytest.raises(ValueError, match=message):
        model.intensity(source)


@pytest.mark.parametrize("sources", [{}, {"bad": object()}, {1: dl.Source()}])
def test_scene_updates_are_checked_without_evaluating_children(model, sources):
    scene = dl.Scene({"star": dl.Source()}).set(sources=sources)
    with pytest.raises((TypeError, ValueError), match="Scene|Source"):
        model.intensity(scene)


def test_resolved_scene_offset_is_one_shared_position(model):
    scene = dl.Scene({"star": model.source}).set(position=zdx.Map(x=np.zeros((3, 2))))
    with pytest.raises(ValueError, match="Scene position must resolve"):
        model.intensity(scene)
    with pytest.raises(ValueError, match="Scene position must resolve"):
        model.wavefront(scene)


def test_source_spectrum_type_and_sampling_units_are_checked(model):
    with pytest.raises(TypeError, match="Source.spectrum"):
        model.intensity(dl.Source().set(spectrum=dlu.Filter()))
    with pytest.raises(ValueError, match="bandpass unit"):
        model.intensity(bandpass=dlu.Axis([1, 2, 3], unit="rad"))


def test_dense_optical_arrays_cannot_add_an_unlabelled_model_batch(model):
    pupil = dl.Optic(transmission=np.ones((2, 1, 1, 6, 6)))
    with pytest.raises(ValueError, match="broadcast to the existing field shape"):
        model.set(pupil=pupil).intensity()


def test_independent_models_use_explicit_vmap(model):
    def calculate(transmission):
        return model.set(pupil=dl.Optic(transmission=transmission)).intensity().data

    transmissions = np.array([0.3, 0.8])
    actual = eqx.filter_jit(jax.vmap(calculate))(transmissions)
    expected = np.stack([calculate(value) for value in transmissions])
    assert np.allclose(actual, expected, rtol=2e-5, atol=1e-7)


def test_equal_wavelength_source_and_aperture_counts_remain_independent(model):
    aperture = dl.SparseOptic(
        [[-0.4, 0], [0.0, 0.1], [0.4, 0]],
        transmission=dlu.Circle(0.6),
        opd=dlu.Basis(np.ones((1, 6, 6)), coeffs=[[0], [3e-8], [-2e-8]], axes=1),
    )
    system = model.set(pupil=aperture)
    actual = assert_jittable(lambda value: value.wavefront().phasor, system)
    wavelengths = model.bandpass.wavelengths
    rows = []
    for wavelength in wavelengths:
        fields = []
        for position in model.source.positions:
            entrance = dl.Wavefront(wavelength, model.pupil_grid)
            entrance = entrance.set(phasor=entrance.phasor / np.sqrt(3))
            local = aperture(entrance).tilt(position)
            field = dl.PropagatorLayer(model.image_grid)(local).coherent_sum()
            fields.append(field.phasor)
        rows.append(np.stack(fields))
    # Rows were calculated wavelength-first; public fields are source-first.
    expected = np.stack(rows).swapaxes(0, 1)
    assert actual.shape == (3, 3, 8, 8)
    assert np.allclose(actual, expected, rtol=3e-5, atol=2e-7)


def test_resolved_jitter_is_one_image_kernel(model):
    model = model.set(jitter=zdx.Map(x=np.ones((3, 3, 3))))
    with pytest.raises(ValueError, match="jitter must resolve"):
        model.intensity()


def test_wrong_sampling_module_from_immutable_update_has_a_named_error(model):
    with pytest.raises(TypeError, match="bandpass"):
        model.set(bandpass=dlu.Spectrum()).intensity()


def test_shared_spatial_map_wins_when_pixel_and_population_sizes_match():
    grid = dl.Grid(n=3, diam=1, unit="m")
    image = dl.Grid(n=3, d=1e-7, unit="rad")
    pupil = dl.Optic(
        transmission=np.array([[0.2, 0.3, 0.9], [1.0, 0.4, 0.7], [0.6, 0.8, 0.1]])
    )
    source = dl.Source(positions=[[0, 0], [1e-7, 0], [0, -1e-7]])
    model = dl.Imager(
        grid, image, pupil, source=source, bandpass=dlu.BandPass(6e-7, 2e-7, n=3)
    )
    actual = assert_jittable(lambda system: system.wavefront().phasor, model)
    rows = []
    for wavelength in model.bandpass.wavelengths:
        fields = []
        for position in source.positions:
            incident = dl.Wavefront(wavelength, grid).tilt(position)
            fields.append(dl.PropagatorLayer(image)(pupil(incident)).phasor)
        rows.append(np.stack(fields))
    # Rows were calculated wavelength-first; public fields are source-first.
    expected = np.stack(rows).swapaxes(0, 1)
    assert actual.shape == (3, 3, 3, 3)
    assert np.allclose(actual, expected, rtol=3e-5, atol=2e-7)
