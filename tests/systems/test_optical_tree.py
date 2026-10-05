"""Recursive optical paths retain source flux and fixed wavelength routing."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl


def make_source(flux=100.0):
    band = dl.BandPass(range=(500, 900), n=4, unit="nm")
    return dl.PointSource(spectrum=dl.Spectrum(1.0, band, flux=flux))


def make_tree(branches, source=None, transition=None):
    pupil = dl.Grid(8, diam=1, unit="m")
    source = make_source() if source is None else source
    return dl.OpticalTree(
        source,
        dl.OpticalSystem([], pupil),
        branches,
        transition,
    )


def test_nested_branches_prune_relative_to_parent_and_share_common_optics():
    calls = []

    class CountingSystem(dl.OpticalSystem):
        def apply(self, wavefront, **context):
            calls.append("K")
            return super().apply(wavefront, **context)

    branches = {
        "H": dl.OpticalBranch(
            dl.OpticalSystem([]),
            dl.Filter(1, dl.BandPass(range=(500, 600), n=1, unit="nm")),
        ),
        "K": dl.OpticalBranch(
            CountingSystem([dl.Optic(transmission=0.8)]),
            dl.Filter(1, dl.BandPass(range=(600, 900), n=3, unit="nm")),
            children={
                "K1": dl.OpticalBranch(
                    dl.OpticalSystem([]),
                    dl.Filter(1, dl.BandPass(range=(600, 700), n=1, unit="nm")),
                ),
                "K2": dl.OpticalBranch(
                    dl.OpticalSystem([]),
                    dl.Filter(1, dl.BandPass(range=(700, 900), n=2, unit="nm")),
                ),
            },
        ),
    }
    original = make_tree(branches)
    with pytest.raises(ValueError, match="prepared"):
        original.model()
    with pytest.raises(ValueError, match="prepared"):
        original.step(zdx.State(), {}, 0.01, jax.random.key(0))
    tree = original.prepare()

    assert original.branches["K"].indices is None
    assert tree.branches["H"].indices == (0,)
    assert tree.branches["K"].indices == (1, 2, 3)
    assert tree.branches["K"].children["K1"].indices == (0,)
    assert tree.branches["K"].children["K2"].indices == (1, 2)
    rates = tree.model()
    assert calls == ["K"]
    assert tuple(rates) == ("H", "K.K1", "K.K2")
    assert np.allclose(
        np.array([rate.data.sum() for rate in rates.values()]),
        np.array([25, 16, 32]),
        rtol=2e-6,
    )


def test_equal_split_and_source_flux_gradient_under_jit():
    tree = make_tree(
        {
            "first": dl.OpticalBranch(dl.OpticalSystem([]), dl.Filter(0.5)),
            "second": dl.OpticalBranch(dl.OpticalSystem([]), dl.Filter(0.5)),
        }
    ).prepare()
    eager = tree.model()
    compiled = eqx.filter_jit(lambda model: model.model())(tree)
    for name in ("first", "second"):
        assert np.allclose(eager[name].data, compiled[name].data)
        assert np.allclose(eager[name].data.sum(), 50, rtol=2e-6)

    def first_rate(flux):
        model = tree.set("source.spectrum.flux", flux)
        return model.model()["first"].data.sum()

    assert np.allclose(jax.grad(first_rate)(np.array(100.0)), 0.5, rtol=2e-6)


def test_multiple_source_positions_are_summed_incoherently():
    band = dl.BandPass(range=(500, 900), n=4, unit="nm")
    source = dl.Source(
        spectrum=dl.Spectrum(1.0, band, flux=np.array([100.0, 200.0])),
        positions=np.zeros((2, 2)),
    )
    tree = make_tree(
        {"camera": dl.OpticalBranch(dl.OpticalSystem([]))}, source=source
    ).prepare()
    assert np.allclose(tree.model()["camera"].data.sum(), 300, rtol=2e-6)


def test_detector_sequence_acts_on_integrated_rate():
    detector = dl.DetectorSystem([dl.SensitivityLayer(0.8)])
    tree = make_tree(
        {"camera": dl.OpticalBranch(dl.OpticalSystem([]), detector=detector)}
    ).prepare()
    assert np.allclose(tree.model()["camera"].data.sum(), 80, rtol=2e-6)


def test_compact_jones_field_keeps_its_wavelength_axis_when_pruned():
    pupil = dl.Grid(8, diam=1, unit="m")
    jones = np.array([[1.0, 0.0], [0.0, 0.5]])
    branch = dl.OpticalBranch(
        dl.OpticalSystem([]),
        dl.Filter(1, dl.BandPass(range=(500, 600), n=1, unit="nm")),
    )
    tree = dl.OpticalTree(
        make_source(),
        dl.OpticalSystem([dl.Optic(jones=jones)], pupil),
        {"camera": branch},
    ).prepare()
    expected = 25 * (1 + 0.5**2) / 2
    assert np.allclose(tree.model()["camera"].data.sum(), expected, rtol=2e-6)


def test_linked_transmission_is_owned_once_across_branches():
    fraction = zdx.Linked(0.25, key="split")
    tree = make_tree(
        {
            "first": dl.OpticalBranch(dl.OpticalSystem([]), dl.Filter(fraction)),
            "second": dl.OpticalBranch(
                dl.OpticalSystem([]), dl.Filter(fraction.defer())
            ),
        }
    ).prepare()
    assert np.allclose(
        np.array([rate.data.sum() for rate in tree.model().values()]),
        np.array([25, 25]),
        rtol=2e-6,
    )

    def total(value):
        model = tree.set("branches.first.transmission.value.value", value)
        return sum(rate.data.sum() for rate in model.model().values())

    assert np.allclose(jax.grad(total)(np.array(0.25)), 200, rtol=2e-6)


def test_reprepare_after_fixed_filter_support_changes():
    first = dl.Filter(1, dl.BandPass(range=(500, 600), n=1, unit="nm"))
    tree = make_tree(
        {"camera": dl.OpticalBranch(dl.OpticalSystem([]), first)}
    ).prepare()
    other = dl.Filter(1, dl.BandPass(range=(800, 900), n=1, unit="nm"))
    changed = tree.set("branches.camera.transmission", other).prepare()
    assert tree.branches["camera"].indices == (0,)
    assert changed.branches["camera"].indices == (3,)
    assert np.allclose(changed.model()["camera"].data.sum(), 25, rtol=2e-6)


def test_state_transition_runs_before_optics_and_step_advances_time():
    class CommandTransition(dl.StateTransition):
        def apply(self, state, inputs, dt, key):
            return state.set(opd=inputs["opd"])

    pupil = dl.Grid(8, diam=1, unit="m")
    state = zdx.State(index=0, time=0.0, opd=np.zeros(pupil.shape))
    tree = dl.OpticalTree(
        make_source(),
        dl.OpticalSystem([dl.Optic(opd=state.ref("opd"))], pupil),
        {"camera": dl.OpticalBranch(dl.OpticalSystem([]))},
        CommandTransition(),
    ).prepare()
    command = np.ones(pupil.shape) * 1e-8

    def step(model, value):
        return model.step(value, {"opd": command}, 0.01, jax.random.key(0))

    eager_state, eager_rates = step(tree, state)
    updated, rates = eqx.filter_jit(step)(tree, state)
    assert np.allclose(updated.opd, command)
    assert int(updated.index) == 1
    assert np.allclose(updated.time, 0.01)
    assert np.allclose(rates["camera"].data.sum(), 100, rtol=2e-6)
    assert np.allclose(updated.opd, eager_state.opd)
    assert np.allclose(updated.time, eager_state.time)
    assert np.allclose(rates["camera"].data, eager_rates["camera"].data)


def test_prepare_rejects_source_without_fixed_bandpass():
    tree = make_tree(
        {"camera": dl.OpticalBranch(dl.OpticalSystem([]))},
        source=dl.PointSource(flux=100),
    )
    with pytest.raises(TypeError, match="fixed BandPass"):
        tree.prepare()
