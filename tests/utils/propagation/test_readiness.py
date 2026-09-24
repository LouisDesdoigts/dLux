"""Public propagation diagnostics and preparation edge cases.

The invalid-input cases cover conditional tails; the numerical checks exercise
shifts, separable composition, sampling margins and physical phase conventions.
"""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.utils import propagation as p
from dLux.utils.propagation.operations import _series


def test_abcd_observables_and_tolerance():
    matrix = np.array([[2.0, 3.0], [-1.0, -1.0]])
    definition = p.ABCD(matrix=matrix)
    assert np.allclose(
        np.array([definition.a, definition.b, definition.c, definition.d]),
        np.array([2, 3, -1, -1]),
    )
    assert np.allclose(
        np.array(
            [
                definition.effective_focal_length,
                definition.paraxial_power,
                definition.geometric_magnification,
                definition.angular_magnification,
                definition.distance,
                definition.front_focal_length,
                definition.back_focal_length,
            ]
        ),
        np.array([1, 1, 2, -1, 3, 1, 2]),
    )
    assert p.abcd_is_unimodular(np.eye(2, dtype=int))
    assert p.abcd_is_unimodular(matrix, tol=1e-5)
    assert not p.abcd_is_unimodular(matrix * 2, tol=1e-5)
    with pytest.raises(ValueError, match="non-negative"):
        p.abcd_is_unimodular(matrix, tol=-1)
    with pytest.raises(ValueError, match="finite"):
        p.abcd_is_unimodular(matrix, tol=np.inf)
    with pytest.raises(ValueError, match="real"):
        p.ABCD.validate(matrix.astype(complex))


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_fourier_shift_is_a_spatial_translation(complex_input, pad):
    array = np.zeros((2, 7, 9)).at[:, 3, 4].set(np.array([1.0, 2.0]))
    if complex_input:
        array = array * (1 + 2j)
    operation = eqx.filter_jit(
        lambda a: p.fourier_shift(a, np.array([1.0, -1.0]), pad=pad)
    )
    expected = np.roll(array, (-1, 1), axis=(-2, -1))
    assert np.allclose(operation(array), expected, atol=1e-6)


@pytest.mark.parametrize(
    "args,kwargs,error",
    [
        ((np.ones(3), [0, 0]), {}, ValueError),
        ((np.ones((3, 3)), [0]), {}, ValueError),
        ((np.ones((3, 3)), [0, 0]), {"pad": 1}, TypeError),
    ],
)
def test_fourier_shift_validation(args, kwargs, error):
    with pytest.raises(error):
        p.fourier_shift(*args, **kwargs)


def test_mft_sampling_diagnostics_and_broadcast():
    axis = np.array([-1.0, 0.0, 1.0])
    margins = eqx.filter_jit(p.mft_phase_parameter)(axis, axis, np.pi / 2)
    assert np.allclose(np.array(margins), 2)
    assert np.isinf(p.mft_sampling_parameter(axis, axis, 0.0))
    assert np.isclose(p.mft_sampling_parameter(axis, axis, np.pi / 2), 2)
    identity = p.MFT((np.eye(3), np.eye(3)))
    assert identity.output_shape == (3, 3)
    batch = identity.broadcast((2,))
    values = np.arange(18.0).reshape(2, 3, 3)
    assert np.allclose(batch(values), values)
    chirp = p.Chirp((np.ones(3), np.ones(3) * 2))
    assert np.allclose(chirp.broadcast((2,))(values), 2 * values)
    fft = p.FFT(3)
    assert fft.n_in == fft.n_out == (3, 3)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((0.0, np.arange(3), 1.0), {}),
        ((np.arange(1), np.arange(3), 1.0), {}),
        ((np.arange(3), np.arange(3), 1.0), {"eps": [1e-5]}),
        ((np.arange(3), np.arange(3), 1.0), {"eps": -1.0}),
        ((np.arange(3), np.arange(3), 1.0), {"eps": np.inf}),
    ],
)
def test_sampling_diagnostic_validation(args, kwargs):
    with pytest.raises(ValueError):
        p.mft_phase_parameter(*args, **kwargs)


def test_separable_preparation_validation():
    """Conditional tails: malformed kernels, factors and incompatible compositions."""
    axis = np.arange(3.0)
    matrix = p.MFT((np.eye(3), np.eye(3)))
    cases = [
        lambda: p.mft_kernel(0.0, axis, 1),
        lambda: p.mft_kernel(axis[:0], axis, 1),
        lambda: p.mft_kernels((axis,), (axis, axis), 1),
        lambda: p.mft_kernels((axis, axis), (axis, axis), (1,)),
        lambda: p.MFT(()),
        lambda: p.MFT((np.eye(3),)),
        lambda: p.MFT((np.zeros((0, 3)), np.eye(3))),
        lambda: matrix.then(p.MFT((np.eye(4), np.eye(4)))),
        lambda: matrix.fuse_input((axis,)),
        lambda: matrix.fuse_output((axis,)),
        lambda: matrix.fuse_input((axis[:2], axis)),
        lambda: matrix.fuse_output((axis, axis[:2])),
        lambda: matrix.fuse_input((0.0, axis)),
        lambda: p.Chirp((axis,)),
        lambda: p.Chirp.from_axes((axis, axis)),
        lambda: p.Chirp.from_axes((axis, axis), linear=(1,)),
        lambda: p.Chirp((axis, axis)).then(p.Chirp((axis[:2], axis))),
        lambda: p.Chirp((axis, axis))(np.ones((4, 4))),
        lambda: p.FFT.prepare(5, 3),
        lambda: p.FFT.prepare(3, input=(axis[:2], axis)),
        lambda: p.FFT.prepare(3, n_out=3, output=(axis[:2], axis)),
        lambda: p.FFT(3)(np.ones((4, 4))),
        lambda: _series(),
    ]
    for call in cases:
        with pytest.raises(ValueError):
            call()
    with pytest.raises(TypeError):
        matrix.then(p.FFT(3))
    with pytest.raises(TypeError):
        _series(1)


def test_thin_power_and_free_space_definitions():
    grid = dlu.Grid(5, 0.1, unit="m")
    wavelength = 0.02
    field = np.ones(grid.shape, dtype=complex)
    lens = p.ThinPower(0.3)
    actual = eqx.filter_jit(lambda a: lens(a, grid, wavelength))(field)
    radius2 = np.sum(grid.coordinates**2, axis=0)
    assert np.allclose(
        actual, np.exp(-1j * np.pi * 0.3 * radius2 / wavelength), atol=2e-6
    )
    assert np.allclose(p.FreeSpace(0.2).abcd.resolve(), np.array([[1, 0.2], [0, 1]]))


def test_propagation_preparation_validation():
    """Conditional tails: incompatible setup is rejected before applying kernels."""
    grid = dlu.Grid(5, 0.1, unit="m")
    field = np.ones(grid.shape)
    cases = [
        (TypeError, lambda: p.Propagator(grid)(field, grid, 1.0, return_grid=1)),
        (TypeError, lambda: p.Propagator(grid).prepare(grid, 1.0, return_grid=1)),
        (TypeError, lambda: p.Propagator(grid).prepare(1, 1.0)),
        (ValueError, lambda: p.Propagator(grid).prepare(grid, 0.0)),
        (TypeError, lambda: p.LCT(np.eye(2), None)),
        (TypeError, lambda: p.SpectralPropagator(np.eye(2), None, grid)),
        (TypeError, lambda: p.SpectralPropagator(np.eye(2), dlu.FFTSpec(), None)),
        (ValueError, lambda: p.FreeSpace(0.1, dlu.FFTSpec(c=1))),
        (ValueError, lambda: p.FreeSpace(0.1, dlu.ResizeSpec(n=3)).prepare(grid, 1.0)),
        (
            ValueError,
            lambda: p.Propagator(grid, abcd=np.eye(2))
            .set(defocus=1.0)
            .prepare(grid, 1.0),
        ),
        (
            ValueError,
            lambda: p.SpectralPropagator(
                p.ABCD(free_space=1.0), dlu.FFTSpec(n=3), grid
            ).prepare(grid, 1.0),
        ),
        (TypeError, lambda: p.FFTPropagator().set(grid_out=grid).prepare(grid, 1.0)),
        (TypeError, lambda: p.Propagator(grid).set(grid_out=1).prepare(grid, 1.0)),
    ]
    for error, call in cases:
        with pytest.raises(error):
            call()


def test_shifted_spectral_fft_matches_an_explicit_frequency_grid():
    grid = dlu.Grid(5, 0.1, unit="m")
    output = dlu.Grid(5, 0.12, unit="m")
    abcd = p.ABCD(free_space=0.2)
    spec = dlu.FFTSpec(c=(0.3, -0.2))
    recipe = p.SpectralPropagator(abcd, spec, output)
    field = np.arange(25.0).reshape(grid.shape).astype(complex)
    # Frequency sampling is reciprocal to the input spatial sampling.
    frequency = dlu.Grid(5, d=2.0, c=(0.3, -0.2), unit="1/m")
    reference = p.SpectralPropagator(abcd, frequency, output)
    actual = eqx.filter_jit(lambda a: recipe(a, grid, 0.01))(field)
    assert np.allclose(actual, reference(field, grid, 0.01), rtol=2e-5, atol=2e-5)


def test_optional_output_and_resize_preparation():
    """Cover missing-output validation and the plain ResizeSpec convenience input."""
    grid = dlu.Grid(5, 0.1, unit="m")
    with pytest.raises(ValueError, match="requires grid_out"):
        p.Propagator().prepare(grid, 0.01)

    field = np.arange(25.0).reshape(grid.shape).astype(complex)
    resized = p.FFTPropagator(dlu.ResizeSpec(pad=3))
    explicit = p.FFTPropagator(dlu.FFTSpec(pad=3))
    assert np.allclose(resized(field, grid, 0.01), explicit(field, grid, 0.01))


def test_physical_inputs_share_singular_and_plural_wavelength_context():
    import zodiax as zdx

    class Spacing(zdx.Expression):
        def evaluate(self, wavelength, **context):
            return np.stack((wavelength, wavelength), axis=-1)

    class FocalLength(zdx.Expression):
        def evaluate(self, wavelengths, **context):
            return 10 * wavelengths

    source = dlu.Grid(3, 0.1, unit="m")
    target = dlu.Grid(3, d=Spacing(), unit="m")
    model = p.Propagator(target, focal_length=FocalLength())
    field = np.ones(source.shape)
    actual = eqx.filter_jit(lambda a: model(a, source, 0.02))(field)
    expected = p.Propagator(dlu.Grid(3, 0.02, unit="m"), focal_length=0.2)(
        field, source, 0.02
    )
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_mft_input_and_spectral_crop_validation():
    """Conditional tails for array application and a cropped FFT spectrum."""
    with pytest.raises(ValueError, match="input shape"):
        p.MFT((np.eye(3), np.eye(3)))(np.ones((4, 4)))
    grid = dlu.Grid(5, 0.1, unit="m")
    operation = p.SpectralPropagator(
        p.ABCD(free_space=0.1), dlu.FFTSpec(pad=3, crop=3), grid
    )
    with pytest.raises(ValueError, match="may not crop"):
        operation.prepare(grid, 0.01)
