"""Field conversions, coherent arithmetic and remaining input-contract branches."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_intensity_containers_expose_values_and_validate_conversions():
    grid = dlu.Grid(3, 1.0)
    data = np.arange(9.0).reshape(3, 3)
    for cls in (dl.PSF, dl.Intensity, dl.Image):
        field = cls(data, grid)
        assert np.array_equal(field.intensity, data)
        with pytest.raises(ValueError, match="mode"):
            field.normalise("unknown")
    assert dl.PSF(data[None], grid).ndim == 1
    for call in (
        lambda: dl.Image.from_intensity(data),
        lambda: dl.Intensity.from_wavefront(data),
        lambda: dl.PSF.from_wavefront(data),
        lambda: dl.PSF(data, None),
    ):
        with pytest.raises(TypeError):
            call()


def test_none_modulation_and_retained_constructor(make_wavefront):
    field = make_wavefront()
    assert field.add_phase(None) is field
    assert field.add_opd(None) is field
    assert field + None is field
    with pytest.warns(DeprecationWarning):
        restored = dl.Wavefront.from_phasor(field.phasor, field.wavelength, field.grid)
    assert np.array_equal(restored.phasor, field.phasor)
    with pytest.raises(ValueError, match="Jones axes"):
        dl.Wavefront(1.0, field.grid, np.ones((2, 3, 8, 8)), jones=True)


def test_compact_population_jones_arithmetic_and_flipping():
    grid = dlu.Grid(3, 0.1)
    wavelengths = np.array([0.5, 0.7, 0.9])
    scalar = dl.Wavefront(wavelengths, grid, np.ones((2, 3, 3, 3)))
    matrices = np.array([1.0, 2.0])[:, None, None] * np.eye(2)
    mono = dl.Wavefront(0.5, grid, np.ones((2, 3, 3)), jones=matrices)
    compact = assert_jittable(lambda a, b: a * b, scalar, mono)
    assert compact.jones.shape == (2, 1, 2, 2)
    expected = (
        scalar.materialise().phasor
        * np.array([1.0, 2.0])[:, None, None, None, None, None]
    )
    assert np.allclose(compact.materialise().phasor, expected)
    assert np.allclose(compact.flip(0).materialise().phasor, np.flip(expected, 0))
    assert np.allclose(
        compact.flip((-2, -1)).materialise().phasor, np.flip(expected, (-2, -1))
    )

    # Addition combines transfer matrices; multiplication stays elementwise.
    for operation in (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b):
        actual = assert_jittable(operation, compact, compact)
        reference = operation(compact.materialise(), compact.materialise())
        assert np.allclose(actual.materialise().phasor, reference.phasor)
    assert np.allclose((compact + 1).phasor, compact.materialise().phasor + 1)
    explicit = compact.materialise()
    assert np.allclose((compact * explicit).phasor, explicit.phasor**2)


def test_coherent_sum_materialises_varying_jones_but_keeps_shared_matrices():
    grid = dlu.Grid(3, 0.1)
    samples = np.ones((2, 3, 3), dtype=complex)
    matrices = np.array([1.0, 2.0])[:, None, None] * np.eye(2)
    field = dl.Wavefront(0.5, grid, samples, jones=matrices)
    result = assert_jittable(lambda x: x.coherent_sum(), field)
    assert result.jones is True
    assert np.allclose(result.phasor, field.materialise().phasor.sum(0))
    shared = field.set(jones=np.eye(2)[None])
    result = shared.coherent_sum()
    assert result.jones.shape == (2, 2)
    assert np.allclose(result.intensity, 4)


def test_explicit_jones_convolution_and_stokes():
    grid = dlu.Grid(5, 0.1)
    field = dl.Wavefront(0.5, grid).materialise()
    kernel = np.ones((3, 3)) / 9
    result = assert_jittable(lambda x: x.convolve(kernel), field)
    reference = dl.Wavefront(0.5, grid).convolve(kernel).materialise()
    assert np.allclose(result.phasor, reference.phasor, atol=1e-7)
    stokes = np.array([2.0, 0.3, 0.2, 0.1])
    assert np.allclose(field.stokes(stokes), stokes[:, None, None] / 25)
    assert np.allclose(field.intensity_from_stokes(stokes), 2 / 25)
    scalar = dl.Wavefront(0.5, grid)
    assert np.allclose(scalar.psf_from_stokes(stokes), 2 / 25)


def test_wavefront_invalid_shapes_are_rejected_at_the_boundary(make_wavefront):
    """Conditional tails: ambiguous axes and incompatible operators remain errors."""
    field = make_wavefront()
    cases = [
        (ValueError, lambda: field.add_phase(np.ones(2))),
        (ValueError, lambda: field.add_phase(np.ones((2, 8, 8)))),
        (ValueError, lambda: field.apply_jones(np.ones((3, 3)))),
        (ValueError, lambda: field.apply_jones(np.ones((3, 8, 2, 2)), spatial=True)),
        (ValueError, lambda: field.stokes(np.ones(3))),
        (ValueError, lambda: field.intensity_from_stokes(np.ones(3))),
        (ValueError, lambda: field.resample_wavelengths(np.ones(2))),
        (
            ValueError,
            lambda: dl.Wavefront(np.array([1.0, 2.0]), field.grid).resample_wavelengths(
                1.0
            ),
        ),
        (TypeError, lambda: field * dlu.Field(np.ones((8, 8)), field.grid)),
        (TypeError, lambda: field.propagate(1)),
    ]
    for error, call in cases:
        with pytest.raises(error):
            call()


def test_spectral_interpolation_rejects_different_spatial_sampling():
    field = dl.Wavefront(
        np.array([1.0, 2.0]), dlu.Grid(3, d=np.array([[1.0, 1.0], [2.0, 2.0]]))
    )
    with pytest.raises(ValueError, match="common spatial grid"):
        field.resample_wavelengths(np.array([1.5]))


def test_propagation_cannot_introduce_population_axes(make_wavefront):
    field = make_wavefront()
    target = dlu.Grid(3, d=np.array([[0.1, 0.1], [0.2, 0.2]]), unit="m")
    with pytest.raises(ValueError, match="Wavefront batch"):
        field.propagate(dlu.Propagator(target, focal_length=1.0))


def test_jones_metadata_cannot_add_or_mislabel_field_axes(make_wavefront):
    field = make_wavefront()
    with pytest.raises(ValueError, match="end in"):
        dl.Wavefront(field.wavelength, field.grid, jones=np.ones((3, 3)))
    extra_population = np.broadcast_to(np.eye(2), (3, 1, 1, 2, 2))
    with pytest.raises(ValueError, match="existing field"):
        field.apply_jones(extra_population, spatial=True)
