"""Numerical composition contracts across coordinate and photometry objects."""

import jax
import jax.numpy as np

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_transform_composes_affine_then_polynomial_with_gradients():
    coordinates = dlu.Grid((5, 3), d=(0.2, 0.3)).coordinates
    affine = dlu.Affine(translation=[0.15, -0.05], rotation=0.2)
    polynomial = dlu.Distortion(
        powers=[[2, 0], [0, 2]], distortion=[[0.2, 0.1], [-0.05, 0.15]]
    )
    transform = dlu.Transform(affine=affine, polynomial=polynomial)

    output = assert_jittable(lambda model: model(coordinates), transform)
    expected = polynomial(affine(coordinates))
    assert np.allclose(output, expected, atol=1e-6)
    assert not np.allclose(output, affine(polynomial(coordinates)))

    composed_loss = lambda shift: transform.set("affine.translation", shift)(
        coordinates
    ).sum()
    explicit_loss = lambda shift: polynomial(
        affine.set(translation=shift)(coordinates)
    ).sum()
    gradient = assert_jittable(jax.grad(composed_loss), affine.translation)
    assert np.allclose(gradient, jax.grad(explicit_loss)(affine.translation))
    assert np.any(np.abs(gradient) > 0)


def test_normalised_photometric_bins_retain_filter_and_qe_losses():
    bandpass = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    photometry = dlu.Photometry(dlu.Spectrum([1.0, 2.0, 3.0]), filter=0.5, qe=0.4)

    result = assert_jittable(lambda model: model(bandpass), photometry)
    # Equal bins of a linear density have the same averages as their centres.
    # Normalise the incident total, then retain both absolute response losses.
    expected = np.asarray([1.0, 2.0, 3.0]) / 6 * 0.5 * 0.4
    assert np.allclose(result.value, expected, atol=1e-7)
    assert np.allclose(result.value.sum(), 0.2)
    assert np.allclose(result.value.sum(), photometry.integrate(bandpass))
