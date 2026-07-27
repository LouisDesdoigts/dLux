"""Tests for dLux.utils.apertures."""

import jax.numpy as np
import pytest

import dLux.utils as dlu


def test_segment_centres():
    assert dlu.segmented_hex_cens(1, 0.5).shape == (1, 2)
    assert dlu.segmented_hex_cens(3, 0.5).shape == (19, 2)
    with pytest.raises(ValueError):
        dlu.segmented_hex_cens(0, 0.5)


def test_non_redundant_support():
    apertures = np.asarray([[[1.0, 0.0]], [[0.5, 1.0]]])
    support = dlu.non_redundant_support(apertures)
    assert support.shape == apertures.shape
    assert np.all(support.sum(axis=0) <= 1)
    assert support[0, 0, 0]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: dlu.circular_aperture(16, 1.0, oversample=1),
        lambda: dlu.segmented_aperture(
            16, 2.0, nrings=2, segment_diameter=0.5, oversample=1
        ),
        lambda: dlu.sparse_aperture(16, 2.0, ((-0.4, 0), (0.4, 0)), 0.4, oversample=1),
        lambda: dlu.hst_like(16, oversample=1),
        lambda: dlu.jwst_like(16, oversample=1),
        lambda: dlu.euclid_like(16, oversample=1),
    ],
)
def test_aperture_factory_contract(factory):
    aperture = factory()
    assert aperture.shape == (16, 16)
    assert np.all(np.isfinite(aperture))


def test_aperture_basis_contract():
    transmission, basis, support = dlu.circular_aperture(
        16, 1.0, oversample=1, zernike_nolls=(1, 4), return_support=True
    )
    assert transmission.shape == support.shape == (16, 16)
    assert basis.shape == (2, 16, 16)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: dlu.segmented_aperture(
            16,
            2.0,
            nrings=2,
            segment_diameter=0.5,
            oversample=1,
            zernike_nolls=(1, 4),
            return_support=True,
        ),
        lambda: dlu.sparse_aperture(
            16,
            2.0,
            ((-0.4, 0), (0.4, 0)),
            0.4,
            oversample=1,
            zernike_nolls=(1, 4),
            return_support=True,
        ),
    ],
)
def test_vectorised_aperture_basis(factory):
    transmission, basis, support = factory()

    assert transmission.shape == (16, 16)
    assert basis.shape[-3:] == (2, 16, 16)
    assert support.shape[-2:] == (16, 16)


def test_spider_validation():
    with pytest.raises(ValueError):
        dlu.circular_aperture(8, 1, spider_width=0.1)
