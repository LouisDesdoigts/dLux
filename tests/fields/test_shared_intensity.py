"""Shared sampled-intensity operations preserve each concrete result type."""

import jax.numpy as np
import pytest

import dLux as dl


@pytest.mark.parametrize("kind", (dl.Intensity, dl.PSF, dl.Image))
def test_intensity_operations_preserve_type_sampling_and_total(kind):
    values = np.arange(1.0, 33.0).reshape(2, 4, 4)
    grid = dl.Grid(4, 0.2)
    field = kind(values, grid)

    normalised = field.normalise(value=3.0)
    assert type(normalised) is kind
    assert np.allclose(normalised.data.sum(), 3.0)
    assert np.array_equal(normalised.intensity, normalised.data)

    binned = normalised.downsample(2)
    assert type(binned) is kind
    assert binned.data.shape == (2, 2, 2)
    assert np.allclose(binned.data.sum(), 3.0)
    assert np.allclose(binned.grid.d, 0.4)
    assert np.array_equal(field.data, values)


def test_response_metadata_and_image_conversion_survive_shared_operations():
    psf = dl.PSF(
        np.ones((2, 4, 4)), dl.Grid(4, 0.1), wavelengths=np.array([500e-9, 600e-9])
    )
    reduced = psf.downsample(2).normalise()
    assert np.array_equal(reduced.wavelengths, psf.wavelengths)
    image = reduced.to_intensity(10).to_image()
    assert type(image) is dl.Image
    assert np.isclose(image.data.sum(), 10)
