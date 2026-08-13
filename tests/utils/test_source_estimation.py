"""Tests for point-source initialisation."""

import jax.numpy as np
import jax.random as jr
import pytest

import dLux.utils as dlu


def gaussian_scene(background=0.0):
    """Return an analytic two-source image and its centred source parameters."""
    y, x = np.mgrid[-16:17, -16:17]
    positions = np.asarray(((-5.25, 3.5), (4.4, -3.2)))
    flux = np.asarray((12.0, 5.0))
    sigma = 1.2

    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf = psf / psf.sum()
    models = [
        np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma**2))
        for px, py in positions
    ]
    models = np.stack(models)
    models = models / models.sum((-2, -1), keepdims=True)
    data = background + np.sum(flux[:, None, None] * models, axis=0)
    return data, psf, positions, flux


def test_estimate_sources_recovers_catalogue():
    data, psf, positions, flux = gaussian_scene(background=2.0)
    result = dlu.estimate_sources(data, psf, 2, background=2.0)

    order = np.argsort(result["position"][:, 0])
    assert np.allclose(result["position"][order], positions, atol=0.15)
    assert np.allclose(result["flux"][order], flux, rtol=0.03)
    assert result["residual"].shape == (2, *data.shape)
    assert np.all(np.diff(result["loss"]) < 0)


def test_estimate_sources_handles_constant_background():
    data, psf, _, _ = gaussian_scene(background=0.0)
    baseline = dlu.estimate_sources(data, psf, 2, background=0.0)
    shifted = dlu.estimate_sources(data + 20.0, psf, 2, background=20.0)
    estimated = dlu.estimate_sources(data + 20.0, psf, 2)

    assert np.allclose(shifted["position"], baseline["position"])
    assert np.allclose(shifted["flux"], baseline["flux"])
    assert np.isclose(estimated["background"], 20.0, atol=1e-3)


def test_estimate_sources_uses_detection_signal_to_noise():
    y, x = np.mgrid[-8:9, -8:9]
    psf = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
    psf = psf / psf.sum()
    data = 6 * psf + 0.2 * jr.normal(jr.key(73), psf.shape)

    result = dlu.estimate_sources(data, psf, 1, background=0.0)

    assert np.all(np.abs(result["position"][0]) < 1)


def test_estimate_sources_rejects_exhausted_detection_area():
    data, psf, _, _ = gaussian_scene()

    with pytest.raises(ValueError, match="No finite detection candidates"):
        dlu.estimate_sources(
            data,
            psf,
            2,
            background=0.0,
            detection_separation=100,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_source": 0}, "positive integer"),
        ({"n_source": 1, "samples": 4}, "odd integer"),
        ({"n_source": 1, "radius": 0}, "greater than zero"),
        (
            {"n_source": 1, "detection_separation": 0},
            "detection_separation",
        ),
    ],
)
def test_estimate_sources_validates_search(kwargs, message):
    data = np.ones((5, 5))
    with pytest.raises(ValueError, match=message):
        dlu.estimate_sources(data, data, **kwargs)
