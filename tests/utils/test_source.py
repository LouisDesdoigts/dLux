import pytest
import jax.numpy as np
from dLux.utils.source import fluxes_from_contrast, positions_from_sep


@pytest.fixture
def mean_flux():
    return 1.0


@pytest.fixture
def contrast():
    return 0.5


@pytest.fixture
def position():
    return np.array([0.0, 0.0])


@pytest.fixture
def separation():
    return 1.0


@pytest.fixture
def position_angle():
    return np.pi / 4


def test_fluxes_from_contrast(mean_flux, contrast):
    result = fluxes_from_contrast(mean_flux, contrast)
    expected = 2 * np.array([contrast * mean_flux, mean_flux]) / (1 + contrast)
    assert np.allclose(result, expected)


def test_positions_from_sep(position, separation, position_angle):
    result = positions_from_sep(position, separation, position_angle)
    r, phi = separation / 2, position_angle
    sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])
    expected = np.array([position + sep_vec, position - sep_vec])
    assert np.allclose(result, expected)
