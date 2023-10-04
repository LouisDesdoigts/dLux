import pytest
import dLux.utils as dlu
from dLux.layers import (
    Zernike,
    ZernikeBasis,
)
from jax import config

config.update("jax_debug_nans", True)


@pytest.fixture
def coords():
    return dlu.pixel_coords(16, 2)


@pytest.fixture
def js():
    return [1, 2, 3, 4]


@pytest.mark.parametrize("nsides", [0, 3, 6])
@pytest.mark.parametrize("j", [1, 2, 3, 4])
def test_zernikes(j, nsides, coords):
    zernike = Zernike(j)
    assert zernike.calculate(coords, nsides).shape == coords.shape[1:]
    with pytest.raises(ValueError):
        Zernike(0)


@pytest.mark.parametrize("nsides", [0, 3, 6])
def test_zernike_basis(js, nsides, coords):
    zernike = ZernikeBasis(js)
    assert (
        zernike.calculate_basis(coords, nsides).shape
        == (len(js),) + coords.shape[1:]
    )
