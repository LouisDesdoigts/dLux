import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)
from dLux import utils
from dLux.utils.zernikes import (
    zernike_name,
    noll_indices,
    zernike_factors,
    zernike,
    zernike_fast,
    zernike_basis,
    polike,
    polike_fast,
    polike_basis,
)


# NOTE: 'Correctness' is non-trivial in this module, so most of these tests are
# high-level run checks for shapes and nans.
def test_zernike_name():
    assert zernike_name(1) == "Piston"
    assert zernike_name(2) == "Tilt X"
    assert zernike_name(3) == "Tilt Y"
    assert zernike_name(4) == "Defocus"
    assert zernike_name(5) == "Astig X"
    assert zernike_name(6) == "Astig Y"
    assert zernike_name(7) == "Coma X"
    assert zernike_name(8) == "Coma Y"
    assert zernike_name(9) == "Trefoil X"
    assert zernike_name(10) == "Trefoil Y"


def test_noll_indices():
    assert noll_indices(1) == (0, 0)
    assert noll_indices(2) == (1, 1)
    assert noll_indices(3) == (1, -1)
    assert noll_indices(4) == (2, 0)
    assert noll_indices(6) == (2, 2)
    assert noll_indices(5) == (2, -2)
    assert noll_indices(7) == (3, -1)
    assert noll_indices(8) == (3, 1)
    assert noll_indices(9) == (3, -3)
    assert noll_indices(10) == (3, 3)


def test_zernike():
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(zernike(5, coordinates)).any()


def test_zernike_basis():
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(zernike_basis((1, 2, 3, 4, 5), coordinates)).any()


def test_polike():
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(polike(6, 5, coordinates)).any()
    with pytest.raises(ValueError):
        polike(2, 5, coordinates)


def test_polike_basis():
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(polike_basis(6, (1, 2, 3, 4, 5), coordinates)).any()


def test_zernike_factors():
    with pytest.raises(ValueError):
        zernike_factors(0)


def test_zernike_fast():
    j = 5
    inds = noll_indices(j)
    factors = zernike_factors(j)
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(zernike_fast(*inds, *factors, coordinates)).any()


def test_polike_fast():
    j = 5
    inds = noll_indices(j)
    factors = zernike_factors(j)
    coordinates = utils.pixel_coords(16, 2)
    assert not np.isnan(polike_fast(6, *inds, *factors, coordinates)).any()
    with pytest.raises(ValueError):
        polike_fast(2, *inds, *factors, coordinates)
