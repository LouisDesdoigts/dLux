import jax.numpy as np
from jax import config, Array
import pytest
import dLux.utils as dlu

config.update("jax_debug_nans", True)


class TestZernike(object):
    """
    Tests the Zernike class.
    """


    def test_constructor(self, create_zernike):
        """
        Tests the constructor.
        """
        # Test inputs <= 0
        with pytest.raises(ValueError):
            create_zernike(0)

        # Test constructor
        zernike = create_zernike()


    def test_calculate(self, create_zernike):
        """
        Tests the calculate method.
        """
        zernike = create_zernike()
        coords = dlu.pixel_coords(16, 1/16)

        # Test calculate
        z = zernike.calculate(coords)
        p = zernike.calculate(coords, nsides=3)

        with pytest.raises(ValueError):
            zernike.calculate(coords, nsides=1)


class TestZernikeBasis(object):
    """
    Tests the ZernikeBasis class.
    """


    def test_constructor(self, create_zernike_basis):
        zernike_basis = create_zernike_basis()


    def test_calculate_basis(self, create_zernike_basis):
        zernike_basis = create_zernike_basis()
        coords = dlu.pixel_coords(16, 1/16)

        # Test calculate
        z = zernike_basis.calculate_basis(coords)