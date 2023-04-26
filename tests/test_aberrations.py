from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


class TestZernike(object):
    """
    Tests the Zernike class.
    """


    def test_constructor(self, create_zernike : callable):
        """
        Tests the constructor.
        """
        # Test inputs <= 0
        with pytest.raises(ValueError):
            create_zernike(0)

        # Test constructor
        zernike = create_zernike()


    def test_calculate(self, create_zernike : callable):
        """
        Tests the calculate method.
        """
        zernike = create_zernike()
        coords = dLux.utils.get_pixel_positions((16, 16), (1/16, 1/16))

        # Test calcualte
        z = zernike.calculate(coords)
        p = zernike.calculate(coords, nsides=3)

        with pytest.raises(ValueError):
            zernike.calculate(coords, nsides=1)


class TestZernikeBasis(object):
    """
    Tests the ZernikeBasis class.
    """


    def test_constructor(self, create_zernike_basis : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        zernike_basis = create_zernike_basis()


    def test_calculate_basis(self, create_zernike_basis : callable):
        """
        Tests the calculate_basis method.
        """
        zernike_basis = create_zernike_basis()
        coords = dLux.utils.get_pixel_positions((16, 16), (1/16, 1/16))

        # Test calcualte
        z = zernike_basis.calculate_basis(coords)


class TestAberrationFactory(object):
    """
    Tests the AberrationFactory class.
    """


    def test_constructor(self, create_aberration_factory : callable):
        """
        Tests the constructor.
        """
        # Test inputs
        with pytest.raises(ValueError):
            create_aberration_factory(nsides=2)

        # Test circular constructor
        aberration_factory = create_aberration_factory()
        assert isinstance(aberration_factory, dLux.optics.ApplyBasisOPD)

        # Test polygonal constructor
        aberration_factory = create_aberration_factory(nsides=3)
        assert isinstance(aberration_factory, dLux.optics.ApplyBasisOPD)