from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config
# config.update("jax_debug_nans", True)

class TestArraySpectrum():
    """
    Tests the ArraySpectrum class.
    """

    def test_constructor(self, create_array_spectrum : callable) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            create_array_spectrum(weights="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_array_spectrum(weights=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_array_spectrum(weights=[])


    def test_get_weights(self, create_array_spectrum : callable) -> None:
        """
        Tests the get_weights method.
        """
        spectrum = create_array_spectrum()
        assert (spectrum.get_weights() == spectrum.weights).all()


    def test_normalise(self, create_array_spectrum : callable) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.arange(10)
        new_spectrum = create_array_spectrum().set('weights',
                                                    new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(), 1.)



class TestPolynomialSpectrum():
    """
    Tests the PolynomialSpectrum class.

    Note this does not test the .normalise() method becuase it does not
    normalise the coefficients, instead the .get_weights() returns a normalised
    weights.
    """


    def test_constructor(self, create_polynomial_spectrum : callable) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            create_polynomial_spectrum(coefficients="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_polynomial_spectrum(coefficients=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_polynomial_spectrum(coefficients=[])

        # Test nan inputs
        with pytest.raises(AssertionError):
            create_polynomial_spectrum(coefficients=[np.nan])

        # Test infinite inputs
        with pytest.raises(AssertionError):
            create_polynomial_spectrum(coefficients=[np.inf])

        create_polynomial_spectrum()

    def test_get_weights(self, create_polynomial_spectrum : callable) -> None:
        """
        Tests the normalisation of the get_weights method.
        """
        assert np.allclose(create_polynomial_spectrum().get_weights().sum(), 1.)


class TestCombinedSpectrum():
    """
    Tests the CombinedSpectrum class
    """


    def test_constructor(self, create_combined_spectrum : callable) -> None:
        """
        Test the constructor.
        """
        # Wavelengths Testing
        # Test string inputs
        with pytest.raises(ValueError):
            create_combined_spectrum(wavelengths="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_combined_spectrum(wavelengths=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_combined_spectrum(wavelengths=[])

        # Weights Testing
        # Test string inputs
        with pytest.raises(ValueError):
            create_combined_spectrum(weights="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_combined_spectrum(weights=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_combined_spectrum(weights=[])


    def test_normalise(self, create_combined_spectrum : callable) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.tile(np.arange(10), (2, 1))
        new_spectrum = create_combined_spectrum().set('weights',
                                                    new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(1), 1.)
