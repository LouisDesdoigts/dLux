from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


# class SpectrumUtility(Utility):
#     """
#     Utility for the Spectrum class.
#     """
#     wavelengths : Array
#     dLux.spectrums.Spectrum.__abstractmethods__ = ()


#     def __init__(self : Utility) -> Utility:
#         """
#         Constrcutor for the Spectrum Utility.
#         """
#         self.wavelengths = np.linspace(500e-9, 600e-9, 10)


#     def construct(self : Utility, wavelengths : Array = None) -> Spectrum:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         wavelengths = self.wavelengths if wavelengths is None else wavelengths
#         return dLux.spectrums.Spectrum(wavelengths)


# class ArraySpectrumUtility(SpectrumUtility):
#     """
#     Utility for the ArraySpectrum class.
#     """
#     weights : Array


#     def __init__(self : Utility) -> Utility:
#         """
#         Constrcutor for the ArraySpectrum Utility.
#         """
#         super().__init__()
#         self.weights = np.arange(10)


#     def construct(self        : Utility,
#                   wavelengths : Array = None,
#                   weights     : Array = None) -> Spectrum:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         wavelengths = self.wavelengths if wavelengths is None else wavelengths
#         weights = self.weights if weights is None else weights
#         return dLux.spectrums.ArraySpectrum(wavelengths, weights)


# class PolynomialSpectrumUtility(SpectrumUtility):
#     """
#     Utility for the PolynomialSpectrum class.
#     """
#     coefficients : Array


#     def __init__(self : Utility) -> Utility:
#         """
#         Constrcutor for the PolynomialSpectrum Utility.
#         """
#         super().__init__()
#         self.coefficients = np.arange(3)


#     def construct(self         : Utility,
#                   wavelengths  : Utility = None,
#                   coefficients : Utility = None) -> Spectrum:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         wavelengths = self.wavelengths if wavelengths is None else wavelengths
#         coefficients = self.coefficients if coefficients is None \
#                                                             else coefficients
#         return dLux.spectrums.PolynomialSpectrum(wavelengths, coefficients)


# class CombinedSpectrumUtility(SpectrumUtility):
#     """
#     Utility for the ArraySpectrum class.
#     """
#     wavelengths : Array
#     weights     : Array


#     def __init__(self : Utility) -> Utility:
#         """
#         Constrcutor for the ArraySpectrum Utility.
#         """
#         super()
#         self.wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
#         self.weights = np.tile(np.arange(10), (2, 1))


#     def construct(self        : Utility,
#                   wavelengths : Utility = None,
#                   weights     : Utility = None) -> Spectrum:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         wavelengths = self.wavelengths if wavelengths is None else wavelengths
#         weights = self.weights if weights is None else weights
#         return dLux.spectrums.CombinedSpectrum(wavelengths, weights)


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


'''
class TestCombinedSpectrum():
    """
    Tests the CombinedSpectrum class
    """
    utility : CombinedSpectrumUtility = CombinedSpectrumUtility()


    def test_constructor(self) -> None:
        """
        Test the constructor.
        """
        # Wavelengths Testing
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(wavelengths="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(wavelengths=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(wavelengths=[])

        # Weights Testing
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(weights="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(weights=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[])


    def test_normalise(self) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.tile(np.arange(10), (2, 1))
        new_spectrum = self.utility.construct().set('weights',
                                                    new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(1), 1.)
        
'''