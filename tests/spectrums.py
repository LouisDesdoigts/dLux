from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


class SpectrumUtility(Utility):
    """
    Utility for the Spectrum class.
    """
    wavelengths : Array
    dLux.spectrums.Spectrum.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the Spectrum Utility.
        """
        self.wavelengths = np.linspace(500e-9, 600e-9, 10)


    def construct(self : Utility, wavelengths : Array = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        return dLux.spectrums.Spectrum(wavelengths)


class ArraySpectrumUtility(SpectrumUtility):
    """
    Utility for the ArraySpectrum class.
    """
    weights : Array


    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the ArraySpectrum Utility.
        """
        super().__init__()
        self.weights = np.arange(10)


    def construct(self        : Utility,
                  wavelengths : Array = None,
                  weights     : Array = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        weights = self.weights if weights is None else weights
        return dLux.spectrums.ArraySpectrum(wavelengths, weights)


class PolynomialSpectrumUtility(SpectrumUtility):
    """
    Utility for the PolynomialSpectrum class.
    """
    coefficients : Array


    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the PolynomialSpectrum Utility.
        """
        super().__init__()
        self.coefficients = np.arange(3)


    def construct(self         : Utility,
                  wavelengths  : Utility = None,
                  coefficients : Utility = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        coefficients = self.coefficients if coefficients is None \
                                                            else coefficients
        return dLux.spectrums.PolynomialSpectrum(wavelengths, coefficients)


class CombinedSpectrumUtility(SpectrumUtility):
    """
    Utility for the ArraySpectrum class.
    """
    wavelengths : Array
    weights     : Array


    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the ArraySpectrum Utility.
        """
        super()
        self.wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        self.weights = np.tile(np.arange(10), (2, 1))


    def construct(self        : Utility,
                  wavelengths : Utility = None,
                  weights     : Utility = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        weights = self.weights if weights is None else weights
        return dLux.spectrums.CombinedSpectrum(wavelengths, weights)


class TestSpectrum(UtilityUser):
    """
    Tests the Spectrum class.
    """
    utility : SpectrumUtility = SpectrumUtility()


    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct("")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct([])


    def test_get_wavelengths(self : UtilityUser) -> None:
        """
        Tests the get_wavelength method.
        """
        spectrum = self.utility.construct()
        assert (spectrum.get_wavelengths() == spectrum.wavelengths).all()


class TestArraySpectrum(UtilityUser):
    """
    Tests the ArraySpectrum class.
    """
    utility : ArraySpectrumUtility = ArraySpectrumUtility()

    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(weights="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(weights=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[])


    def test_get_weights(self : UtilityUser) -> None:
        """
        Tests the get_weights method.
        """
        spectrum = self.utility.construct()
        assert (spectrum.get_weights() == spectrum.weights).all()


    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.arange(10)
        new_spectrum = self.utility.construct().set('weights',
                                                    new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(), 1.)


class TestPolynomialSpectrum(UtilityUser):
    """
    Tests the PolynomialSpectrum class.

    Note this does not test the .normalise() method becuase it does not
    normalise the coefficients, instead the .get_weights() returns a normalised
    weights.
    """
    utility : PolynomialSpectrumUtility = PolynomialSpectrumUtility()


    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(coefficients="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=[])

        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=[np.nan])

        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(coefficients=[np.inf])


    def test_get_weights(self : UtilityUser) -> None:
        """
        Tests the normalisation of the get_weights method.
        """
        assert np.allclose(self.utility.construct().get_weights().sum(), 1.)


class TestCombinedSpectrum(UtilityUser):
    """
    Tests the CombinedSpectrum class
    """
    utility : CombinedSpectrumUtility = CombinedSpectrumUtility()


    def test_constructor(self : UtilityUser) -> None:
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


    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.tile(np.arange(10), (2, 1))
        new_spectrum = self.utility.construct().set('weights',
                                                    new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(1), 1.)