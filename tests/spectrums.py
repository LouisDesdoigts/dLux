import pytest
import jax.numpy as np
from utilities import *

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
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct([np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct([np.inf])
    
    
    def test_get_wavelengths(self : UtilityUser) -> None:
        """
        Tests the get_wavelength method.
        """
        spectrum = self.utility.construct()
        assert (spectrum.get_wavelengths() == spectrum.wavelengths).all()
    
    
    def test_set_wavelengths(self : UtilityUser) -> None:
        """
        Tests the set_wavelength method.
        """
        new_wavelengths = np.linspace(600e-9, 700e-9, 10)
        new_spectrum = self.utility.construct().set_wavelengths(new_wavelengths)
        assert (new_spectrum.wavelengths == new_wavelengths).all()
    
    
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
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[np.inf])
    
    
    def test_get_weights(self : UtilityUser) -> None:
        """
        Tests the get_weights method.
        """
        spectrum = self.utility.construct()
        assert (spectrum.get_weights() == spectrum.weights).all()
    
    
    def test_set_weights(self : UtilityUser) -> None:
        """
        Tests the set_weights method.
        """
        new_weights = np.arange(10)
        new_spectrum = self.utility.construct().set_weights(new_weights)
        assert (new_spectrum.weights == new_weights).all()
    
    
    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.arange(10)
        new_spectrum = self.utility.construct().\
                                        set_weights(new_weights).normalise()
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
    
    
    def test_get_coefficients(self : UtilityUser) -> None:
        """
        Tests the get_coefficents method.
        """
        spectrum = self.utility.construct()
        assert (spectrum.get_coefficients() == spectrum.coefficients).all()
    
    
    def test_set_coefficients(self : UtilityUser) -> None:
        """
        Tests the set_coefficents method.
        """
        new_coefficients = np.arange(10)
        new_spectrum = self.utility.construct(). \
                                            set_coefficients(new_coefficients)
        assert (new_spectrum.coefficients == new_coefficients).all()
    
    
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
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(wavelengths=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(wavelengths=[np.inf])
        
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
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(weights=[np.inf])
    
    
    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        new_weights = np.tile(np.arange(10), (2, 1))
        new_spectrum = self.utility.construct().set_weights(new_weights).normalise()
        assert np.allclose(new_spectrum.weights.sum(1), 1.)