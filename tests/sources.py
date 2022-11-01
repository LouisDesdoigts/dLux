import pytest
import jax.numpy as np
from utilities import *


class TestSource(UtilityUser):
    """
    Tests the Source class.
    """
    utility : SourceUtility = SourceUtility()
    
    
    def test_constructor(self : UtilityUser) -> None:
        """
        Test the constructor class.
        """
        # Position
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(position="")
        
        # Test zero dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(position=5.)
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(position=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(position=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(position=[np.inf])
        
        # Flux
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(flux="")
        
        # Test 1d dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(flux=[1.])
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(flux=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(flux=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(flux=[np.inf])
        
        # Spectrum
        # Test non-spectrum input
        with pytest.raises(AssertionError):
            self.utility.construct(spectrum=[])
        
        # Name
        # Test non-string input
        with pytest.raises(AssertionError):
            self.utility.construct(name=[])
    
    
    # Getters
    def test_get_position(self : UtilityUser) -> None:
        """
        Tests the get_position method.
        """
        source = self.utility.construct()
        assert (source.get_position() == source.position).all()
    
    
    def test_get_flux(self : UtilityUser) -> None:
        """
        Tests the get_flux method.
        """
        source = self.utility.construct()
        assert (source.get_flux() == source.flux).all()
    
    
    def test_get_wavelengths(self : UtilityUser) -> None:
        """
        Tests the get_wavelengths method.
        """
        source = self.utility.construct()
        assert (source.get_wavelengths() == source.spectrum.wavelengths).all()
    
    
    def test_get_weights(self : UtilityUser) -> None:
        """
        Tests the get_weights method.
        """
        source = self.utility.construct()
        assert (source.get_weights() == source.spectrum.weights).all()
    
    
    # Setters
    def test_set_position(self : UtilityUser) -> None:
        """
        Tests the set_position method.
        """
        new_position = np.array([1., 1.])
        new_source = self.utility.construct().set_position(new_position)
        assert (new_source.position == new_position).all()
    
    
    def test_set_flux(self : UtilityUser) -> None:
        """
        Tests the set_flux method.
        """
        new_flux = np.array(1.5)
        new_source = self.utility.construct().set_flux(new_flux)
        assert (new_source.flux == new_flux).all()
    
    
    def test_set_spectrum(self : UtilityUser) -> None:
        """
        Tests the set_spectrum method.
        """
        new_spectrum = dLux.spectrums.ArraySpectrum(np.linspace(600e-9, \
                                                                700e-9, 10))
        new_source = self.utility.construct().set_spectrum(new_spectrum)
        assert new_source.spectrum is new_spectrum


class TestResolvedSource(UtilityUser):
    """
    Tests the ResolvedSourve class.
    """
    utility : ResolvedSourceUtility = ResolvedSourceUtility()
    pass


class TestRelativeFluxSource(UtilityUser):
    """
    Tests the RelativeFluxSource class.
    """
    utility : RelativeFluxSourceUtility = RelativeFluxSourceUtility()
    
    
    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(contrast="")
        
        # Test one dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(contrast=[5.])
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(contrast=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(contrast=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(contrast=[np.inf])
    
    
    def test_get_contrast(self : UtilityUser) -> None:
        """
        Tests the get_contrast method.
        """
        source = self.utility.construct()
        assert (source.get_contrast() == source.contrast).all()
    
    
    def test_get_flux(self : UtilityUser) -> None:
        """
        Tests the get_flux method.
        """
        source = self.utility.construct()
        flux_out = source.get_flux()
        assert flux_out.shape == (2,)
        assert np.allclose(flux_out[0]/flux_out[1], source.contrast)
    
    
    def test_set_contrast(self : UtilityUser) -> None:
        """
        Tests the set_contrast method.
        """
        new_contrast = np.array(1.5)
        new_source = self.utility.construct().set_contrast(new_contrast)
        assert (new_source.contrast == new_contrast).all()


class TestRelativePositionSource(UtilityUser):
    """
    Tests the RelativePositionSource class.
    """
    utility : RelativePositionSourceUtility = RelativePositionSourceUtility()
    
    
    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Separation
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(separation="")
        
        # Test one dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(separation=[5.])
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(separation=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(separation=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(separation=[np.inf])
        
        # position_angle
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(position_angle="")
        
        # Test one dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(position_angle=[5.])
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(position_angle=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(position_angle=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(position_angle=[np.inf])
    
    
    def test_get_separation(self : UtilityUser) -> None:
        """
        Tests the get_separation method.
        """
        source = self.utility.construct()
        assert (source.get_separation() == source.separation).all()
    
    
    def test_get_position_angle(self : UtilityUser) -> None:
        """
        Tests the get_position_angle method.
        """
        source = self.utility.construct()
        assert (source.get_position_angle() == source.position_angle).all()
    
    
    def test_get_position(self : UtilityUser) -> None:
        """
        Tests the get_position method.
        """
        source = self.utility.construct()
        position_out = source.get_position()
        sep_vec = position_out[0] - position_out[1]
        separation = np.hypot(sep_vec[0], sep_vec[1])
        position_angle = np.arctan2(sep_vec[0], sep_vec[1])
        assert position_out.shape == (2,2)
        assert np.allclose(source.separation, separation).all()
        assert np.allclose(source.position_angle, position_angle).all()
    
    
    def test_set_separation(self : UtilityUser) -> None:
        """
        Tests the set_separation method.
        """
        source = self.utility.construct()
        new_separation = np.array(1.5)
        new_source = source.set_separation(new_separation)
        assert (new_source.separation == new_separation).all()
    
    
    def test_set_position_angle(self : UtilityUser) -> None:
        """
        Tests the set_position_angle method.
        """
        source = self.utility.construct()
        new_position_angle = np.array(np.pi)
        new_source = source.set_position_angle(new_position_angle)
        assert (new_source.position_angle == new_position_angle).all()


class TestPointSource(UtilityUser):
    """
    Tests the PointSource class.
    """
    utility : PointSourceUtility = PointSourceUtility()
    
    
    def test_model(self : UtilityUser) -> None:
        """
        Tests the model method.
        """
        source = self.utility.construct()
        optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        detector = dLux.core.Detector([dLux.AddConstant(0.)])
        filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        source.model(optics, detector, filter_in)
        source.model(optics, filter_in=filter_in)


class TestArrayDistribution(UtilityUser):
    """
    Tests the ArrayDistribution class.
    """
    utility : ArrayDistributionUtility = ArrayDistributionUtility()
    
    
    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(distribution="")
        
        # Test one dimension input
        with pytest.raises(AssertionError):
            self.utility.construct(distribution=[5.])
        
        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(distribution=[])
        
        # Test nan inputs
        with pytest.raises(AssertionError):
            self.utility.construct(distribution=[np.nan])
        
        # Test infinite inputs
        with pytest.raises(AssertionError):
            self.utility.construct(distribution=[np.inf])
    
    
    def test_get_distribution(self : UtilityUser) -> None:
        """
        Tests the get_distribution method.
        """
        source = self.utility.construct()
        assert (source.get_distribution() == source.distribution).all()
    
    
    def test_set_distribution(self : UtilityUser) -> None:
        """
        Tests the set_distribution method.
        """
        source = self.utility.construct()
        new_distribution = np.ones((4, 4))
        new_distribution /= new_distribution.sum()
        new_source = source.set_distribution(new_distribution)
        assert (new_source.distribution == new_distribution).all()
    
    
    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        source = self.utility.construct()
        new_distribution = np.ones((4, 4))
        new_source = source.set_distribution(new_distribution).normalise()
        assert np.allclose(new_source.distribution.sum(), 1.)
    
    
    def test_model(self : UtilityUser) -> None:
        """
        Tests the model method.
        """
        source = self.utility.construct()
        optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        detector = dLux.core.Detector([dLux.AddConstant(0.)])
        filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        source.model(optics, detector, filter_in)
        source.model(optics, filter_in=filter_in)


class TestBinarySource(UtilityUser):
    """
    Tests the BinarySource class.
    """
    utility : BinarySourceUtility = BinarySourceUtility()
    
    
    def test_model(self : UtilityUser) -> None:
        """
        Tests the model method.
        """
        source = self.utility.construct()
        optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        detector = dLux.core.Detector([dLux.AddConstant(0.)])
        filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


class TestPointExtendedSource(UtilityUser):
    """
    Tests the PointExtendedSource class.
    """
    utility : PointExtendedSourceUtility = PointExtendedSourceUtility()
    
    
    def test_model(self : UtilityUser) -> None:
        """
        Tests the model method.
        """
        source = self.utility.construct()
        optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        detector = dLux.core.Detector([dLux.AddConstant(0.)])
        filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        source.model(optics, detector, filter_in)
        source.model(optics, filter_in=filter_in)


class TestPointAndExtendedSource(UtilityUser):
    """
    Tests the PointAndExtendedSource class.
    """
    utility : PointAndExtendedSourceUtility = PointAndExtendedSourceUtility()
    
    
    def test_constructor(self : UtilityUser) -> None:
        """
        Tests the constructor.
        """
        # Test non Combined Spectrum input
        with pytest.raises(AssertionError):
            spec = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, 600e-9, 10))
            self.utility.construct(spectrum=spec)
    
    
    def test_model(self : UtilityUser) -> None:
        """
        Tests the model method.
        """
        source = self.utility.construct()
        optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        detector = dLux.core.Detector([dLux.AddConstant(0.)])
        filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)