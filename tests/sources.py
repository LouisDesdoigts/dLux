from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


Array = np.ndarray


class SourceUtility(Utility):
    """
    Utility for the Source class.
    """
    position : Array
    flux     : Array
    spectrum : dLux.spectrums.Spectrum
    name     : str
    dLux.sources.Source.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Source Utility.
        """
        self.position = np.array([0., 0.])
        self.flux     = np.array(1.)
        self.spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10))
        self.name = "Source"


    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.Source(position, flux, spectrum, name=name)


class ResolvedSourceUtility(SourceUtility):
    """
    Utility for the ResolvedSource class.
    """
    dLux.sources.ResolvedSource.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ResolvedSource Utility.
        """
        pass


    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.Source(position, flux, spectrum, name=name)


class RelativeFluxSourceUtility(SourceUtility):
    """
    Utility for the RelativeFluxSource class.
    """
    contrast : Array
    dLux.sources.RelativeFluxSource.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the RelativeFluxSource Utility.
        """
        super().__init__()
        self.contrast = np.array(2.)


    def construct(self       : Utility,
                  position   : Array    = None,
                  flux       : Array    = None,
                  spectrum   : Spectrum = None,
                  contrast   : Array    = None,
                  name       : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        contrast = self.contrast if contrast is None else contrast
        name     = self.name     if name     is None else name
        return dLux.sources.RelativeFluxSource(contrast, position=position,
                                               flux=flux, spectrum=spectrum,
                                               name=name)


class RelativePositionSourceUtility(SourceUtility):
    """
    Utility for the RelativePositionSource class.
    """
    separation : Array
    position_angle : Array
    dLux.sources.RelativePositionSource.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the RelativePositionSource Utility.
        """
        super().__init__()
        self.separation  = np.array(1.)
        self.position_angle = np.array(0.)


    def construct(self           : Utility,
                  position       : Array    = None,
                  flux           : Array    = None,
                  spectrum       : Spectrum = None,
                  separation     : Array    = None,
                  position_angle : Array    = None,
                  name           : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position       = self.position       if position       is None \
                                                            else position
        flux           = self.flux           if flux           is None \
                                                            else flux
        spectrum       = self.spectrum       if spectrum       is None \
                                                            else spectrum
        separation     = self.separation     if separation     is None \
                                                            else separation
        position_angle = self.position_angle if position_angle is None \
                                                            else position_angle
        name           = self.name           if name           is None \
                                                            else name
        return dLux.sources.RelativePositionSource(separation, position_angle,
                                                   position=position, flux=flux,
                                                   spectrum=spectrum, name=name)


class PointSourceUtility(SourceUtility):
    """
    Utility for the PointSource class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointSource Utility.
        """
        super().__init__()


    def construct(self        : Utility,
                  position    : Array    = None,
                  flux        : Array    = None,
                  spectrum    : Spectrum = None,
                  name        : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.PointSource(position, flux, spectrum, name=name)


class MultiPointSourceUtility(SourceUtility):
    """
    Utility for the MultiPointSource class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the MultiPointSource Utility.
        """
        super().__init__()
        self.position = np.zeros((3, 2))
        self.flux     = np.ones(3)
        self.spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10))
        self.name = "Source"


    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.MultiPointSource(position, flux, spectrum, name=name)


class ArrayDistributionUtility(SourceUtility):
    """
    Utility for the ArrayDistribution class.
    """
    distribution : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ArrayDistribution Utility.
        """
        super().__init__()
        distribution = np.ones((5, 5))
        self.distribution = distribution/distribution.sum()


    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.ArrayDistribution(position, flux, distribution,
                                              spectrum, name=name)


class BinarySourceUtility(RelativePositionSourceUtility, \
                          RelativeFluxSourceUtility):
    """
    Utility for the BinarySource class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the BinarySource Utility.
        """
        super().__init__()
        wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        weights     = np.tile(np.arange(10), (2, 1))
        self.spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)


    def construct(self           : Utility,
                  position       : Array    = None,
                  flux           : Array    = None,
                  spectrum       : Spectrum = None,
                  separation     : Array    = None,
                  position_angle : Array    = None,
                  contrast       : Array    = None,
                  name           : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position_angle = self.position_angle if position_angle is None \
                                                    else position_angle
        position   = self.position   if position   is None else position
        flux       = self.flux       if flux       is None else flux
        spectrum   = self.spectrum   if spectrum   is None else spectrum
        separation = self.separation if separation is None else separation
        contrast   = self.contrast   if contrast   is None else contrast
        name       = self.name       if name       is None else name
        return dLux.sources.BinarySource(position, flux, separation, \
                                  position_angle, contrast, spectrum, name=name)


class PointExtendedSourceUtility(RelativeFluxSourceUtility, \
                                 ArrayDistributionUtility):
    """
    Utility for the PointExtendedSource class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointExtendedSource Utility.
        """
        super().__init__()


    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  contrast     : Array    = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        contrast     = self.contrast     if contrast     is None else contrast
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.PointExtendedSource(position, flux, distribution,
                                                contrast, spectrum, name=name)


class PointAndExtendedSourceUtility(RelativeFluxSourceUtility, \
                                    ArrayDistributionUtility):
    """
    Utility for the PointAndExtendedSource class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointAndExtendedSource Utility.
        """
        super().__init__()
        wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        weights = np.tile(np.arange(10), (2, 1))
        self.spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)


    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  contrast     : Array    = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        contrast     = self.contrast     if contrast     is None else contrast
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.PointAndExtendedSource(position, flux, distribution,
                                                contrast, spectrum, name=name)


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


    def test_get_flux(self : UtilityUser) -> None:
        """
        Tests the get_flux method.
        """
        source = self.utility.construct()
        flux_out = source.get_flux()
        assert flux_out.shape == (2,)
        assert np.allclose(flux_out[0]/flux_out[1], source.contrast)


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
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


class TestMultiPointSource(UtilityUser):
    """
    Tests the MultiPointSource class.
    """
    utility : MultiPointSourceUtility = MultiPointSourceUtility()


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

        # Test 1 dim input
        with pytest.raises(AssertionError):
            self.utility.construct(position=np.ones(2))

        # Flux
        # Test string inputs
        with pytest.raises(ValueError):
            self.utility.construct(flux="")

        # Test zero length input
        with pytest.raises(AssertionError):
            self.utility.construct(flux=[])

        # Test 2 dim input
        with pytest.raises(AssertionError):
            self.utility.construct(flux=np.ones((2, 2)))

        # Spectrum
        # Test non-spectrum input
        with pytest.raises(AssertionError):
            self.utility.construct(spectrum=[])

        # Name
        # Test non-string input
        with pytest.raises(AssertionError):
            self.utility.construct(name=[])


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


    def test_get_distribution(self : UtilityUser) -> None:
        """
        Tests the get_distribution method.
        """
        source = self.utility.construct()
        assert (source.get_distribution() == source.distribution).all()


    def test_normalise(self : UtilityUser) -> None:
        """
        Tests the normalise method.
        """
        source = self.utility.construct()
        new_distribution = np.ones((4, 4))
        new_source = source.set('distribution', new_distribution).normalise()
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
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


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
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


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