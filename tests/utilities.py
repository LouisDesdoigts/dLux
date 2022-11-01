from __future__ import annotations
import jax.numpy as np
import dLux
import typing
import abc

__author__ = "Jordan Dennis"
__date__ = "28/06/2022"

Array = typing.NewType("Array", np.ndarray)
PlaneType = typing.NewType("PlaneType", dLux.wavefronts.PlaneType)
Wavefront = typing.NewType("Wavefront", dLux.wavefronts.Wavefront)
Propagator = typing.NewType("Propagator", dLux.propagators.Propagator)


class UtilityUser():
    """
    The base utility class. These utility classes are designed to 
    define safe constructors and constants for testing. These   
    classes are for testing purposes only. 
    """
    utility : Utility


    def get_utility(self : UtilityUser) -> Utility:
        """
        Accessor for the utility. 

        Returns 
        -------
        utility : Utility
            The utility
        """
        return self.utility


class Utility():
    """
    """
    def __init__(self : Utility) -> Utility:
        """
        Construct a new Utility.

        Returns
        : Utility 
            The utility. 
        """
        pass


    def construct(self : Utility) -> object:
        """
        Safe constructor for the dLuxModule, associated with 
        this utility.

        Returns
        -------
        : dLuxModule
            A safe dLuxModule for testing.
        """
        pass


    def approx(self : Utility, 
               result : Array, 
               comparator : Array) -> Array:
        """
        Compare two arrays to within floating point precision.

        Parameters
        ----------
        result : Array
            The result that you want to test for nearness to
            comparator.
        comparator : Array
            The comparison array.

        Returns
        -------
        : Array[bool]
            True if the array elements are similar to float 
            error, False otherwise. 
        """
        lower_bound = (result - 0.0005) <= comparator
        upper_bound = (result + 0.0005) >= comparator
        return lower_bound & upper_bound 


######################
### Base Utilities ###
######################
class BaseUtility(Utility):
    """
    Utility for the Base class.
    """
    param1 : float
    param2 : float
    
    
    class A(dLux.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float
        b     : B
        
        
        def __init__(self, param, b):
            """
            Constructor for the Base testing class
            """
            self.param = param
            self.b = b
        
        
        def model(self):
            """
            Sample modelling function
            """
            return self.param**2 + self.b.param**2
    
    
    class B(dLux.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float
        
        
        def __init__(self, param):
            """
            Constructor for the Base testing class
            """
            self.param = param
    
    
    def __init__(self : Utility):
        """
        Constructor for the Optics Utility.
        """ 
        self.param1 = 1.
        self.param2 = 1.
    
    
    def construct(self : Utility, 
                  param1 : float = None, 
                  param2 : float = None):
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        param1 = self.param1 if param1 is None else param1
        param2 = self.param2 if param2 is None else param2
        return self.A(param1, self.B(param2))


class ExtendedBaseUtility(BaseUtility):
    """
    Utility for the Base class.
    """
    pass



######################
### Core Utilities ###
######################
class OpticsUtility(Utility):
    """
    Utility for the Optics class.
    """
    layers : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Optics Utility.
        """    
        self.layers = [
            dLux.optics.CreateWavefront(16, 1),
            dLux.optics.CompoundAperture([0.5]),
            dLux.optics.NormaliseWavefront(),
            dLux.propagators.CartesianMFT(16, 1., 1e-6)
        ]
    
    
    def construct(self : Utility, layers : list = None) -> Optics:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        layers = self.layers if layers is None else layers
        return dLux.core.Optics(layers)


class DetectorUtility(Utility):
    """
    Utility for the Detector class.
    """
    layers : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Detector Utility.
        """    
        self.layers = [
            dLux.detectors.AddConstant(1.)
        ]
    
    
    def construct(self : Utility, layers : list = None) -> Detector:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        layers = self.layers if layers is None else layers
        return dLux.core.Detector(layers)


class SceneUtility(Utility):
    """
    Utility for the Scene class.
    """
    sources : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Scene Utility.
        """
        self.sources = [
            PointSourceUtility().construct()
        ]
    
    
    def construct(self : Utility, sources : list = None) -> Scene:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        sources = self.sources if sources is None else sources
        return dLux.core.Scene(sources)


class FilterUtility(Utility):
    """
    Utility for the Filter class.
    """
    wavelengths : Array
    throughput  : Array
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Filter Utility.
        """
        self.wavelengths = np.linspace(1e-6, 10e-6, 10)
        self.throughput  = np.linspace(0, 1, len(self.wavelengths))
        self.order       = int(1)
    
    
    def construct(self        : Utility, 
                  wavelengths : Array = None, 
                  throughput  : Array = None,
                  filter_name : str   = None) -> Filter:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        throughput  = self.throughput  if throughput  is None else throughput
        
        if filter_name is None:
            return dLux.core.Filter(wavelengths, throughput)
        else:
            return dLux.core.Filter(wavelengths, throughput, \
                                    filter_name=filter_name)


class InstrumentUtility(Utility):
    """
    Utility for the Instrument class.
    """
    optics   : Optics
    scene    : Scene
    detector : Detector
    filter   : Filter
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Instrument Utility.
        """    
        self.optics   = OpticsUtility().construct()
        self.scene    = SceneUtility().construct()
        self.detector = DetectorUtility().construct()
        self.filter   = FilterUtility().construct()
    
    
    def construct(self            : Utility,
                  optics          : Optics   = None,
                  scene           : Scene    = None,
                  detector        : Detector = None,
                  filter          : Filter   = None,
                  optical_layers  : list     = None,
                  sources         : list     = None,
                  detector_layers : list     = None,
                  input_layers    : bool     = False,
                  input_both      : bool     = False) -> Instrument:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        optics   = self.optics   if optics   is None else optics
        scene    = self.scene    if scene    is None else scene
        detector = self.detector if detector is None else detector
        filter   = self.filter   if filter   is None else filter
        
        if input_both:
            return dLux.core.Instrument(optics=optics,
                                        scene=scene,
                                        detector=detector,
                                        filter=filter,
                                        optical_layers=optical_layers,
                                        sources=sources,
                                        detector_layers=detector_layers)
        elif not input_layers:
            return dLux.core.Instrument(optics=optics,
                                        scene=scene,
                                        detector=detector,
                                        filter=filter)
        else:
            return dLux.core.Instrument(filter=filter,
                                        optical_layers=optical_layers,
                                        sources=sources,
                                        detector_layers=detector_layers)


##########################
### Spectrum Utilities ###
##########################
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
    
    
    def construct(self : Utility, wavelengths : Array = None,
                  weights : Array = None) -> Spectrum:
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
    
    
    def construct(self : Utility, wavelengths : Utility = None,
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
    
    
    def construct(self : Utility, wavelengths : Utility = None,
                  weights : Utility = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        weights = self.weights if weights is None else weights
        return dLux.spectrums.CombinedSpectrum(wavelengths, weights)


########################
### Source Utilities ###
########################
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
                  contrast : Array    = None,
                  name       : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position   = self.position   if position   is None else position
        flux       = self.flux       if flux       is None else flux
        spectrum   = self.spectrum   if spectrum   is None else spectrum
        contrast = self.contrast if contrast is None else contrast
        name       = self.name       if name       is None else name
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
        return dLux.sources.PointSource(position, flux, spectrum, name=name)
    
    
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
        return dLux.sources.ArrayDistribution(position, flux, spectrum, \
                                              distribution, name=name)
    
    
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
    
    
    def construct(self        : Utility,
                  position    : Array    = None,
                  flux        : Array    = None,
                  spectrum    : Spectrum = None,
                  separation  : Array    = None,
                  position_angle : Array    = None,
                  contrast  : Array    = None,
                  name        : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position    = self.position    if position    is None else position
        flux        = self.flux        if flux        is None else flux
        spectrum    = self.spectrum    if spectrum    is None else spectrum
        separation  = self.separation  if separation  is None else separation
        position_angle = self.position_angle if position_angle is None else position_angle
        contrast  = self.contrast  if contrast  is None else contrast
        name        = self.name        if name        is None else name
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
        return dLux.sources.PointExtendedSource(position, flux, spectrum, \
                                         distribution, contrast, name=name)
    
    
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
        return dLux.sources.PointAndExtendedSource(position, flux, spectrum, \
                                         distribution, contrast, name=name)


###########################
### Wavefront Utilities ###
###########################
class WavefrontUtility(Utility):
    """
    Utility for Wavefront class.
    """
    wavelength  : Array
    pixel_scale : Array
    plane_type  : PlaneType
    amplitude   : Array
    phase       : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Wavefront Utility.
        """
        self.wavelength  = np.array(550e-09)
        self.pixel_scale = np.array(1.)
        self.plane_type  = dLux.PlaneType.Pupil
        self.amplitude   = np.ones((1, 16, 16))
        self.phase       = np.zeros((1, 16, 16))


    def construct(self        : Utility,
                  wavelength  : Array = None,
                  pixel_scale : Array = None,
                  plane_type  : dLux.wavefronts.PlaneType = None,
                  amplitude   : Array = None,
                  phase       : Array = None) -> Wavefront:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelength  = self.wavelength  if wavelength  is None else wavelength
        pixel_scale = self.pixel_scale if pixel_scale is None else pixel_scale
        plane_type  = self.plane_type  if plane_type  is None else plane_type
        amplitude   = self.amplitude   if amplitude   is None else amplitude
        phase       = self.phase       if phase       is None else phase
        return dLux.wavefronts.Wavefront(wavelength, pixel_scale, amplitude,
                                         phase, plane_type)


class CartesianWavefrontUtility(WavefrontUtility):
    """
    Utility for CartesianWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianWavefront Utility.
        """
        super().__init__()


class AngularWavefrontUtility(WavefrontUtility):
    """
    Utility for AngularWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianWavefront Utility.
        """
        super().__init__()


class FarFieldFresnelWavefrontUtility(WavefrontUtility):
    """
    Utility for FarFieldFresnelWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the FarFieldFresnelWavefront Utility.
        """
        super().__init__()



############################
### Propagator Utilities ###
############################
class PropagatorUtility(Utility):
    """
    Utility for Propagator class.
    """
    dLux.propagators.Propagator.__abstractmethods__ = ()
    inverse : bool


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Propagator Utility.
        """
        self.inverse  = False


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.Propagator(inverse=inverse)


class FixedSamplingPropagatorUtility(PropagatorUtility):
    """
    Utility for FixedSamplingPropagator class.
    """
    dLux.propagators.FixedSamplingPropagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Propagator Utility.
        """
        super().__init__()


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.FixedSamplingPropagator(inverse=inverse)


class VariableSamplingPropagatorUtility(PropagatorUtility):
    """
    Utility for VariableSamplingPropagator class.
    """
    dLux.propagators.VariableSamplingPropagator.__abstractmethods__ = ()
    npixels_out     : int
    pixel_scale_out : Array
    shift           : Array
    pixel_shift     : bool


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the VariableSamplingPropagator Utility.
        """
        super().__init__()
        self.npixels_out      = 16
        self.pixel_scale_out  = np.array(1.)
        self.shift            = np.zeros(2)
        self.pixel_shift      = False


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : Array = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None,
                  inverse         : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.VariableSamplingPropagator(pixel_scale_out,
                            npixels_out, shift, pixel_shift, inverse=inverse)


class CartesianPropagatorUtility(PropagatorUtility):
    """
    Utility for CartesianPropagator class.
    """
    dLux.propagators.CartesianPropagator.__abstractmethods__ = ()
    focal_length : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianPropagator Utility.
        """
        super().__init__()
        self.focal_length = np.array(1.)


    def construct(self         : Utility,
                  focal_length : Array = None,
                  inverse      : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        focal_length = self.focal_length \
                        if focal_length is None else focal_length
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.CartesianPropagator(focal_length, \
                                                    inverse=inverse)


class AngularPropagatorUtility(PropagatorUtility):
    """
    Utility for AngularPropagator class.
    """
    dLux.propagators.AngularPropagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularPropagator Utility.
        """
        super().__init__()
        self.focal_length = np.array(1.)


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.AngularPropagator(inverse=inverse)


class FarFieldFresnelUtility(PropagatorUtility):
    """
    Utility for FarFieldFresnel class.
    """
    dLux.propagators.FarFieldFresnel.__abstractmethods__ = ()
    propagation_shift : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the FarFieldFresnel Utility.
        """
        super().__init__()
        self.propagation_shift = np.array(1e-3)


    def construct(self              : Utility,
                  propagation_shift : Array = None,
                  inverse           : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        propagation_shift = self.propagation_shift \
                            if propagation_shift is None else propagation_shift
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.FarFieldFresnel(propagation_shift, \
                                                inverse=inverse)



class CartesianMFTUtility(CartesianPropagatorUtility,
                          VariableSamplingPropagatorUtility):
    """
    Utility for CartesianMFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianMFT Utility.
        """
        super().__init__()


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : float = None,
                  focal_length    : Array = None,
                  inverse         : bool  = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.CartesianMFT(npixels_out, pixel_scale_out,
                                     focal_length, inverse, shift, pixel_shift)


class AngularMFTUtility(AngularPropagatorUtility,
                        VariableSamplingPropagatorUtility):
    """
    Utility for AngularMFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularMFT Utility.
        """
        super().__init__()


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : float = None,
                  inverse         : bool  = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.AngularMFT(npixels_out, pixel_scale_out,
                                     focal_length, inverse, shift, pixel_shift)


class CartesianFFTUtility(CartesianPropagatorUtility,
                          FixedSamplingPropagatorUtility):
    """
    Utility for CartesianFFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianFFT Utility.
        """
        super().__init__()


    def construct(self         : Utility,
                  focal_length : Array = None,
                  inverse      : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.CartesianFFT(focal_length, inverse)


class AngularFFTUtility(AngularPropagatorUtility,
                        FixedSamplingPropagatorUtility):
    """
    Utility for AngularFFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularFFT Utility.
        """
        super().__init__()


    def construct(self    : Utility, inverse : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.AngularFFT(inverse)



class CartesianFresnelUtility(FarFieldFresnelUtility, CartesianMFTUtility):
    """
    Utility for CartesianFresnel class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianFresnel Utility.
        """
        super().__init__()


    def construct(self              : Utility,
                  npixels_out       : int   = None,
                  pixel_scale_out   : float = None,
                  focal_length      : Array = None,
                  propagation_shift : Array = None,
                  inverse           : bool  = None,
                  shift             : Array = None,
                  pixel_shift       : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        propagation_shift = self.propagation_shift if propagation_shift is None\
                                                        else propagation_shift
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.CartesianFresnel(npixels_out, pixel_scale_out,
                 focal_length, propagation_shift, inverse, shift, pixel_shift)