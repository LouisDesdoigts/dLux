"""
utilities.py
This file contains testing utilities to help with the generation of 
safe test wavefronts, propagators, layers and detectors. The defined
classes are:
"""
__author__ = "Jordan Dennis"
__date__ = "28/06/2022"


import jax.numpy as numpy
import dLux
import typing


Array = typing.NewType("Array", numpy.ndarray)
Wavefront = typing.NewType("Wavefront", object)
PhysicalWavefront = typing.NewType("PhysicalWavefront", Wavefront)
AngularWavefront = typing.NewType("AngularWavefont", Wavefront)
GaussianWavefront = typing.NewType("GaussianWavefront", Wavefront)
Utility = typing.NewType("Utility", object)
WavefrontUtility = typing.NewType("WavefrontUtility", Utility)
PhysicalWavefrontUtility = typing.NewType("PhysicalWavefrontUtility", 
    WavefrontUtility)
AngularWavefrontUtility = typing.NewType("AngularWavefrontUtility", 
    WavefrontUtility)
GaussianWavefrontUtility = typing.NewType("GaussianWavefrontUtility",
    WavefrontUtility)


class UtilityUser(object):
    """
    The base utility class. These utility classes are designed to 
    define safe constructors and constants for testing. These   
    classes are for testing purposes only. 
    """
    utility : Utility


    def get_utility(self : Utility) -> Utility:
        """
        Accessor for the utility. 

        Returns 
        -------
        : Utility
            The utility
        """
        return self 


class Utility(object):
    """
    """


    def __init__(self : Utility) -> Utility:
        """
        Construct a new Utility.

        Returns
        : Utility 
            The utility. 
        """
        raise TypeError("Abstract class Utility should" + \
            "not be directly substantiated.")

    
    def construct(self : Utility) -> dLuxModule:
        """
        Safe constructor for the dLuxModule, associated with 
        this utility.

        Returns
        -------
        : dLuxModule
            A safe dLuxModule for testing.
        """
        raise TypeError("The abstract construct method" + \
            "should never be directly invoked")


class WavefrontUtility(Utility):
    """
    Defines safe state constants and a simple constructor for a safe
    `Wavefront` object. 

    Attributes
    ----------
    offset : Array[float]
        A simple array defining the angular displacement of the 
        wavefront. 
    wavelength : float
        A safe wavelength for the testing wavefronts in meters
    """
    wavelength : float
    offset : Array
    size : int
    amplitude : Array 
    phase : Array


    def __init__(self : WavefrontUtility, /, 
            wavelength : float = None, 
            offset : Array = None,
            size : int = None,
            amplitude : Array = None,
            phase : Array = None) -> WavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float = 550.e-09
            The safe wavelength for the utility in meters.
        offset : Array = [0., 0.]
            The safe offset for the utility in meters.
        size : int
            A parameter for defining consistent wavefront pixel arrays 
            without causing errors.
        amplitude : Array[float] 
            A simple array defining electric field amplitudes without 
            causing errors.
        phase : Array[float]
            A simple array defining the pixel phase for a wavefront, 
            defined to be safe. 

        Returns 
        -------
        : WavefrontUtility 
            The new utility for generating test cases.
        """
        self.wavelength = 550.e-09 if not wavelength else wavelength
        self.offset = [0., 0.] if not offset else offset           
        self.size = 128 if not size else size
        self.amplitude = numpy.ones((self.size, self.size)) if not \
            amplitude else amplitude
        self.phase = numpy.zeros((self.size, self.size)) if not \
            phase else phase

        assert self.size == self.amplitude.shape[0]
        assert self.size == self.amplitude.shape[1]
        assert self.size == self.phase.shape[0]
        assert self.size == self.phase.shape[1]          
 

    def construct(self : WavefrontUtility) -> Wavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : Wavefront
            The safe testing wavefront.
        """
        return dLux\
            .Wavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)


    def get_wavelength(self : WavefrontUtitlity) -> float:
        """
        Accessor for the wavelength associated with this utility.

        Returns
        -------
        : int
            The wavelength of the utility and hence any `Wavefront` 
            objects it creates in meters.
        """
        return self.wavelength


    def get_size(self : WavefrontUtility) -> int:
        """
        Accessor for the `size` constant.

        Returns
        -------
        : int
            The side length of a pixel array currently stored.
        """
        return self.size


    def get_amplitude(self : WavefrontUtility) -> Array:
        """
        Accessor for the `amplitude` constant.

        Returns 
        -------
        : Array
            The square array of pixel amplitudes in SI units of 
            electric field.
        """
        return self.amplitude


    def get_phase(self : WavefrontUtility) -> Array:
        """
        Accessor for the `phase` constant.

        Returns
        -------
        : Array
            The square array of pixel phases in radians.
        """
        return self.phase


    def get_offset(self : WavefrontUtility) -> Array:
        """
        Accessor for the `offset` constant.

        Returns
        -------
        : Array
            The angle that the wavefront makes with the x and 
            y planes in radians.
        """
        return self.offset


class PhysicalWavefrontUtility(WavefontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `PhysicalWavefront`.
    """
    def __init__(self : PhysicalWavefrontUtility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None) -> PhysicalWavefront:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 

        Returns
        -------
        : PhysicalWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)


    def construct(self : PhysicalWavefrontUtility) -> PhysicalWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .PhysicalWavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)
        return wavefront


class AngularWavefrontUtility(WavefontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `PhysicalWavefront`.
    """
    def __init__(self : AngularWavefrontUtility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None) -> AngularWavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 

        Returns
        -------
        : AngularWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)


    def construct(self : AngularWavefrontUtility) -> AngularWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .AngularWavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)
        return wavefront


class GaussianWavefrontUtility(PhysicalWavefrontUtility):
    """
    Defines safe state constants and a simple constructor for a 
    safe state `GaussianWavefront` object. 

    Attributes
    ----------
    beam_radius : float
        A safe radius for the GaussianWavefront in meters.
    phase_radius : float
        A safe phase radius for the GaussianWavefront in radians.
    position : float
        A safe position for the GaussianWavefront in meters.
    """
    beam_radius : float 
    phase_radius : float
    position : float


    def __init__(self : GaussianWavefrontUtility, 
            wavelength : float = None,
            offset : Array = None,
            size : int = None,
            amplitude : Array = None,
            phase : Array = None,
            beam_radius : float = None,
            phase_radius : float = None,
            position : float = None) -> GaussianWavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians.
        beam_radius : float 
            The radius of the gaussian beam in meters.
        phase_radius : float
            The phase radius of the gaussian beam in radians.
        position : float
            The position of the gaussian beam in meters.

        Returns
        -------
        : GaussianWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)
        self.beam_radius = 1. if not beam_radius else beam_radius
        self.phase_radius = 0. if not phase_radius else phase_radius
        self.position = 0. if not position else position


    def construct(self : GaussianWavefrontUtility) -> GaussianWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .GaussianWavefront(
                self.beam_radius, self.wavelength, 
                self.phase_radius, self.position, self.offset)\
            .update_phasor(self.amplitude, self.phase)
        return wavefront
