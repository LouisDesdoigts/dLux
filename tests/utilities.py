"""
utilities.py
This file contains testing utilities to help with the generation of 
safe test wavefronts, propagators, layers and detectors. The defined
classes are:

WavefrontUtitlity
PhysicalWavefrontUtility
GaussianWavefrontUtility
GaussianPropagatorUtility
FraunhoferPropagatorUtility
FresnelPropagatorUtility
"""
__author__ = "Jordan Dennis"
__date__ = "28/06/2022"


import jax.numpy as numpy
import dLux


class WavefrontUtility(object):
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


    def __init__(self : WavefrontUtility, /, 
            wavelength : float = None, 
            offset : Array = None) -> WavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float = 550.e-09
            The safe wavelength for the utility in meters.
        offset : Array = [0., 0.]
            The safe offset for the utility in meters.

        Returns 
        -------
        : WavefrontUtility 
            The new utility for generating test cases.
        """
        self.wavelength = 550.e-09 if not wavelength else wavelength
        self.offset = [0., 0.] if not offset else offset           
            

    def construct_wavefront(self : WavefrontUtility) -> Wavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : Wavefront
            The safe testing wavefront.
        """
        return dLux.Wavefront(self.wavelength, self.offset)


class PhysicalWavefrontUtility(WavefontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `PhysicalWavefront`.

    Attributes
    ----------
    size : int
        A parameter for defining consistent wavefront pixel arrays 
        without causing errors.
    amplitude : Array[float] 
        A simple array defining electric field amplitudes without 
        causing errors.
    phase : Array[float]
        A simple array defining the pixel phase for a wavefront, 
        defined to be safe. 
    """
    size : int
    amplitude : Array 
    phase : Array


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
        super().__init__(wavelength, offset)
        self.size = 128 if not size else size
        self.amplitude = numpy.ones((self.size, self.size)) if not \
            amplitude else amplitude
        self.phase = numpy.zeros((self.size, self.size)) if not \
            phase else phase

        assert self.size == self.amplitude.shape[0]
        assert self.size == self.amplitude.shape[1]
        assert self.size == self.phase.shape[0]
        assert self.size == self.phase.shape[1]


    def construct_wavefront(
            self : PhysicalWavefrontUtility) -> PhysicalWavefront:
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


    def construct_wavefront(
            self : GaussianWavefrontUtility) -> GaussianWavefront:
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
