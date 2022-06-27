import equinox 
import jax.numpy as numpy
import typing


Wavefront = typing.UserType("Wavefront", equinox.Module)
Array = typing.UserType("Array", numpy.ndarray)


class Wavefront(equinox.Module):
    """
    An abstract module that should never be directly substantiated.
    This class represents a general optical wavefront although the 
    exact implementation must go through a subclass. Wavefront
    objects are assumed to be square.


    The Wavefront and its are not intended to be public functionality
    and are initialised in a vectorised manner by the CreateWavefront 
    layer which represents the source of the optical disturbance. 


    Attributes
    ----------
    amplitude : Array
        The electric field amplitudes over the wavefront. The 
        amplitude is assumed to be in SI units. 
    phase : Array
        The phases of each pixel on the Wavefront. The phases are 
        assumed to be unitless.
    wavel : Float
        The wavelength of the light. Assumed to be in metres.
    pixelscale : Float
        The physical dimensions of each square pixel. Assumed to be 
        metres. 
    """
    amplitude : Array
    phase : Array
    wavel : float


    def __init__(self, wavelength : float):
        """
        Initialises a minimal wavefront specified only by the 
        wavelength. 

        Parameters
        ----------
        wavelength : float 
            The monochromatic wavelength associated with this 
            wavefront. 
        """


    def get_real(self) -> Array:
        """
        Throws
        ------
        : TypeError
            If self.amplitude or self.phase have not been initialised
            externally.  

        Returns 
        -------
        : Array
            The real component of the optical disturbance with 
            SI units of electric field.  
            
        """


    def get_imaginary(self) -> Array:
        """
        Throws
        ------
        : TypeError
            If self.amplitude or self.phase have not been initialised
            externally. 

        Returns
        -------
        : Array
            The imaginary component of the optical disturbance with 
            the SI units of electric field. 
        """


    def multiply_amplitude(self, 
            weights : typing.Union[float, Array]) -> Wavefront:
        """
        Modify the amplitude of the wavefront via elementwise 
        multiplication. 

        Throws
        ------
        : TypeError
            If `self.amplitude` has not initialised externally.
        : ValueError
            If `weights` is not a scalar, or an array of the same 
            dimensions. i.e.
            ```py
            ((weights.shape == (1,)) \
                | (weights.shape == self.amplitude.shape) == True
            ```

        Parameters
        ----------
        weights : Union[float, array]
            An array that has the same dimensions as self.amplitude 
            by which elementwise multiply each pixel. 
            A float to apply to the entire array at once. May simulate 
            transmission through a translucent element.

        Returns
        -------
        : Wavefront
            The new Wavefront with the applied changes to the 
            amplitude array. 
        """


    def add_phase(self: Wavefront, 
            amounts : typing.Union[float, Array]) -> Wavefront:
        """
        Used to update the wavefront phases based on the current 
        position using elementwise addition. 

        Throws
        ------
        : TypeError
            If self.phase has not been initialised externally.
        : ValueError
            If `amounts` is not of the same dimensions as `self.phases`
            or `amounts`. i.e. 
            ```py
            ((weights.shape == (1,)) \
                | (weights.shape == self.amplitude.shape) == True
            ```

        Parameters
        ----------
        amounts : Union[float, array]
            The amount of phase to add to the current phase value of 
            each pixel. A scalar modifies the global phase of the 
            wavefront. 

        Returns
        -------
        : Wavefront
            The new wavefront with the updated array of phases. 
        """


    def update_phasor(self, amplitude : Array, 
            phase : Array) -> Wavefront:  
        """
        Used to write the state of the optical disturbance. Ignores 
        the current state. It is assumed that `amplitude` and `phase`
        have the same shape i.e. `amplitude.shape == phase.shape`.
        It is not assumed that the shape of the wavefront is 
        maintained i.e. `self.amplitude.shape == amplitude.shape`
        is __not__ required. 

        Parameters
        ----------
        amplitude : Array
            The electric field amplitudes of the wavefront. Assumed to
            have the SI units of electric field. 
        phase : Array
            The phases of each pixel in the new wavefront. Assumed to 
            be unitless.

        Returns
        -------
        : Wavefront
            The new wavefront with specified by `amplitude` and `phase`        
        """


    def wavefront_to_point_spread_function(self) -> Array:
        """
        Calculates the _P_oint _S_pread _F_unction (PSF) of the 
        wavefront. 

        Throws
        ------
        : TypeError
            If `self.amplitude` has not been externally initialised.

        Returns
        -------
        : Array
            The PSF of the wavefront.
        """


    def add_optical_path_difference(self: Wavefront, 
            path_difference : typing.Union[float, Array]) -> Wavefront:
        """
        Changes the state of the wavefront based on the optical path 
        taken. 

        Throws
        ------
        : TypeError
            If `self.phase` has not been externally initialised
        : ValueError
            If `path_difference.shape != self.phase.shape` or 
            If `path_difference.shape != (1,)`

        Parameters
        ----------
        path_difference : Union[float, Array]
            The physical path difference in meters of either the 
            entire wavefront or each pixel individually. 
        
        Returns
        -------
        : Wavefront
            The new wavefront with the phases updated according to 
            `path_difference`     
        """


    # TODO: Notify @LouisDesdoigts of the numpy.norm and numpy.pnorm
    # family of functions. 
    def normalise(self : Wavefront) -> Wavefront:
        """
        Reduce the electric field amplitude of the wavefront to a 
        range between 0 and 1. Guarantees that:
        ```py
        self.get_amplitude().max() == 1.
        self.get_amplitude().min() == 0.
        ```
        
        Throws
        ------
        : TypeError
            If `self.amplitude` has not been externally instantiated.

        Returns
        -------
        : Wavefront
            The new wavefront with the normalised electric field 
            amplitudes. The amplitude is now unitless. 
        """


    def get_pixel_positions(self : Wavefront, 
            number_of_pixels : int) -> Array:
        """
        A static helper method for correctly creating paraxially pixel 
        arrays for optical transformations. 

        Parameters
        ----------
        number_of_pixels : int 
            The length of the paraxial pixel array.

        Returns
        -------
        : Array[float]
            The paraxial pixel positions of with dimensions 
            `number_of_pixels`
        """


    def get_pixel_grid(self : Wavefront) -> Array:
        """
        The pixel positions corresponding to each entry in the 
        optical disturbances stored in the Wavefront. 

        Throws
        ------
        : TypeError
            If `self.amplitude` has not been externally initialised

        Returns
        -------
        : Array 
            The pixel positions of the optical disturbances. 
            Guarantees `self.get_pixel_grid().shape == 
            self.amplitude.shape`
        """


    def get_coordinates(self : Wavefront) -> Array:
        """
        The physical coordinates of each optical disturbance in meters.

        Throws
        ------
        : TypeError
            If `self.amplitude` has not been externally initialised
        : TypeError
            If `self.pixelscale` has not been externally initialised

        Returns
        -------
        : Array
            The physical positions of the optical disturbance. 
            Guarantees that `self.get_coordinates().shape == 
            self.amplitude.shape`.
        """


    # TODO: Confirm ownership of the `invert` family.
    # TODO: Compare jaxpr with reverse. 
    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across both axes. 

        Throws
        ------
        : ValueError
            If `self.amplitude` is not externally initialised.
        : ValueError
            If `self.phase` not externally initialised.

        Returns
        -------
        : Wavefront
            The new `Wavefront` with the phase and amplitude arrays
            reversed accros both axes.
        """


    def invert_x(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across the x axis.

        Throws
        ------
        : ValueError
            If `self.amplitude` is not externally initialised.
        : ValueError 
            If `self.phase` is not externally initialised. 

        Returns
        -------
        : Wavefront
            The new `Wavefront` with the phase and the amplitude arrays
            reversed along the x axis. 
        """


    def invert_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across the y axis.

        Throws
        ------        
        : ValueError
            If `self.amplitude` is not externally initialised.
        : ValueError 
            If `self.phase` is not externally initialised. 

        Returns
        -------
        : Wavefront
            The new wavefront with the phase and the amplitude arrays 
            reversed along the y axis.
        """


class PlanarWavefront(Wavefront):
    """
    
