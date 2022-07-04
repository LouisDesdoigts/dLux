import equinox as eqx 
import jax.numpy as np
import typing


Wavefront = typing.NewType("Wavefront", eqx.Module)
PhysicalWavefront = typing.NewType("PhysicalWavefront", Wavefront)
AngularWavefront = typing.NewType("AngularWavefront", Wavefront)
GaussianWavefront = typing.NewType("FresnelWavefront", Wavefront)
Array = typing.NewType("Array", np.ndarray)


class Wavefront(eqx.Module):
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
    wavelength : float
        The wavelength of the light. Assumed to be in metres.
    offset : Array
        The polarisation state of the wave described by the x and 
        y phase lag of the wavefront. This quantity is unitless and 
        it is assumed that `offset.shape == (2, )`  
    """
    wavelength : float
    offset : Array
    amplitude : Array
    phase : Array
    pixel_scale : float


    def __init__(self : Wavefront, wavelength : float, 
            offset : Array) -> Wavefront:
        """
        Initialises a minimal wavefront specified only by the 
        wavel. 

        Parameters
        ----------
        wavelength : float 
            The monochromatic wavelength associated with this 
            wavefront. 
        offset : Array
            The x and y angles of incidence to the surface assumed to 
            be in radians. 
        """
        self.wavelength = np.array(wavelength).astype(float) # Jax Safe
        self.offset = np.array(offset).astype(float) # To be instantiated by CreateWavefront        
        self.amplitude = None
        self.phase = None
        self.pixel_scale = None


    def get_pixel_scale(self : GaussianWavefront) -> float:
        """
         Returns
        -------
        pixel_scale : float
            The pixel_scale associated with the current position.
        """
        return self.pixel_scale


    def get_offset(self : Wavefront) -> Array:
        """
        Returns
        -------
        offset : Array
            The x and y angles of deviation that the wavefront makes
            from the mirror.
        """
        return self.offset


    def set_offset(self : Wavefront, offset : Array) -> Wavefront:
        """
        Parameters
        ----------
        offset : Array (f64[2])
            The angles that the `Wavefront` makes with the x and y 
            axis assumed to be in radians.
        
        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated offset. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.offset, self, offset,
            is_leaf = lambda leaf : leaf is None )


    def get_wavelength(self : Wavefront) -> float:
        """
        Returns
        -------
        wavelength : float
            The wavelength of the `Wavefront` in meters. 
        """
        return self.wavelength


    def set_wavelength(self : Wavefront, wavelength : float) -> Wavefront:
        """
        Parameters
        ----------
        wavelength : float  
            The new wavelength of the `Wavefront` assumed to be in meters.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated wavelength. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.wavelength, self, wavelength,
            is_leaf = lambda leaf : leaf is None)


    def get_amplitude(self : Wavefront) -> Array:
        """
        Returns
        -------
        amplitude : Array 
            The electric field amplitude in SI units of electric field. 
        """
        return self.amplitude


    def get_phase(self : Wavefront) -> Array:
        """
        Returns
        -------
        phase : Array 
            The phase of the Wavefront; a unitless qunatity.
        """
        return self.phase


    def set_amplitude(self : Wavefront, amplitude : Array) -> Wavefront:
        """
        Parameters
        ---------
        amplitude : Array
            The new amplitudes of the `Wavefront` assumed to be in the 
            SI units of electric field.         

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated amplitude. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.amplitude, self, amplitude,
            is_leaf = lambda leaf : leaf is None)


    def set_phase(self : Wavefront, phase : Array) -> Wavefront:
        """
        Parameters
        ----------
        phase : Array
            The new phases of the `Wavefront` assumed to be unitless.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with `Wavefront.get_phase() == phase`.
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.phase, self, phase,
            is_leaf = lambda leaf : leaf is None) 


    def get_real(self : Wavefront) -> Array:
        """
        The real component of the `Wavefront`. 

        Throws
        ------
        : TypeError
            If self.amplitude or self.phase have not been initialised
            externally.  

        Returns 
        -------
        wavefront : Array
            The real component of the optical disturbance with 
            SI units of electric field.  
        """
        return self.get_amplitude() * np.cos(self.get_phase())
        

    def get_imaginary(self : Wavefront) -> Array:
        """
        The imaginary component of the `Wavefront`

        Throws
        ------
        : TypeError
            If self.amplitude or self.phase have not been initialised
            externally. 

        Returns
        -------
        wavefront : Array
            The imaginary component of the optical disturbance with 
            the SI units of electric field. 
        """
        return self.get_amplitude() * np.sin(self.get_phase())

    
    def number_of_pixels(self : Wavefront) -> int:
        """
        The side length of the pixel array that represents the 
        electric field of this wavefront.

        Throws
        ------
        error : TypeError
            If the amplitude and phase of the wavefront have not been
            externally initialised.

        Returns 
        -------
        pixels : int
            The number of pixels that represent this wavefront in 
            memory along one side.
        """
        return self.get_amplitude().shape[0]        


    def multiply_amplitude(self : Wavefront, 
            array_like : typing.Union[float, Array]) -> Wavefront:
        """
        Modify the amplitude of the wavefront via elementwise 
        multiplication. 

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not initialised externally.
        error : ValueError
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
        wavefront : Wavefront
            The new Wavefront with the applied changes to the 
            amplitude array. 
        """
        return self.set_amplitude(self.get_amplitude() * array_like)


    def add_phase(self : Wavefront, 
            phases : typing.Union[float, Array]) -> Wavefront:
        """
        Used to update the wavefront phases based on the current 
        position using elementwise addition. 

        Throws
        ------
        error : TypeError
            If self.phase has not been initialised externally.
        error : ValueError
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
        wavefront : Wavefront
            The new wavefront with the updated array of phases. 
        """
        return self.set_phase(self.get_phase() + phases)


    def update_phasor(self : Wavefront, amplitude : Array, 
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
        wavefront : Wavefront
            The new wavefront with specified by `amplitude` and `phase`        
        """
        return self.set_phase(phase).set_amplitude(amplitude)


    def wavefront_to_psf(self : Wavefront) -> Array:
        """
        Calculates the _P_oint _S_pread _F_unction (PSF) of the 
        wavefront. 

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        return self.get_amplitude() ** 2


    def add_opd(self: Wavefront, 
            path_difference : typing.Union[float, Array]) -> Wavefront:
        """
        Changes the state of the wavefront based on the optical path 
        taken. 

        Throws
        ------
        error : TypeError
            If `self.phase` has not been externally initialised
        error : ValueError
            If `path_difference.shape != self.phase.shape` or 
            If `path_difference.shape != (1,)`

        Parameters
        ----------
        path_difference : Union[float, Array]
            The physical path difference in meters of either the 
            entire wavefront or each pixel individually. 
        
        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phases updated according to 
            `path_difference`     
        """
        phase_difference = 2 * np.pi * path_difference / self.wavelength
        return self.add_phase(phase_difference)


    def get_complex_form(self : Wavefront) -> Array:
        """
        The electric field described by this Wavefront in complex 
        form.

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally instantiated.
        error : TypeError
            If `self.phase` has not been externally instantiated.
        
        Returns
        -------
        field : Array[complex]
            The complex electric field with both the real and 
            imaginary components in SI units.
        """
        return self.get_amplitude() * np.exp(1j * self.get_phase()) 


    def normalise(self : Wavefront) -> Wavefront:
        """
        Reduce the electric field amplitude of the wavefront to a 
        range between 0 and 1. Guarantees that:
        
        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally instantiated.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field 
            amplitudes. The amplitude is now unitless. 
        """
        total_intensity = np.linalg.norm(self.get_amplitude())
        return self.multiply_amplitude(1 / total_intensity)


    def get_pixel_coordinates(self : Wavefront, 
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
        pixel_coordinates : Array
            The paraxial pixel positions of with dimensions 
            `number_of_pixels`
        """
        return np.arange(number_of_pixels) - (number_of_pixels - 1) / 2


    def get_pixel_grid(self : Wavefront) -> Array:
        """
        The pixel positions corresponding to each entry in the 
        optical disturbances stored in the Wavefront. 

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised

        Returns
        -------
        pixel_grid : Array 
            The pixel positions of the optical disturbances. 
            Guarantees `self.get_pixel_grid().shape == 
            self.amplitude.shape`
        """
        pixel_positions = self.get_pixel_coordinates(self.amplitude.shape[0])
        x_positions, y_positions = \
            np.meshgrid(pixel_positions, pixel_positions)
        return np.array([x_positions, y_positions])


    def get_pixel_positions(self : Wavefront) -> Array:
        """
        The physical coordinates of each optical disturbance in meters.

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised
        error : TypeError
            If `self.pixel_scale` has not been externally initialised

        Returns
        -------
        pixel_positions : Array
            The physical positions of the optical disturbance. 
            Guarantees that `self.get_coordinates().shape == 
            self.amplitude.shape`.
        """
        return self.get_pixel_scale() * self.get_pixel_grid()


    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across both axes. 

        Throws
        ------
        error : ValueError
            If `self.amplitude` is not externally initialised.
        error : ValueError
            If `self.phase` not externally initialised.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays
            reversed accros both axes.
        """
        return self.invert_x().invert_y()


    def invert_x(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across the x axis.

        Throws
        ------
        error : ValueError
            If `self.amplitude` is not externally initialised.
        error : ValueError 
            If `self.phase` is not externally initialised. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and the amplitude arrays
            reversed along the x axis. 
        """
        new_amplitude = self.amplitude[:, ::-1]
        new_phase = self.phase[:, ::-1]
        return self.update_phasor(new_amplitude, new_phase)


    def invert_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront across the y axis.

        Throws
        ------        
        error : ValueError
            If `self.amplitude` is not externally initialised.
        error : ValueError 
            If `self.phase` is not externally initialised. 

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phase and the amplitude arrays 
            reversed along the y axis.
        """
        new_amplitude = self.amplitude[::-1, :]
        new_phase = self.phase[::-1, :]
        return self.update_phasor(new_amplitude, new_phase)


    # TODO: Make logic Jax-Safe
    def interpolate(self : Wavefront, coordinates : Array, 
            real_imaginary : bool = False) -> tuple:
        """
        Interpolates the `Wavefront` at the points specified by 
        coordinates. The default interpolation uses the amplitude 
        and phase although by passing `real_imgainary == True` 
        the interpolation can be based on the real and imaginary 
        information. The main use of this function is as a helper 
        method to `self.paraxial_interpolate`.

        Parameters
        ----------
        coordinates : Array
            The coordinates to interpolate the optical disturbance
            at. Assumed to have the units meters. 
        real_imaginary : bool
            Whether to use the amplitude-phase or real-imaginary
            representation for the interpolation. The amplitude-
            phase representation is slightly faster.

        Returns
        -------
        field : tuple[Array, Array]
            The amplitude and phase of the wavefront at `coordinates`
            based on a linear interpolation.
        """
        if not real_imaginary:
            new_amplitude = map_coordinates(
                self.amplitude, coordinates, order=1)
            new_phase = map_coordinates(
                self.phase, coordinates, order=1)
        else:
            real = map_coordinates(
                self.get_real(), coordinates, order=1)
            imaginary = map_coordinates(
                self.get_imaginary(), coordinates, order=1)
            new_amplitude = np.hypot(real, imaginary)
            new_phase = np.arctan2(imaginary, real)
        return new_amplitude, new_phase


    def paraxial_interpolate(self : Wavefront, 
            pixel_scale_out : float, number_of_pixels_out : int,
            real_imaginary : bool = False) -> Wavefront: 
        """
        Interpolates the `Wavefront` so that it remains centered on 
        each pixel (paraxial). Calculation can be performed using 
        either the real-imaginary or amplitude-phase representations 
        of the wavefront. The default is amplitude-phase. 

        Parameters
        ----------
        pixel_scale_out : float
            The dimensions of a single square pixel after the 
            interpolation.
        number_of_pixels_out : int
            The number of pixels along one side of the square
            `Wavefront` after the interpolation. 
        real_imaginary : bool = False
            Whether to use the real-imaginary representation of the 
            wavefront for the interpolation. 

        Returns
        -------
        wavefront : PhysicalWavefront
            The new wavefront with the updated optical disturbance. 
        """
        # Get coords arrays
        number_of_pixels_in = self.amplitude.shape[0]
        ratio = pixel_scale_out / self.pixel_scale
        
        centre = (number_of_pixels_in - 1) / 2
        new_centre = (number_of_pixels_out - 1) / 2
        pixels = ratio * (-new_centre, new_centre, number_of_pixels_out) + centre
        x_pixels, y_pixels = np.meshgrid(pixels, pixels)
        coordinates = np.array([y_pixels, x_pixels])
        new_amplitude, new_phase = self.interpolate(
            coordinates, real_imaginary=real_imaginary)
        
        # Update Phasor
        self = self.update_phasor(new_amplitude, new_phase)
        
        # Conserve energy
        self = self.multiply_amplitude(ratio)
        
        return self.set_pixel_scale(pixel_scale_out)


    def set_pixel_scale(self : Wavefront, pixel_scale : float) -> Wavefront:
        """
        Mutator for the pixel_scale.

        Parameters
        ----------
        pixel_scale : float
            The new pixel_scale associated with the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new Wavefront object with the updated pixel_scale
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.pixel_scale, self, pixel_scale,
            is_leaf = lambda leaf : leaf is None)


    def pad_to(self : Wavefront, number_of_pixels_out : int) -> Wavefront:
        """
        Pads the `Wavefront` with zeros. Assumes that 
        `number_of_pixels_out > self.amplitude.shape[0]`. 
        Note that `Wavefronts` with even pixel dimensions can 
        only be padded (without interpolation) to even pixel 
        dimensions and vice-versa. 

        Throws
        ------
        error : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[0]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 

        Returns
        -------
        wavefront : PhysicalWavefront
            The new `Wavefront` with the optical disturbance zero 
            padded to the new dimensions.
        """
        number_of_pixels_in = self.amplitude.shape[0]

        if number_of_pixels_in % 2 != number_of_pixels_out % 2:
            raise ValueError("Only supports even -> even or odd -> odd padding")
       
        # TODO: Error (>) for the JAX silent error. 
 
        new_centre = number_of_pixels_out // 2
        centre = number_of_pixels_in // 2
        remainder = number_of_pixels_in % 2
        padded = np.zeros([number_of_pixels_out, number_of_pixels_out])
        
        new_amplitude = padded.at[
                new_centre - centre : centre + new_centre + remainder, 
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        new_phase = padded.at[
                new_centre - centre : centre + new_centre + remainder, 
                new_centre - centre : centre + new_centre + remainder
            ].set(self.phase)
        return self.update_phasor(new_amplitude, new_phase)


    def crop_to(self : Wavefront, number_of_pixels_out : int) -> Wavefront:
        """
        Crops a `Wavefront`'s optical disturbance. Assumes that 
        `number_of_pixels_out < self.amplitude.shape[0]`. 
        `Wavefront`s with an even number of pixels can only 
        be cropped to an even number of pixels without interpolation
        and vice-versa.    
        
        Throws
        ------
        error : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[0]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 


        Returns
        -------
        wavefront : PhysicalWavefront
            The new `Wavefront` with the optical disturbance zero 
            cropped to the new dimensions.
        """
        number_of_pixels_in = self.amplitude.shape[0]
        
        if number_of_pixels_in%2 != number_of_pixels_out%2:
            raise ValueError("Only supports even -> even or 0dd -> odd cropping")
        
        new_centre = number_of_pixels_in // 2
        centre = number_of_pixels_out // 2

        new_amplitude = self.amplitude[
            new_centre - centre : new_centre + centre, 
            new_centre - centre : new_centre + centre]
        new_phase = self.phase[
            new_centre - centre : new_centre + centre, 
            new_centre - centre : new_centre + centre]

        return self.update_phasor(new_amplitude, new_phase)


class PhysicalWavefront(Wavefront):
    """
    A simple plane wave extending the abstract `Wavefront` class. 
    Assumes that the wavefront is square. This is Physical as 
    opposed to Angular, because there are no infinite distances.  

    Attributes
    ----------
    plane_type : str
        The type of plane occupied by the wavefront. 
    amplitude : Array
        The electric field amplitudes over the wavefront. The 
        amplitude is assumed to be in SI units. 
    phase : Array
        The phases of each pixel on the Wavefront. The phases are 
        assumed to be unitless.
    pixel_scale : float
        The physical dimensions of each square pixel. Assumed to be 
        metres.
    offset : Array
        The angle that the `Wavefront` makes with the OpticalElement
        in radians. This is a (1, 2) array such that 
        `offset <= 2 * np.pi`. 
    """
    plane_type : str # For debugging


    def __init__(self : PhysicalWavefront, wavelength : float, 
            offset : Array) -> PhysicalWavefront:
        """
        Constructor for a `PhysicalWavefront`.

        Parameters
        ----------
        wavelength : float 
            The wavelength associated with this wavefront in 
            meters.
        offset : Array
            The angle that the wavefront makes with the element
            in the x and y coordinates.

        Returns
        -------
        wavefront : PhysicalWavefront
            The new wavefront with `None` at the extra leaves. 
        """
        super().__init__(wavelength, offset)
        self.plane_type = None


    def get_plane_type(self : Wavefront) -> str:
        """
        Returns
        -------
        plane : str
            The plane that the `Wavefront` is currently in. The 
            options are currently "Pupil", "Focal" and "None".
        """
        return self.plane_type


    def set_plane_type(self : Wavefront, plane : str) -> Wavefront:
        """
        Parameters
        ----------
        plane : str
            A string describing the plane that the `Wavefront` is 
            currently in.

        Returns 
        -------
        wavefront : Wavefront 
            The new `Wavefront` with the update plane information.
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.plane_type, self, plane,
            is_leaf = lambda leaf : leaf is None)


    def transfer_function(self : PhysicalWavefront, 
            distance : float) -> float:
        """
        The _O_ptical _T_ransfer _F_unction corresponding to the 
        evolution of the wavefront when propagating a distance.
        
        Parameters
        ----------
        distance : float
            The distance that is getting propagated in meters.

        Returns
        -------
        phase : Array
            The phase that represents the optical transfer. 
        """
        wavenumber = 2. * np.pi / self.get_wavelength()
        return np.exp(1.0j * wavenumber * distance) / \
            (1.0j * self.get_wavelength() * distance)


class AngularWavefront(Wavefront):
    """
    A wavefront parametrised by phase and amplitude arrays, as 
    well as the pixel scale. The units are radians as opposed to 
    meters (in `PhysicalWavefront`).

    Attributes
    ----------
    plane_type : str
        The type of plane occupied by the wavefront. 
    amplitude : Array
        The electric field amplitudes over the wavefront. The 
        amplitude is assumed to be in SI units. 
    phase : Array
        The phases of each pixel on the Wavefront. The phases are 
        assumed to be unitless.
    pixel_scale : float
        The physical dimensions of each square pixel. Assumed to be 
        metres.
    """ 
    # TODO: Convince @LouisDesdoigts that this should be in a 
    # separate debugging class.
    plane_type : str


    def __init__(self : AngularWavefront, wavelength : float, 
            offset : Array) -> AngularWavefront:
        """
        Constructor for a `AngularWavefront`.

        Parameters
        ----------
        wavelength : float 
            The wavelength associated with this wavefront in 
            meters.
        offset : Array
            The angle that the wavefront makes with the element
            in the x and y coordinates.

        Returns
        -------
        wavefront : AngularWavefront
            The new wavefront with `None` at the extra leaves. 
        """
        super().__init__(wavelength, offset)
        self.plane_type = None        


    def get_plane_type(self : Wavefront) -> str:
        """
        Returns
        -------
        plane : str
            The plane that the `Wavefront` is currently in. The 
            options are currently "Pupil", "Focal" and "None".
        """
        return self.plane_type


    def set_plane_type(self : Wavefront, plane : str) -> Wavefront:
        """
        Parameters
        ----------
        plane : str
            A string describing the plane that the `Wavefront` is 
            currently in.

        Returns 
        -------
        wavefront : Wavefront 
            The new `Wavefront` with the update plane information.
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.plane_type, self, plane,
            is_leaf = lambda leaf : leaf is None)


class GaussianWavefront(Wavefront):
    """
    Expresses the behaviour and state of a wavefront propagating in 
    an optical system under the fresnel assumptions. This 
    implementation is based on the same class from the `poppy` 
    library [poppy](https://github.com/spacetelescope/poppy/fresnel.py)
    and Chapter 3 from _Applied Optics and Optical Engineering_
    by Lawrence G. N.


    Approximates the wavefront as a Gaussian Beam parametrised by the 
    radius of the beam, the phase radius, the phase factor and the 
    Rayleigh distance. Propagation is based on two different regimes 
    for a total of four different opertations. 
    
    Attributes
    ----------
    position : float
        The position of the wavefront in the optical system assumed 
        to be in meters.
    beam_radius : float
        The radius of the beam assumed to be in metres. 
    phase_radius : float
        The phase radius of the gaussian beam assumed to be unitless. 
    """
    position : float 
    beam_radius : float
    phase_radius : float


    def __init__(self : GaussianWavefront, 
            offset : Array,
            wavelength : float) -> GaussianWavefront:
        """
        Creates a wavefront with an empty amplitude and phase 
        arrays but of a given wavelength and phase offset. 

        Parameters
        ----------
        beam_radius : float
            Radius of the beam at the initial optical plane.
        wavelength : float
            Wavelength of the monochromatic light.
        offset : Array
            Phase shift of the initial optical plane. 
        phase_radius :  float
            The phase radius of the GuasianWavefront. This is a unitless
            quantity. 
        """
        super().__init__(wavelength, offset)
        self.beam_radius = None
        self.phase_radius = np.inf
        self.position = None 


    def get_position(self : GaussianWavefront) -> float:
        """
        Accessor for the position of the wavefront. 

        Returns 
        -------
        position : float 
            The position of the `Wavefront` from its starting point 
            in meters.
        """
        return self.position


    def set_position(self : GaussianWavefront, 
            position : float) -> GaussianWavefront:
        """
        Mutator for the position of the wavefront. Changes the 
        pixel_scale which is a function of the position.  

        Parameters
        ----------
        position : float
            The new position of the wavefront from its starting point 
            assumed to be in meters. 
        
        Returns
        -------
        wavefront : GaussianWavefront
            This wavefront at the new position. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.position, self, position,
            is_leaf = lambda leaf : leaf is None)


    def get_beam_radius(self : GaussianWavefront) -> float:
        """
        Accessor for the radius of the wavefront.

        Returns
        -------
        beam_radius : float
            The radius of the `GaussianWavefront` in meters.
        """
        return self.beam_radius


    def get_phase_radius(self : GaussianWavefront) -> float:
        """
        Accessor for the phase radius of the wavefront.

        Returns
        -------
        phase_radius : float 
            The phase radius of the wavefront. This is a unitless 
            quantity.
        """
        return self.phase_radius


    def rayleigh_distance(self: GaussianWavefront) -> float:
        """
        Calculates the rayleigh distance of the Gaussian beam.
        
        Returns
        -------
        rayleigh_distance : float
            The Rayleigh distance of the wavefront in metres.
        """
        return np.pi * self.get_beam_radius() ** 2\
            / self.get_wavelength()


    def transfer_function(self: GaussianWavefront, distance: float) -> Array:
        """
        The optical transfer function (OTF) for the gaussian beam.
        Assumes propagation is along the axis. 

        Parameters
        ----------
        distance : float
            The distance to propagate the wavefront along the beam 
            via the optical transfer function in metres.

        Returns
        -------
        phase : float 
            A phase representing the evolution of the wavefront over 
            the distance. 

        References
        ----------
        Wikipedia contributors. (2022, January 3). Direction cosine. 
        In Wikipedia, The Free Encyclopedia. June 23, 2022, from 
        https://en.wikipedia.org/wiki/Direction_cosine

        Wikipedia contributors. (2022, January 3). Spatial frequecy.
        In Wikipedia, The Free Encyclopedia. June 23, 2022, from 
        https://en.wikipedia.org/wiki/Spatial_frequency
        """
        coordinates = self.get_pixel_positions()
        radius = np.sqrt((coordinates ** 2).sum(axis=0))
        xi = coordinates[0, :, :] / radius / self.get_wavelength()
        eta = coordinates[1, :, :] / radius / self.get_wavelength()
        return np.exp(1j * np.pi * self.get_wavelength() \
            * distance * (xi ** 2 + eta ** 2))


    def quadratic_phase_factor(self: GaussianWavefront, 
            distance: float) -> float:
        """
        Convinience function that simplifies many of the diffraction
        equations. Caclulates a quadratic phase factor associated with 
        the beam. 

        Parameters
        ----------
        distance : float
            The distance of the propagation measured in metres. 

        Returns
        -------
        phase : float
            The near-field quadratic phase accumulated by the beam
            from a propagation of distance.
        """      
        return np.exp(1j * np.pi * \
            (self.get_pixel_positions() ** 2).sum(axis=0) \
            / self.get_wavelength() / distance)


    def location_of_waist(self: GaussianWavefront) -> float:
        """
        Calculates the position of the waist along the direction of 
        propagation based of the current state of the wave.

        Returns
        -------
        waist : float
            The position of the waist in metres.
        """
        return - self.get_phase_radius() / \
            (1 + (self.get_phase_radius() / \
            self.rayleigh_distance()) ** 2)


    def waist_radius(self: GaussianWavefront) -> float:
        """
        The radius of the beam at the waist.

        Returns
        -------
        waist_radius : float
            The radius of the beam at the waist in metres.
        """
        return self.get_beam_radius() / \
            np.sqrt(1 + (self.rayleigh_distance() \
                / self.get_beam_radius()) ** 2) 


    def calculate_pixel_scale(self: GaussianWavefront, position: float) -> None:
        """
        The pixel scale at the position along the axis of propagation.
        Assumes that the wavefront is square. That is:
        ```
        x, y = self.amplitude.shape
        (x == y) == True
        ```

        Parameters
        ----------
        position : float
            The position of the wavefront aling the axis of propagation
            in metres.
        """
        # TODO: get_number_of_pixels() Used frequenctly not a function
        number_of_pixels = self.amplitude.shape[0]
        new_pixel_scale = self.get_wavelength() * np.abs(position) / \
            number_of_pixels / self.get_pixel_scale()  
        return new_pixel_scale 
        

    def is_inside(self: GaussianWavefront, distance: float) -> bool:
        """ 
        Determines whether a point at along the axis of propagation 
        at distance away from the current position is inside the 
        rayleigh distance. 

        Parameters
        ----------
        distance : float
            The distance to test in metres.

        Returns
        -------
        inside : bool
            true if the point is within the rayleigh distance false 
            otherwise.
        """
        return np.abs(self.get_position() + distance - \
            self.location_of_waist()) <= self.rayleigh_distance()


    def set_phase_radius(self : GaussianWavefront, 
            phase_radius : float) -> GaussianWavefront:
        """
        Mutator for the phase_radius.

        Parameters
        ----------
        phase_radius : float
            The new phase_radius in meters.

        Returns
        -------
        wavefront : GaussianWavefront
            A modified GaussianWavefront with the new phase_radius.
        """
        return eqx.tree_at(lambda wavefront : wavefront.phase_radius, 
            self, phase_radius, is_leaf = lambda leaf : leaf is None)


    def set_beam_radius(self : GaussianWavefront, 
            beam_radius : float) -> GaussianWavefront:
        """
        Mutator for the `beam_radius`.

        Parameters
        ----------
        beam_radius : float
            The new beam_radius in meters.

        Returns
        -------
        wavefront : GaussianWavefront
            A modified GaussianWavefront with the new beam_radius.
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.beam_radius, self, beam_radius,
            is_leaf = lambda leaf : leaf is None)
