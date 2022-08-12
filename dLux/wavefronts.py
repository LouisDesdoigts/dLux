import equinox as eqx 
import jax.numpy as np
import typing
import enum


__all__ = ["Wavefront", "GaussianWavefront", 
    "CartesianWavefront", "AngularWavefront", "PlaneType"]


Wavefront = typing.NewType("Wavefront", object)
CartesianWavefront = typing.NewType("CartesianWavefront", Wavefront)
AngularWavefront = typing.NewType("AngularWavefront", Wavefront)
GaussianWavefront = typing.NewType("FresnelWavefront", Wavefront)
PlaneType = typing.NewType("PlaneType", object)
Array = typing.NewType("Array", np.ndarray)


class PlaneType(enum.IntEnum):
    """
    Enumeration object to keep track of plane types, and have 
    jax-safe logic based of wavefront location
    """
    Pupil = 1
    Focal = 2
    Intermediate = 3


class Wavefront(eqx.Module):
    """
    An abstract module that should never be directly instantiated.
    This class represents a general optical wavefront although the 
    exact implementation must go through a subclass. Wavefront
    objects are assumed to be square.


    The Wavefront and its contents are not intended to be public 
    functionality. The object is initialised using in a vectorised 
    manner using jax.vmap over both wavelength and offset. These 
    values are initialised within the OpticalSystem class by the 
    CreateWavefront layer which represents the source of the input
    wavefront. 


    Attributes
    ----------
    wavelength : float, meters
        The wavelength of the `Wavefront`.
    offset : Array, radians
        The (x, y) angular offset of the `Wavefront` from 
        the optical axis.
    amplitude : Array, power
        The electric field amplitude of the `Wavefront`. 
    phase : Array, radians
        The electric field phase of the `Wavefront`.
    pixel_scale : float, meters/pixel
        The physical dimensions of each square pixel.
    plane_type : enum
        The current plane of wavefront, can be Pupil, Focal
        or Intermediate
    """
    wavelength : float
    offset : Array
    amplitude : Array
    phase : Array
    pixel_scale : float
    plane_type : PlaneType


    def __init__(self : Wavefront, wavelength : float, 
            offset : Array) -> Wavefront:
        """
        Initialises a minimal `Wavefront` specified only by the 
        wavelength and offset.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront` in meters. 
        offset : Array, radians
            The (x, y) angular offset of the wavefront from 
            the optical axis.
        """
        self.wavelength = np.array(wavelength).astype(float)
        self.offset = np.array(offset).astype(float) 
        self.amplitude = None   # To be instantiated by CreateWavefront
        self.phase = None       # To be instantiated by CreateWavefront
        self.pixel_scale = None # To be instantiated by CreateWavefront
        self.plane_type = None   # To be instantiated by CreateWavefront


    def get_pixel_scale(self : GaussianWavefront) -> float:
        """
         Returns
        -------
        pixel_scale : float, meters/pixel
            The current pixel_scale associated with the wavefront.
        """
        return self.pixel_scale


    def get_offset(self : Wavefront) -> Array:
        """
        Returns
        -------
        offset : Array, radians
            The (x, y) angular offset of the wavefront from the optical 
            axis.
        """
        return self.offset


    def set_offset(self : Wavefront, offset : Array) -> Wavefront:
        """
        Parameters
        ----------
        offset : Array, radians
            The (x, y) angular offset of the wavefront from the optical 
            axis.
        
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
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        """
        return self.wavelength


    def set_wavelength(self : Wavefront, wavelength : float) -> Wavefront:
        """
        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.

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
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        """
        return self.amplitude


    def get_phase(self : Wavefront) -> Array:
        """
        Returns
        -------
        phase : Array, radians
            The phases of each pixel on the `Wavefront`.
        """
        return self.phase


    def set_amplitude(self : Wavefront, amplitude : Array) -> Wavefront:
        """
        Parameters
        ---------
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 

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
        phase : Array, radians
            The phases of each pixel on the `Wavefront`.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated phase. 
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
            The real component of the complex `Wavefront`.
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
           The imaginary component of the complex `Wavefront`.
        """
        return self.get_amplitude() * np.sin(self.get_phase())

    
    def number_of_pixels(self : Wavefront) -> int:
        """
        The side length of the pixel array that represents the 
        electric field of the `Wavefront`. Calcualtes the `pixels`
        value from the shape of the amplitude array.

        Throws
        ------
        error : TypeError
            If the amplitude and phase of the `Wavefront` have not been
            externally initialised.

        Returns 
        -------
        pixels : int
            The number of pixels that represent the `Wavefront` along 
            one side.
        """
        return self.get_amplitude().shape[-1]
    
    
    def get_diameter(self : Wavefront) -> float:
        """
        Returns the current Wavefront diameter
        
        TODO: Add tests for this function
        
        Throws
        ------
        : TypeError
            If self.amplitude or self.phase have not been initialised
            externally. 

        Returns
        -------
        diameter : float
           The current diameter of the wavefront
        """
        return self.number_of_pixels() * self.get_pixel_scale()


    def multiply_amplitude(self : Wavefront, 
            array_like : typing.Union[float, Array]) -> Wavefront:
        """
        Multiply the amplitude of the `Wavefront` by either a float or
        array. 

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
        array_like : Union[float, array]
            An array that has the same dimensions as self.amplitude 
            by which elementwise multiply each pixel. 
            A float to apply to the entire array at once.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the applied changes to the 
            amplitude array. 
        """
        return self.set_amplitude(self.get_amplitude() * array_like)


    def add_phase(self : Wavefront, 
            array_like : typing.Union[float, Array]) -> Wavefront:
        """
        Add either a float or array of phase to `Wavefront`.

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
        array_like : Union[float, array]
            The amount of phase to add to the current phase value of 
            each pixel. A scalar modifies the global phase of the 
            wavefront. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated array of phases. 
        """
        return self.set_phase(self.get_phase() + array_like)


    def update_phasor(self : Wavefront, amplitude : Array, 
            phase : Array) -> Wavefront:  
        """
        Used to update the state of the wavefront. This should typically
        only be called from within a propagator layers in order to ensure
        that values such as pixelscale are updates appropriately. It is 
        assumed that `amplitude` and `phase` have the same shape 
        i.e. `amplitude.shape == phase.shape`. It is not assumed that the 
        shape of the wavefront is  maintained 
        i.e. `self.amplitude.shape == amplitude.shape` is __not__ required. 

        Parameters
        ----------
        amplitude : Array, power
            The electric field amplitudes of the wavefront.
        phase : Array, radians
            The phases of each pixel in the new wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` specified by `amplitude` and `phase`        
        """
        return self.set_phase(phase).set_amplitude(amplitude)


    def wavefront_to_psf(self : Wavefront) -> Array:
        """
        Calculates the Point Spread Function (PSF), ie the squared modulus
        of the complex wavefront.

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        return np.sum(self.get_amplitude() ** 2, axis=0)


    def add_opd(self: Wavefront, 
            path_difference : typing.Union[float, Array]) -> Wavefront:
        """
        Applies the wavelength-dependent phase based on the supplied 
        optical path difference.

        Throws
        ------
        error : TypeError
            If `self.phase` has not been externally initialised
        error : ValueError
            If `path_difference.shape != self.phase.shape` or 
            If `path_difference.shape != (1,)`

        Parameters
        ----------
        path_difference : Union[float, Array], meters
            The physical optical path difference of either the 
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
            The complex electric field of the wavefront.
        """
        return self.get_amplitude() * np.exp(1j * self.get_phase()) 


    def normalise(self : Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.
        
        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally instantiated.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field 
            amplitudes.
        """
        total_intensity = np.linalg.norm(self.get_amplitude())
        return self.multiply_amplitude(1 / total_intensity)


    def get_pixel_coordinates(self : Wavefront, 
            number_of_pixels : int) -> Array:
        """
        Returns a vector of pixel indexes centered on the optical axis with 
        length number_of_pixels.
        
        
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
        Returns the wavefront pixel grid relative to the optical axis.

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised

        Returns
        -------
        pixel_grid : Array 
            The paraxial pixel coordaintes.
            Guarantees `self.get_pixel_grid().shape == 
            self.amplitude.shape`
        """
        pixel_positions = self.get_pixel_coordinates(self.number_of_pixels())
        x_positions, y_positions = \
            np.meshgrid(pixel_positions, pixel_positions)
        return np.array([x_positions, y_positions])


    def get_pixel_positions(self : Wavefront) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters

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
    

    def get_plane_type(self : Wavefront) -> str:
        """
        Returns
        -------
        plane : str
            The plane that the `Wavefront` is currently in. The 
            options are currently "Pupil", "Focal" and "None".
        """
        return self.plane_type


    def set_plane_type(self : Wavefront, plane : PlaneType) -> Wavefront:
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


    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about both axes. 

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
            reversed along both axes.
        """
        return self.invert_x().invert_y()


    def invert_x(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about the x axis.

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
        Reflects the wavefront about the y axis.

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
        and phase, however by passing `real_imgainary=True` 
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
            representation for the interpolation.

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
        the optical axis. Calculation can be performed using 
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
        wavefront : Wavefront
            The new wavefront with the updated optical disturbance. 
        """
        # Get coords arrays
        number_of_pixels_in = self.number_of_pixels()
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
        `number_of_pixels_out > self.amplitude.shape[-1]`. 
        Note that `Wavefronts` with even pixel dimensions can 
        only be padded (without interpolation) to even pixel 
        dimensions and vice-versa. 

        Throws
        ------
        error : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[-1]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the optical disturbance zero 
            padded to the new dimensions.
        """
        number_of_pixels_in = self.number_of_pixels()

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
        Crops the `Wavefront`. Assumes that 
        `number_of_pixels_out < self.amplitude.shape[-1]`. 
        `Wavefront`s with an even number of pixels can only 
        be cropped to an even number of pixels without interpolation
        and vice-versa.    
        
        Throws
        ------
        error : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[-1]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 


        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the optical disturbance zero 
            cropped to the new dimensions.
        """
        number_of_pixels_in = self.number_of_pixels()
        
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


class CartesianWavefront(Wavefront):
    """
    A plane wave extending the abstract `Wavefront` class. 
    Stores phase and amplitude arrays. pixel_scale has units of 
    meters/pixel. Assumes the wavefront is square. This is Cartesian 
    as opposed to Angular, because there are no infinite distances.  
    """
    
    def __init__(self : CartesianWavefront, wavelength : float, 
            offset : Array) -> CartesianWavefront:
        """
        Constructor for a `CartesianWavefront`.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
            
        Returns
        -------
        wavefront : CartesianWavefront
            The new wavefront with `None` at the extra leaves. 
        """
        super().__init__(wavelength, offset)
        self.plane_type = None

    
    # TODO: Ask Jordan what this is used for
    def transfer_function(self : CartesianWavefront, 
            distance : float) -> float:
        """
        The Optical Transfer Function corresponding to the 
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
    A plane wave extending the abstract `Wavefront` class. 
    Stores phase and amplitude arrays. pixel_scale has units of 
    meters per pixel in pupil planes and radians per pixel in 
    focal planes. Assumes the wavefront is square.
    """ 

    def __init__(self : AngularWavefront, wavelength : float, 
            offset : Array) -> AngularWavefront:
        """
        Constructor for `AngularWavefront`.

        Parameters
        ----------
     wavelength : float, meters
        The wavelength of the `Wavefront`.
    offset : Array, radians
        The (x, y) angular offset of the `Wavefront` from 
        the optical axis.

        Returns
        -------
        wavefront : AngularWavefront
            The new wavefront with `None` at the extra leaves. 
        """
        super().__init__(wavelength, offset)


class GaussianWavefront(Wavefront):
    """
    Expresses the behaviour and state of a wavefront propagating in 
    an optical system under the fresnel assumptions. This 
    implementation is based on the same class from the `poppy` 
    library [poppy](https://github.com/spacetelescope/poppy/fresnel.py)
    and Chapter 3 from _Applied Optics and Optical Engineering_
    by Lawrence G. N.


    Approximates the wavefront as a Gaussian Beam parameterised by the 
    radius of the beam, the phase radius, the phase factor and the 
    Rayleigh distance. Propagation is based on two different regimes 
    for a total of four different opertations. 
    
    Attributes
    ----------
    position : float, meters
        The position of the wavefront in the optical system.
    beam_radius : float, meters
        The radius of the beam. 
    phase_radius : float, unitless
        The phase radius of the gaussian beam.
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
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
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
        number_of_pixels = self.number_of_pixels()
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
