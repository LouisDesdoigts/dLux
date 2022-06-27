import equinox 
import jax.numpy as numpy
import typing


Wavefront = typing.UserType("Wavefront", equinox.Module)
PlanaWavefront = typing.UserType("PlanarWavefront", Wavefront)
FresnelWavefront = typing.UserType("FresnelWavefront", Wavefront)
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
    wavel : float
        The wavel of the light. Assumed to be in metres.
    pixelscale : float
        The physical dimensions of each square pixel. Assumed to be 
        metres. 
    """
    amplitude : Array
    phase : Array
    wavel : float
    pixel_scale : float


    def __init__(self : Wavefront, wavel : float):
        """
        Initialises a minimal wavefront specified only by the 
        wavel. 

        Parameters
        ----------
        wavel : float 
            The monochromatic wavel associated with this 
            wavefront. 
        """
        self.wavel = wavelength # Jax Safe
        self.amplitude = None # To be instantiated by CreateWavefront
        self.phase = None # To be instantiated by CreateWavefront
        self.pixel_scale = None # To be instantiated by CreateWavefront        
 

    # TODO: Reconfirm the debate over getters. 
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
        : Array
            The real component of the optical disturbance with 
            SI units of electric field.  
        """
        return self.amplitude * numpy.cos(self.phase)
        

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
        : Array
            The imaginary component of the optical disturbance with 
            the SI units of electric field. 
        """
        return self.amplitude * numpy.sin(self.phase)


    # TODO: Name review. 
    def multiply_amplitude(self : Wavefront, 
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
        # TODO: lambda wavefront : wavefront.get_phase() over 
        # lambda : wavefront.phase
        return equinox.tree_at(lambda wavefront : wavefront.phase,
            self, self.amplitude * weights)


    def add_phase(self : Wavefront, 
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
        return equinox.tree_at(lambda wavefront : wavefront.phase,
            self, self.phase + amounts)


    def update_phasor(self : Wavefront, amplitude : Array, 
            phase : Array) -> Wavefront:  
        """
        Used to write         Throws
        ------ the state of the optical disturbance. Ignores 
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
        # TODO: Although the others do not want basix mutators and 
        # accessors point out that the body of this code could be:
        # return self.set_phase(phase).set_amplitude(amplitude)
        new_wavefront = equinox.tree_at(
            lambda wavefront : wavefront.amplitude, self, amplitude)
        new_wavefront = equinox.tree_at(
            lambda wavefront : wavefront.phase, new_wavefront, phase)
        return new_wavefront


    # TODO: Probably on the nose, taking PEP8 too far like some 
    # fanatical nerd
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
        return self.amplitude ** 2


    # TODO: Review name. More being a fanatical nerd.
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
        phase_difference = 2 * numpy.pi * path_difference / self.wavel
        return self.add_phase(phase_difference)


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
        total_intensity = numpy.norm(self.amplitude ** 2)
        return self.multiply_amplitude(1 / total_intenstiy)


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
        return numpy.arange(number_of_pixels) - (number_of_pixels - 1) / 2


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
        pixel_positions = self.get_pixel_positions(self.amplitude.shape[0])
        x_positions, y_positions = \
            numpy.meshgrid(pixel_positions, pixel_positions)
        return numpy.array([x_positions, y_positions])


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
        return self.pixel_scale * self.get_pixel_grid()


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
        # TODO: Review the syntax below with @LouisDesdoigs
        return self.invert_x().invert_y()


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
        new_amplitude = self.amplitude[:, ::-1]
        new_phase = self.phase[:, ::-1]
        return self.update_phasor(new_amplitude, new_phase)


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
        new_amplitude = self.amplitude[::-1, :]
        new_phase = self.phase[::-1, :]
        return self.update_phasor(new_amplitude, new_phase)


# TODO: Ask @LouisDesdoigts what the `plane_type` attribute is 
class PlanarWavefront(Wavefront):
    """
    A simple plane wave extending the abstract `Wavefront` class. 
    Assumes that the wavefront is square. 

    Attributes
    ----------
    offset : Array[float]
        The polarisation state of the wave described by the x and 
        y phase lag of the wavefront. This quantity is unitless and 
        it is assumed that `offset.shape == (2, )`  
    pixel_scale : float
        The physical size in meters of each pixel on the detector 
        screen. 
    plane_type : str
        The type of plane occupied by the wavefront. 
    """
    self.offset : Array
    self.plane_type : str


    def __init__(self : Wavefront, wavel : float,
            offset : Array) -> PlanarWavefront:
        """
        Parameters
        ----------
        offset : Array
            The polarisation state of the wavefront specified by the 
            x and y phase differences. 

        Returns
        -------
        : PlanarWavefront
            A minimal `PlanaWavefront` to be externally initialised
            by `CreateWavefront`. 
        """
        super().__init__(wavel)
        self.offset = offset # Jax Safe
        self.plane_type = "Pupil" # Jax Unsafe


    # TODO: Ask @LouisDesdoigts if this logic needs to be gradable.
    # and non-statically `jit`able. 
    def interpolate(self : PlanarWavefront, coordinates : Array, 
            real_imaginary : bool = False) -> tuple[Array, Array]:
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
        : tuple[Array, Array]
            The amplitude and phase of the wavefront at `coordinates`
            based on a linear interpolation.
        """
        # TODO: Review new names with @LouisDesdoigts. 
        if not real_imaginary:
            new_amplitude = map_coordinates(
                self.amplitude, coordinates, order=1)
            new_phase = map_coordinates(
                self.phase, coordinates, order=1)
        else:
            real = map_coordinates(
                self.get_real, coordinates, order=1)
            imaginary = map_coordinates(
                self.get_imaginary, coordinates, order=1)
            new_amplitude = numpy.hypot(real, imaginary)
            new_phase = numpy.arctan2(imaginary, real)
        return new_amplitude, new_phase


    def paraxial_interpolate(self : PlanarWavefront, 
            pixel_scale_out : float, number_of_pixels_out : int,
            real_imaginary : bool = False) -> PlanarWavefront: 
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
        : PlanarWavefront
            The new wavefront with the updated optical disturbance. 
        """
        # Get coords arrays
        number_of_pixels_in = self.amplitude.shape[0]
        ratio = pixel_scale_out / self.pixel_scale
        
        centre = (number_of_pixels_in - 1) / 2
        new_centre = (number_of_pixels_out - 1) / 2
        pixels = ratio * (-new_centre, new_centre, number_of_pixels_out) + centre
        # TODO: These seem to be assigned backwards. 
        y_pixels, x_pixels = numpy.meshgrid(pixels, pixels)
        coordinates = numpy.array([x_pixels, y_pixels])
        # TODO: ampl and phase are not defined here.
        new_amplitude, new_phase = self.interp(
            self.amplitude, self.phase, real_imaginary=real_imaginary)
        
        # Update Phasor
        self = self.update_phasor(new_amplitude, new_phase)
        
        # Conserve energy
        self = self.multiply_amplitude(ratio)
        
        # Update pixelscale
        # TODO: This could be a set_pixel_scale()
        return equinox.tree_at(
            lambda wavefront : wavefront.pixelscale, self, pixel_scale_out)


    # TODO: Need to review the `jit` and `grad` issues with 
    # @LouisDesdoigts and check that side effects are permitted 
    # for the function.
    # TODO: Point out that the `.at` syntax guarantees that this 
    # breaks silently when padding is smaller than wavefront. 
    def pad_to(self : PlanarWavefront, 
            number_of_pixels_out : int) -> PlanarWavefront:
        """
        Pads the `Wavefront` with zeros. Assumes that 
        `number_of_pixels_out > self.amplitude.shape[0]`. 
        Note that `Wavefronts` with even pixel dimensions can 
        only be padded (without interpolation) to even pixel 
        dimensions and vice-versa. 

        Throws
        ------
        : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[0]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 


        Returns
        -------
        : PlanarWavefront
            The new `Wavefront` with the optical disturbance zero 
            padded to the new dimensions.
        """
        number_of_pixels_in = self.amplitude.shape[0]

        # TODO: number_of_pixels_out and npix_in are not defined here.  
        if number_of_pixels_in % 2 != number_of_pixels_out % 2:
            raise ValueError("Only supports even -> even or odd -> odd padding")
        
        new_centre = number_of_pixels_out // 2
        centre = number_of_pixels_in // 2
        remainder = number_of_pixels_in % 2
        padded = numpy.zeros([number_of_pixels_out, number_of_pixels_out])
        
        new_amplitude = padded.at[
                new_centre - centre : centre + new_centre + remainder, 
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        new_phase = padded.at[
                new_centre - centre : centre + new_centre + remainder, 
                new_centre - centre : centre + new_centre + remainder
            ].set(self.phase)
        return self.update_phasor(new_amplitude, new_phase)


    def crop_to(self : PlanarWavefront, 
            number_of_pixels_out : int) -> PlanarWavefront:
        """
        Crops a `Wavefront`'s optical disturbance. Assumes that 
        `number_of_pixels_out < self.amplitude.shape[0]`. 
        `Wavefront`s with an even number of pixels can only 
        be cropped to an even number of pixels without interpolation
        and vice-versa.    
        
        Throws
        ------
        : ValueError
            If `number_of_pixels_out%2 != self.amplitude.shape[0]%2`
            i.e. padding an even (odd) `Wavefront` to odd (even).

        Parameters
        ----------
        number_of_pixels_out : int
            The square side length of the array after it has been 
            zero padded. 


        Returns
        -------
        : PlanarWavefront
            The new `Wavefront` with the optical disturbance zero 
            cropped to the new dimensions.
        """
        number_of_pixels_in = self.amplitude.shape[0]
        
        # TODO: npix_in and npix_out not defined. Does this logic work?
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


class GaussianWavefront(dLux.PhysicalWavefront):
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

    def __init__(self: FresnelWavefront, beam_radius: float, 
            wavel: float, phase_radius: float, 
            position: float=0.0) -> FresnelWavefront:
        """
        Creates a wavefront with an empty amplitude and phase 
        arrays but of a given wavel and phase offset. 

        Parameters
        ----------
        beam_radius : float
            Radius of the beam at the initial optical plane.
        wavel : float
            Wavelength of the monochromatic light.
        offset : Array
            Phase shift of the initial optical plane. 
        amplitude : Array        
            An array containing the electric field amplitudes over the 
            wavefront. Assumed to be square.
        phase : Array
            An array containing the electric field phase over the 
            wavefront. Assumed to be square.
        """
        # TODO: Is offset required in this wavefront
        super().__init__(wavel)
        self.beam_radius = beam_radius
        self.phase_radius = phase_radius
        self.amplitude = None
        self.phase = None
        self.position = position


    def rayleigh_distance(self: FresnelWavefront) -> float:
        """
        Calculates the rayleigh distance of the Gaussian beam.
        
        Returns
        -------
        : float
            The Rayleigh distance of the wavefront in metres.
        """
        return numpy.pi * self.beam_radius ** 2 / self.wavel


    def transfer_function(self: FresnelWavefront, distance: float) -> Array:
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
        : float 
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
        # TODO: Take this to a code review because I believe it is 
        # wrong. 
        # - I need to confirm that this is the correct generation of 
        # the spatial frequency variables as this is not clarified in 
        # the primary reference/ 
        # - I need to confirm that we use the distance not the position 
        coordinates = self.get_xycoords()
        radius = numpy.sqrt((coordinates ** 2).sum(axis=0))
        xi = coordinates[0, :, :] / radius / self.wavel
        eta = coordinates[1, :, :] / radius / self.wavel       
        return numpy.exp(1j * numpy.Pi * self.wavel \
            * distance * (xi ** 2 + eta ** 2))


    def quadratic_phase_factor(self: FresnelWavefront, 
            distance: float) -> Float:
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
        : float
            The near-field quadratic phase accumulated by the beam
            from a propagation of distance.
        """      
        return 1j * numpy.pi * \
            (self.get_xycoords() ** 2).sum(axis=0) \
            / self.wavel / distance


    def location_of_waist(self: FresnelWavefront) -> float:
        """
        Calculates the position of the waist along the direction of 
        propagation based of the current state of the wave.

        Returns
        -------
        : float
            The position of the waist in metres.
        """
        return - self.phase_radius / \
            (1 + (self.phase_radius / self.rayleigh_distance()) ** 2)


    def waist_radius(self: FresnelWavefront) -> float:
        """
        The radius of the beam at the waist.

        Returns
        -------
        : float
            The radius of the beam at the waist in metres.
        """
        return beam_radius / \
            numpy.sqrt(1 + (self.rayleigh_distance() / self.beam_radius) ** 2) 


    def calculate_pixel_scale(self: FresnelWavefront, position: float) -> None:
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
        number_of_pixels = self.amplitude.shape[0]
        new_pixel_scale = self.wavel * numpy.abs(position) / \
            number_of_pixels / self.pixel_scale  
        return equinox.tree_at(lambda wavefront : self.pixel_scale,
            self, new_pixel_scale)   
        

    def is_inside(self: FresnelWavefront, distance: float) -> Boolean:
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
        : Boolean
            true if the point is within the rayleigh distance false 
            otherwise.
        """
        # TODO: Review the location of this function which increases
        # coupling with the GaussianPropagator. 
        return numpy.abs(self.position + distance - \
            self.location_of_waist()) <= self.rayleigh_distance()
