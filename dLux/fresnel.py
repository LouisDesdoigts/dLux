import dLux
import typing
import jax 
import jax.numpy as np
import equinox as eqx

Scalar = typing.TypeVar("Scalar") # 0d
Vector = typing.TypeVar("Vector") # 1d
Array =  typing.TypeVar("Array") # 2d +

Wavefront   = typing.TypeVar("Wavefront")
Propagator  = typing.TypeVar("Propagator")
PlaneType   = typing.TypeVar("PlaneType")
Layer       = typing.TypeVar("Layers")


__all__ = ["GaussianWavefront", "GaussianPropagator", "GaussianLens"]
__author__ = "Jordan Dennis"


class GaussianWavefront(dLux.wavefronts.Wavefront):
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
        The *initial* radius of the beam. 
        
        
    waist_position : float, meters
        The position of the beam waist along the optical axis. 
    spherical : bool
        A convinience tracker for knowing if the wavefront is 
        currently spherical.
    rayleigh_factor : float
        Used to determine the range over which the wavefront remains
        planar. 
    focal_length : float, meters
        Used for the conversion between angular and physical units. 
    """
    angular : bool
    spherical : bool
    waist_radius : Scalar
    position : Scalar
    waist_position : Scalar
    rayleigh_factor : Scalar
    focal_length : Scalar
    
 
    def __init__(self            : Wavefront,
                 wavelength      : Scalar,
                 offset          : Vector,
                 pixel_scale     : Scalar,
                 plane_type      : PlaneType,
                 amplitude       : Array, 
                 phase           : Array,
                 waist_radius    : Scalar,
                 rayleigh_factor : Scalar = 2.) -> Wavefront:
        """
        Creates a wavefront with an empty amplitude and phase 
        arrays but of a given wavelength and phase offset. 
        Assumes that the beam starts at the waist following from 
        the `poppy` convention.

        Parameters
        ----------
        beam_radius : float, meters
            Radius of the beam at the initial optical plane.
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
        rayleigh_factor : float
            A multiplicative factor determining the threshold for 
            considering the wavefront spherical.
        """
        super().__init__(wavelength, offset, pixel_scale, plane_type,
                        amplitude, phase)
        self.waist_radius = np.asarray(waist_radius).astype(float)  
        self.position = np.asarray(0., dtype=float)
        self.waist_position = np.asarray(0., dtype=float)
        self.rayleigh_factor = np.asarray(rayleigh_factor, dtype=float)
        self.focal_length = np.inf 
        self.angular = np.asarray(False, dtype=bool)
        self.spherical = np.asarray(False, dtype=bool)

        
    ### Getters ###
    def get_position(self : Wavefront) -> float:
        """
        Accessor for the position of the wavefront. 

        Returns 
        -------
        position : float 
            The position of the `Wavefront` from its starting point 
            in meters.
        """
        return self.position
    

    def get_phase_radius(self : Wavefront) -> float:
        """
        Accessor for the phase radius of the wavefront.

        Returns
        -------
        phase_radius : float 
            The phase radius of the wavefront. This is a unitless 
            quantity.
        """
        return self.phase_radius
    
    
    def get_pixel_positions(self : Wavefront) -> Array:
        """
        Returns
        -------
        positions : Tensor, meters
            The position of each pixel aligned according to the `fft` 
            algorithm that is implemented by `numpy`.
        """
        pixels = self.number_of_pixels()
        positions = np.array(np.indices((pixels, pixels)) - pixels / 2.)
        return self.get_pixel_scale() * positions


    def is_angular(self : Wavefront) -> bool:
        """
        
        """
        return self.angular
    
    # NOTE: The pixel scale cannot be set when self.angular == True
    # NOTE: This has the correct units always/
    def get_pixel_scale(self : Wavefront):
        """
        
        NOTE - This seems dodgey, becuase if we enfore only using
        getter and setter this would result in inifinite recursion.
        Maybe this is okay though becuase it IS The getter
        
        Returns
        -------
        pixel_scale : The width of a single pixel in the array
            representing the `Wavefront`. Note that this differs
            from `poppy` because it includes the oversample. 
        """
        return jax.lax.cond(self.is_angular(),
            lambda : self.pixel_scale / self.focal_length,
            lambda : self.pixel_scale)
    
    
    def get_waist_radius(self : Wavefront):
        """
        Calculates the current waist radius as half of the diameter
        
        Mainly just exists to keep consistency with textbook terms
        """
        return self.waist_radius


    def rayleigh_distance(self : Wavefront) -> Scalar:
        """
        Calculates the rayleigh distance of the Gaussian beam.
        
        Returns
        -------
        rayleigh_distance : float
            The Rayleigh distance of the wavefront in metres.
        """
        return np.pi * self.get_waist_radius() ** 2 / self.get_wavelength()
    
    
    ### Setters ###
    # NOT USED
    def set_position(self : Wavefront, 
            position : float) -> Wavefront:
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
        wavefront : Wavefront
            This wavefront at the new position. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.position, self, position,
            is_leaf = lambda leaf : leaf is None)
    
    
    # NOT UESD
    def set_phase_radius(self : Wavefront, 
            phase_radius : float) -> Wavefront:
        """
        Mutator for the phase_radius.

        Parameters
        ----------
        phase_radius : float
            The new phase_radius in meters.

        Returns
        -------
        wavefront : Wavefront
            A modified Wavefront with the new phase_radius.
        """
        return eqx.tree_at(lambda wavefront : wavefront.phase_radius, 
            self, phase_radius, is_leaf = lambda leaf : leaf is None)
    
    
    # NOTE: ordering convention: dunder, _..., ..._at, ..._after, 
    # set_... get_...
    # NOTE: naming conventself._outside_to_outsideion: position -> absolute place on optical
    # axis and distance -> movement.
    def set_waist_position(self : Wavefront, waist_position : Scalar) -> Wavefront:
        """
        Parameters
        ----------
        waist_position : float, meters 
            The absolute position of the waist of the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The `Wavefront` with the updated parameters.  
        """
        return eqx.tree_at(lambda wave : wave.waist_position,
            self, waist_position)    
    

    def set_waist_radius(self : Wavefront, waist_radius : Scalar) -> Wavefront:
        """
        Parameters
        ----------
        beam_radius : float
            The new beam_radius in meters.

        Returns
        -------
        wavefront : Wavefront
            A modified `Wavefront` with the new `beam_radius`.
        """
        return eqx.tree_at(lambda wave : wave.waist_radius,
            self, waist_radius)


    def set_spherical(self : Wavefront, spherical : bool) -> Wavefront:
        """
        Parameters
        ----------
        spherical : bool
            Toggle the state of the `Wavefront` to and from `spherical`.
        
        Returns
        -------
        wavefront : Wavefront 
            The `Wavefront` with the parameters modified. 
        """
        
        # Pretty sure the 'spherical' input need to wrapped into a jnp array
        return eqx.tree_at(lambda wave : wave.spherical, self, spherical)
    

    # TODO: Do I even want to include this functionality as it is 
    # only for the translation between the angular and the physical
    # coordinates. I'm not sure that we need this. 
    def set_focal_length(self : Wavefront, focal_length : Scalar) -> Wavefront:
        """
        Parameters
        ----------
        focal_length : float, meters
            The `focal_length` of the `Wavefront`.

        Returns
        -------
        wavefront : Wavefront
            The `Wavefront` with the parameters modified. 
        """
        return eqx.tree_at(lambda wave : wave.focal_length,
            self, focal_length)


    # NOT USED
    def calculate_pixel_scale(self: Wavefront, position: float) -> None:
        """
        The pixel scale at the position along the axis of propagation.

        Parameters
        ----------
        position : float
            The position of the wavefront along the axis of propagation
            in metres.
        """
        new_pixel_scale = self.get_wavelength() * np.abs(position) / \
            self.number_of_pixels() / self.get_pixel_scale()  
        return new_pixel_scale 
        
    
    # NOT USED
    def is_inside(self: Wavefront, distance: float) -> bool:
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
            self.waist_position) <= self.rayleigh_distance()


    # NOTE: This also needs an ..._after name. I could use something
    # like quadratic_phase_after() or phase_after() 
    def quadratic_phase(self : Wavefront, distance : Scalar) -> Array:
        """
        Convinience function that simplifies many of the diffraction
        equations. Caclulates a quadratic phase factor associated with 
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
        positions = self.get_pixel_positions()
        rho_squared = (positions ** 2).sum(axis = 0) 
        return np.exp(1.j * np.pi * rho_squared / distance /\
            self.wavelength)


    def transfer(self : Wavefront, distance : Scalar) -> Array:
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
        """
        positions = self.get_pixel_positions()
        x, y = positions[0], positions[1]
        rho_squared = \
            (x / (self.get_pixel_scale() ** 2 \
                * self.number_of_pixels())) ** 2 + \
            (y / (self.get_pixel_scale() ** 2 \
                * self.number_of_pixels())) ** 2
        # Transfer Function of diffraction propagation eq. 22, eq. 87
        return np.exp(-1.j * np.pi * self.wavelength * \
                distance * rho_squared)


    def curvature_at(self : Wavefront, position : float) -> float:
        """
        Calculate the radius of curvature of the `Wavefront` phase
        at the absolute position: `position`.

        Parameters
        ----------
        position : float, meters
            The absolute position of the wave along the optical axis 
            from spawn.

        Returns
        -------
        radius_of_curvature : float, radians
            The radius of phase curvature for the wavefront. 
        """
        relative_position = position - self.waist_position
        return relative_position + \
            self.rayleigh_distance() ** 2 / relative_position


    def radius_at(self : Wavefront, position : Scalar) -> Scalar:
        """
        Calculate the radius of the `Wavefront` at an absolute
        position.

        Parameters
        ----------
        position : float, meters
            The absolute position of the `Wavefront` since spawn.
        
        Returns
        -------
        radius : float, meters
            The radius of the beam. 
        """
        relative_position = position - self.waist_position
        return self.get_waist_radius() * \
            np.sqrt(1.0 + \
                (relative_position / self.rayleigh_distance()) ** 2)

    
    def is_planar_at(self : Wavefront, position : Scalar) -> bool:
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
        return np.abs(self.waist_position - position) \
            < self.rayleigh_distance()

    
    # NOTE: Also updates, so I want better names for these rather than 
    # after. 
    # NOTE: This is only for transitions from planar to spherical 
    # or vice versa so it needs a much better name than current. 
    def pixel_scale_after(self : Wavefront, distance : float) -> Wavefront:
        """
        Calculate and assign the pixel scale of the `Wavefront` after
        travelling distance. Note that this transformation is dependent
        on the mode of propagation and is only correct for
        `_spherical_to_waist` and `_waist_to_spherical` but not for 
        `_plane_to_plane`. 

        Parameters
        ----------
        distance : float, meters
            The distance of propagation from the current position.
        
        Returns
        -------
        wavefront : Wavefront
            The wavefront but the pixel_scale has been updated.
        """
        pixel_scale = self.get_wavelength() * np.abs(distance) /\
            (self.number_of_pixels() * self.get_pixel_scale())
        return self.set_pixel_scale(pixel_scale)

    
    def position_after(self : Wavefront, 
            distance : Scalar) -> Wavefront:
        """
        Move the wavefront forward by `distance`.

        Parameters
        ----------
        distance : float, meters
            The distance of propagation.
        
        Returns
        -------
        wavefront : Wavefront 
            The `Wavefront` with the `position` leaf updated.
        """
        position = self.position + distance
        return eqx.tree_at(lambda wave : wave.position, self, position)



class GaussianPropagator(eqx.Module):
    """
    Represents the propagation of a FarFieldFresnel wavefront some distance.

    Parameters
    ----------
    distance: float, meters
        The propagation distance.
    """
    distance : float


    def __init__(
            self: Propagator, 
            distance: float) -> Propagator:
        """
        Parameters
        ----------
        distance: float, meters
            The propagation distance. 
        """
        self.distance = np.asarray(distance).astype(float)


    def _fourier_transform(
            self: Propagator, 
            field: Array) -> Array:
        """
        Calculate the normalised fourier transform of an array.

        Parameters
        ----------
        field: Array
            The electric field of the wavefront (complex).

        Returns
        -------
        field: Array
            The fourier transform of the complex electric field. 
        """
        return 1 / field.shape[0] * np.fft.fft2(field)


    def _inverse_fourier_transform(
            self : Propagator, 
            field : Array) -> Array:
        """
        Calculate the normalised inverse fourier transform. 

        Parameters
        ----------
        field: Array
            The complex electric field of the wavefront. 

        Returns
        -------
        field: Array
            The inverse fourier transform of the electric field. 
        """
        return field.shape[0] * np.fft.ifft2(field)


    # NOTE: need to add in the standard FFT normalising factor
    # as in the propagator. 
    def _propagate(
            self : Propagator, 
            field : Array, 
            distance : float) -> Array:
        """
        The name in this case is a misnomer since we are not making the 
        Fraunhofer approximation. In this case it has just been used for 
        consistency within the package.

        Parameters
        ----------
        field: Array 
            The complex electric field of the wavefront. 
        distance: float, meters
            The distance of propagation. 

        Returns 
        -------
        field: Array
            Either the inverse or fourier transform of the wavefront depending 
            on the sign of the distance. 
        """
        return jax.lax.cond(distance > 0,
            lambda : self._inverse_fourier_transform(field),
            lambda : self._fourier_transform(field))
        

    def _plane_to_plane(
            self : Propagator, 
            wavefront : Wavefront,
            distance : float) -> Wavefront:
        """
        Propagate from within the planar regime to within the planar regime. 
        
        Parameters
        ----------
        wavefront: Wavefront
            The wavefront to propagate.
        distance: float, meters
            The distance to propagate the wavefront. 

        Returns 
        -------
        wavefront: Wavefront 
            The wavefront after it has been propagated. 
        """
        field = self._fourier_transform(wavefront.get_complex_form())
        field *= np.fft.fftshift(wavefront.transfer(distance))  # eq. 6.68
        field = self._inverse_fourier_transform(field)
        
        wavefront =  wavefront\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))

        return wavefront
 


    def _waist_to_spherical(
            self : Propagator, 
            wavefront : Wavefront, 
            distance : float) -> Wavefront:
        """
        Propagate from within the planar regime to within the spherical regime. 
        
        Parameters
        ----------
        wavefront: Wavefront
            The wavefront to propagate.
        distance: float, meters
            The distance to propagate the wavefront. 

        Returns 
        -------
        wavefront: Wavefront 
            The wavefront after it has been propagated. 
        """ 
        # Lawrence eq. 83,88
        field = wavefront.get_complex_form()
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance)) # Wavelength dependent
        field = self._propagate(field, distance) # Wrapper for forwards/reverse FFTs
        wavefront =  wavefront\
            .pixel_scale_after(distance)\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)   
        return wavefront


    # Wavefront.spherical must be True initially
    def _spherical_to_waist(
            self : Propagator, 
            wavefront : Wavefront,
            distance : float) -> Wavefront:
        """
        Propagate from within the spherical regime to within the planar regime. 
        
        Parameters
        ----------
        wavefront: Wavefront
            The wavefront to propagate.
        distance: float, meters
            The distance to propagate the wavefront. 

        Returns 
        -------
        wavefront: Wavefront 
            The wavefront after it has been propagated. 
        """
        # Lawrence eq. 89
        field = wavefront.get_complex_form()
        field = self._propagate(field, distance)
        wavefront = wavefront.pixel_scale_after(distance)
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance)) # Wavelength dependent
        
        wavefront = wavefront\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)\
            .position_after(distance)
        return wavefront


    def _inside_to_inside(
            self : Propagator, 
            wave : Wavefront) -> Wavefront:
        """
        Propagate from behind the beam waist to behind the beam waist. 

        Parameters
        ----------
        wave: Wavefront
            The wavefront to propagate.
        
        Returns
        -------
        wave: Wavefront
            The propagated wavefront. 
        """        
        wave = self._plane_to_plane(wave, self.distance)
        return wave


    def _inside_to_outside(
            self : Propagator, 
            wave : Wavefront) -> Wavefront: 
        """
        Propagate from behind the beam waist to ahead of the beam waist. 

        Parameters
        ----------
        wave: Wavefront
            The wavefront to propagate.
        
        Returns
        -------
        wave: Wavefront
            The propagated wavefront. 
        """        
        start = wave.position
        end = wave.position + self.distance
        wave = self._plane_to_plane(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        return wave


    def _outside_to_inside(
            self : Propagator, 
            wave : Wavefront) -> Wavefront:
        """
        Propagate from ahead of the beam waist to behind the beam waist. 

        Parameters
        ----------
        wave: Wavefront
            The wavefront to propagate.
        
        Returns
        -------
        wave: Wavefront
            The propagated wavefront. 
        """        
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._plane_to_plane(wave, end - wave.waist_position)
        return wave


    def _outside_to_outside(
            self : Propagator, 
            wave : Wavefront) -> Wavefront:
        """
        Propagate from behind the beam waist to behind the next beam waist. 

        Parameters
        ----------
        wave: Wavefront
            The wavefront to propagate.
        
        Returns
        -------
        wave: Wavefront
            The propagated wavefront. 
        """        
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        return wave


    def __call__(
            self: Propagator, 
            parameters: dict) -> Wavefront:
        """
        Propagate a wavefront in the Fresnel regime.

        Parameters
        ----------
        parameters: dict
            A dictionary of parameters containing a "Wavefront" field. 
    
        Returns
        -------
        parameters: dict
            The same dictionary of parameters with the "Wavefront" field
            updated. 
        """ 
        field = np.fft.fftshift(wave.get_complex_form())
        wave = wave.update_phasor(np.abs(field), np.angle(field))
        position = wave.position + self.distance
        decision = 2 * wave.spherical + wave.is_planar_at(position)
        
        wave = jax.lax.switch(
            decision,
            [self._inside_to_outside, self._inside_to_inside, 
            self._outside_to_outside, self._outside_to_inside], wave) 

        field = np.fft.fftshift(wave.get_complex_form())
        wave = wave.update_phasor(np.abs(field), np.angle(field))        
        return wave


class GaussianLens(eqx.Module):
    """
    This is a special lens that interacts with the GaussianWavefront allowing
    the internal parameters of the GaussianBeam to be updated. 

    Parameters
    ----------
    focal_length: float
        The focal length of the lens. 
    """
    focal_length : float


    def __init__(
            self: Layer, 
            focal_length: float) -> Layer:
        """
        Parameters
        ----------
        focal_length: float, meters 
            The focal length of the lens. 
        """
        self.focal_length = np.asarray(focal_length).astype(float)


    def _phase(
            self : Layer, 
            wave : Wavefront, 
            distance : float) -> Array:
        """
        Generate the quadratic phase that is associated with the lens. 

        Parameters
        ----------
        wave: Wavefront 
            The wavefront. This is used to generate the pixel coordinates 
            for consistency. 
        distance: float, meters
            I do not know how to effectively describe this. It is the 
            effective focal length created by the curvature of the wavefront 
            interacting with the curvature of the lens. 

        Returns 
        -------
        phase: Array
            The phase change of the wavefront from passing through the lens.
        """
        position = wave.get_pixel_positions()
        rho_squared = (position ** 2).sum(axis = 0)
        return np.exp(1.j / np.pi * rho_squared / distance *\
            wave.wavelength)


    def __call__(
            self: Layer, 
            parameters: dict) -> Wavefront:
        """
        Interact the lens with the wavefront.  

        Parameters
        ----------
        parameters: dict
            A dictionary of parameters containing a "Wavefront" field. 
    
        Returns
        -------
        parameters: dict
            The same dictionary of parameters with the "Wavefront" field
            updated. 
        """ 
        from_waist = wave.waist_position - wave.position
        was_spherical = np.abs(from_waist) > wave.rayleigh_factor * \
            wave.rayleigh_distance()

        curve = wave.curvature_at(wave.position)

        curve_at = jax.lax.cond(
            was_spherical,
            lambda : from_waist,
            lambda : np.inf)

        curve_after = jax.lax.cond(
            was_spherical,
            lambda : 1. / (1. / curve  - 1. / self.focal_length),
            lambda : -self.focal_length)

        radius = wave.radius_at(wave.position)
        curve_ratio = (wave.wavelength * curve_after / np.pi / radius ** 2) ** 2
        curve_matched = curve == self.focal_length

        waist_position_after = jax.lax.cond(
            curve_matched,
            lambda : wave.position,
            lambda : -curve_after / (1. + curve_ratio) + wave.position)

        waist_radius_after = jax.lax.cond(
            curve_matched,
            lambda : radius,
            lambda : radius / np.sqrt(1. + 1. / curve_ratio))

        focal_length_after = jax.lax.cond(
            np.isinf(wave.focal_length),
            lambda : self.focal_length,
            lambda : curve_after / curve_at * wave.focal_length)

        wave = wave\
            .set_focal_length(focal_length_after)\
            .set_waist_radius(waist_radius_after)\
            .set_waist_position(waist_position_after)

        from_new_waist = waist_position_after - wave.position
        is_spherical = np.abs(from_new_waist) > wave.rayleigh_distance()
 
        distance = 1. / jax.lax.cond(
            wave.spherical,
            lambda : jax.lax.cond(
                is_spherical,
                lambda : 1. / self.focal_length + 1. / from_new_waist - 1. / curve_at,
                lambda : 1. / self.focal_length - 1. / curve_at),
            lambda : jax.lax.cond(
                is_spherical, 
                lambda : 1. / self.focal_length + 1. / from_new_waist,
                lambda : 1. / self.focal_length))

        field = wave.get_complex_form() * self._phase(wave, distance)
        phase = np.angle(field)
        amplitude = np.abs(field)

        return wave\
            .set_phase(phase)\
            .set_amplitude(amplitude)\
            .set_spherical(is_spherical)\
            .set_waist_position(waist_position_after)\
            .set_waist_radius(waist_radius_after)\
            .set_focal_length(focal_length_after)

