import dLux
import typing
import jax 
import jax.numpy as np
import equinox as eqx


Propagator = typing.TypeVar("Propagator")
Wavefront = typing.TypeVar("Wavefront")
Layer = typing.TypeVar("Layer")
Matrix = typing.TypeVar("Matrix")
Tensor = typing.TypeVar("Tensor")
Vector = typing.TypeVar("Vector")


class GaussianWavefront(dLux.Wavefront):
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
    waist_radius : float, meters
        The radius of the beam. 
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
    waist_radius : float 
    position : float
    waist_position : float
    rayleigh_factor : float
    focal_length : float
    
 
    def __init__(self : Wavefront,
            offset : Vector,
            wavelength : float,
            beam_radius : float,
            rayleigh_factor : float = 2.) -> Wavefront:
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
            A multiplicative factor determining the threeshild for 
            considering the wavefront spherical.
        """
        super(GaussianWavefront, self).__init__(wavelength, offset)
        self.waist_radius = np.asarray(beam_radius).astype(float)  
        self.position = np.asarray(0.).astype(float)
        self.waist_position = np.asarray(0.).astype(float)
        self.rayleigh_factor = np.asarray(rayleigh_factor).astype(float)
        self.focal_length = np.inf 
        self.angular = False
        self.spherical = False


    # NOTE: This also needs an ..._after name. I could use something
    # like quadratic_phase_after() or phase_after() 
    def quadratic_phase(self : Wavefront, distance : float) -> Matrix:
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
        positions = self.get_pixel_positions()
        rho_squared = (positions ** 2).sum(axis = 0) 
        return np.exp(1.j * np.pi * rho_squared / distance /\
            self.wavelength)


    # NOTE: This is plane_to_plane transfer function. I should give it
    # a better name like fraunhofer_phase_after()
    def transfer(self : Wavefront, distance : float) -> Matrix:
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
            (x / (self.pixel_scale ** 2 \
                * self.number_of_pixels())) ** 2 + \
            (y / (self.pixel_scale ** 2 \
                * self.number_of_pixels())) ** 2
        # Transfer Function of diffraction propagation eq. 22, eq. 87
        return np.exp(-1.j * np.pi * self.wavelength * \
                distance * rho_squared)


    def rayleigh_distance(self : Wavefront) -> float:
        """
        Calculates the rayleigh distance of the Gaussian beam.
        
        Returns
        -------
        rayleigh_distance : float
            The Rayleigh distance of the wavefront in metres.
        """
        return np.pi * self.waist_radius ** 2 / self.wavelength


    # NOTE: The pixel scale cannot be set when self.angular == True
    # NOTE: This has the correct units always/
    def get_pixel_scale(self : Wavefront):
        return jax.lax.cond(self.angular,
            lambda : self.pixel_scale / self.focal_length,
            lambda : self.pixel_scale)


    # NOTE: Should only be called when self.angular == True
    def field_of_view(self):
        return self.number_of_pixels() * self.get_pixel_scale()


    # NOTE: naming convention. ..._at indicates absolute position
    # ..._after indicates a distance from current position. 
    # either should make all the same or be clear. 
    def curvature_at(self : Wavefront, position : float) -> float:
        relative_position = position - self.waist_position
        return relative_position + \
            self.rayleigh_distance() ** 2 / relative_position


    def radius_at(self : Wavefront, position : float) -> float:
        relative_position = position - self.waist_position
        return self.waist_radius * \
            np.sqrt(1.0 + \
                (relative_position / self.rayleigh_distance()) ** 2)
           
 
    def is_planar_at(self : Wavefront, position : float) -> bool:
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
        pixel_scale = self.get_wavelength() * np.abs(distance) /\
            (self.number_of_pixels() * self.get_pixel_scale())
        return eqx.tree_at(lambda wave : wave.pixel_scale,
            self, pixel_scale, is_leaf = lambda leaf : leaf is None)


    def position_after(self : Wavefront, distance : float) -> Wavefront:
        position = self.position + distance
        return eqx.tree_at(lambda wave : wave.position, self, position)

    
    # NOTE: ordering convention: dunder, _..., ..._at, ..._after, 
    # set_... get_...
    # NOTE: naming conventself._outside_to_outsideion: position -> absolute place on optical
    # axis and distance -> movement.
    def set_waist_position(self : Wavefront, waist_position : float) -> Wavefront:
        return eqx.tree_at(lambda wave : wave.waist_position,
            self, waist_position)    
    

    def set_waist_radius(self : Wavefront, waist_radius : float) -> Wavefront:
        return eqx.tree_at(lambda wave : wave.waist_radius,
            self, waist_radius)


    def set_spherical(self : Wavefront, spherical : bool) -> Wavefront:
        return eqx.tree_at(lambda wave : wave.spherical, self, spherical)
    #def set_angular(self : Wavefront, angular : bool) -> Wavefront:
    # NOTE: focal_length will probably not stay as an attribute of the 
    # wavefront but will be upgraded to an optical element attribute.
    def set_focal_length(self : Wavefront, focal_length : float) -> Wavefront:
        return eqx.tree_at(lambda wave : wave.focal_length,
            self, focal_length)


    def get_pixel_positions(self : Wavefront) -> Tensor:
        pixels = self.phase.shape[0]
        positions = np.array(np.indices((pixels, pixels)) - pixels / 2.)
        return self.pixel_scale * positions


class GaussianPropagator(eqx.Module):
    distance : float

    
    def __init__(self : Propagator, distance : float):
        self.distance = np.asarray(distance).astype(float)


    def _fourier_transform(self : Propagator, field : Matrix) -> Matrix:
        # return np.fft.ifft2(field)
        return 1 / field.shape[0] * np.fft.fft2(field)


    def _inverse_fourier_transform(self : Propagator, field : Matrix) -> Matrix:
        # return np.fft.fft2(field)
        return field.shape[0] * np.fft.ifft2(field)


    # NOTE: need to add in the standard FFT normalising factor
    # as in the propagator. 
    def _propagate(self : Propagator, field : Matrix, 
            distance : float) -> Matrix:
        # NOTE: is this diagnosable directly from the stored parameter
        # would be nice if the "transfer" function could be automatically
        # chosen from the "spherical" and one other thing. 
        # should probably avoid logic overcrowding
        return jax.lax.cond(distance > 0,
            lambda : self._inverse_fourier_transform(field),
            lambda : self._fourier_transform(field))
            

    # NOTE: Wavefront must be planar 
    # NOTE: Uses eq. 82, 86, 87
    def _plane_to_plane(self : Propagator, wavefront : Wavefront,
            distance : float):
        # NOTE: Seriously need to change the name to get_field()
        field = self._fourier_transform(wavefront.get_complex_form())
        field *= np.fft.fftshift(wavefront.transfer(distance))  # eq. 6.68
        field = self._inverse_fourier_transform(field)
        # NOTE: wavefront.from_field is looking good right about now
        return wavefront\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))
 

    # NOTE: I'm thinking that the logic for repacking the wavefront
    # should occur somewhere else although I guess that it can't really
    # NOTE: Must start with a planar wavefront
    def _waist_to_spherical(self : Propagator, wavefront : Wavefront, 
            distance : float) -> Wavefront:
        # Lawrence eq. 83,88
        field = wavefront.get_complex_form()
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance))
        field = self._propagate(field, distance)
#        field = jax.lax.cond(
#            distance > 0, 
#            lambda field : np.fft.ifft2(field),
#            lambda field : np.fft.fft2(field),
#            field) 
        pixel_scale = wavefront.pixel_scale_after(distance) 
        return wavefront\
            .pixel_scale_after(distance)\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)       


    # Wavefront.spherical must be True initially
    def _spherical_to_waist(self : Propagator, wavefront : Wavefront,
            distance : float) -> Wavefront:
        # Lawrence eq. 89
        field = wavefront.get_complex_form()
        field = self._propagate(field, distance)
#        field = jax.lax.cond(
#            distance > 0, 
#            lambda field : np.fft.ifft2(field),
#            lambda field : np.fft.fft2(field),
#            field) 
        wavefront = wavefront.pixel_scale_after(distance)
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance))
        return wavefront\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)\
            .position_after(distance)


    def _inside_to_inside(self : Propagator, wave : Wavefront) -> Wavefront:
#        field = wave.get_complex_form()
#        field = np.fft.fftshift(field)
#        wave = wave.update_phasor(np.abs(field), np.angle(field)) 
        wave = self._plane_to_plane(wave, self.distance)
        return wave
#        field = wave.get_complex_form()
#        field = np.fft.fftshift(field)
#        return wave.update_phasor(np.abs(field), np.angle(field))


    def _inside_to_outside(self : Propagator, wave : Wavefront) -> Wavefront: 
        start = wave.position
        end = wave.position + self.distance
        wave = self._plane_to_plane(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        # TODO: This may belong between the plane to plane and the 
        # waist_to_spherical
#        field = np.fft.fftshift(wave.get_complex_form())
#        wave = wave.update_phasor(np.abs(field), np.angle(field))
        return wave


    def _outside_to_inside(self : Propagator, wave : Wavefront) -> Wavefront:
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._plane_to_plane(wave, end - wave.waist_position)
#        field = np.fft.fftshift(wave.get_complex_form())
#        wave = wave.update_phasor(np.abs(field), np.angle(field))
        return wave


    def _outside_to_outside(self : Propagator, wave : Wavefront) -> Wavefront:
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        return wave


    # NOTE: So I could attempt to move all of the functionality into 
    # the wavefront class and do very little here. Damn, I need to 
    # fit it into the overall architecture. 
    # TODO: Implement the oversample in the fixed sampling propagator
    # Coordiantes must be in meters for the propagator
    def __call__(self : Propagator, wave : Wavefront) -> Wavefront:
        # NOTE: need to understand this mystery. 
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
    focal_length : float
    # TODO: Should this store its position in the optical system?
    # No I don't think that it should. 

    def __init__(self : Layer, focal_length : float) -> Layer:
        self.focal_length = np.asarray(focal_length).astype(float)


    def _phase(self : Layer, wave : Wavefront, 
            distance : float) -> Matrix:
        position = wave.get_pixel_positions()
        rho_squared = (position ** 2).sum(axis = 0)
        return np.exp(1.j / np.pi * rho_squared / distance *\
            wave.wavelength)


    def __call__(self : Layer, wave : Wavefront) -> Wavefront:
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


#    def apply_image_plane_fftmft(self, optic):
#        """
#        Apply a focal plane mask using fft and mft methods to highly sample at the focal plane.
#        
#        Parameters
#        ----------
#        optic : FixedSamplingImagePlaneElement
#        """
#        _log.debug("------ Applying FixedSamplingImagePlaneElement using FFT and MFT sequence ------")
#        
#        # readjust pixelscale to wavelength being propagated
#        fpm_pxscl_lamD = ( optic.pixelscale_lamD * optic.wavelength_c.to(u.meter) / self.wavelength.to(u.meter) ).value 
#
#        # get the fpm phasor either using numexpr or numpy
#        scale = 2. * np.pi / self.wavelength.to(u.meter).value
#        if accel_math._USE_NUMEXPR:
#            _log.debug("Calculating FPM phasor from numexpr.")
#            trans = optic.get_transmission(self)
#            opd = optic.get_opd(self)
#            fpm_phasor = ne.evaluate("trans * exp(1.j * opd * scale)")
#        else:
#            _log.debug("numexpr not available, calculating FPM phasor with numpy.")
#            fpm_phasor = optic.get_transmission(self) * np.exp(1.j * optic.get_opd(self) * scale)
#        
#        nfpm = fpm_phasor.shape[0]
#        n = self.wavefront.shape[0]
#        
#        nfpmlamD = nfpm*fpm_pxscl_lamD*self.oversample
#
#        mft = poppy.matrixDFT.MatrixFourierTransform(centering=optic.centering)
#
#        self.wavefront = accel_math._ifftshift(self.wavefront)
#        self.wavefront = accel_math.fft_2d(self.wavefront, forward=False, fftshift=True) # do a forward FFT to virtual pupil
#        self.wavefront = mft.perform(self.wavefront, nfpmlamD, nfpm) # MFT back to highly sampled focal plane
#        self.wavefront *= fpm_phasor
#        self.wavefront = mft.inverse(self.wavefront, nfpmlamD, n) # MFT to virtual pupil
#        self.wavefront = accel_math.fft_2d(self.wavefront, forward=True, fftshift=True) # FFT back to normally-sampled focal plane
#        self.wavefront = accel_math._fftshift(self.wavefront)
#        
#        _log.debug("------ FixedSamplingImagePlaneElement: " + str(optic.name) + " applied ------")
#
#
#    def _resample_wavefront_pixelscale(self, detector):
#        """ Resample a Fresnel wavefront to a desired detector sampling.
#        The interpolation is done via the scipy.ndimage.zoom function, by default
#        using cubic interpolation.  If you wish a different order of interpolation,
#        set the `.interp_order` attribute of the detector instance.
#        Parameters
#        ----------
#        detector : Detector class instance
#            Detector that defines the desired pixel scale
#        Returns
#        -------
#        The wavefront object is modified to have the appropriate pixel scale and spatial extent.
#        """
#
#        if self.angular:
#            raise NotImplementedError("Resampling to detector doesn't yet work in angular coordinates for Fresnel.")
#
#        pixscale_ratio = (self.pixelscale / detector.pixelscale).decompose().value
#
#        if np.abs(pixscale_ratio - 1.0) < 1e-3:
#            _log.debug("Wavefront is already at desired pixel scale "
#                       "{:.4g}.  No resampling needed.".format(self.pixelscale))
#            self.wavefront = utils.pad_or_crop_to_shape(self.wavefront, detector.shape)
#            return
#
#        super(FresnelWavefront, self)._resample_wavefront_pixelscale(detector)
#
#        self.n = detector.shape[0]

