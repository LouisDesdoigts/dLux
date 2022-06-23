import dLux
import jax 
import jax.numpy as numpy
import equinox
import typing
import functools

# Type annotations
Array = numpy.array
Float = typing.Union[numpy.float64, numpy.float32, numpy.float16]
Integer = typing.Union[numpy.int64, numpy.int32, numpy.int16, numpy.int8]
Complex = typing.Union[numpy.complex128, numpy.complex64]
Boolean = numpy.bool_
Real = typing.Union[Integer, Float]
Imaginary = 1j 
Wavefront = dLux.PhysicalWavefront
FresnelWavefront = typing.NewType("FresnelWavefront", Wavefront)


class FresnelWavefront(dLux.PhysicalWavefront):
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
    """
    # Dunder attributes
    # TODO: Test and profile a working slots implementation.
    # __slots__ = ("INDEX_GENERATOR", "position", "wavelength")

    # Constants
    INDEX_GENERATOR = numpy.array([1, 2])

    # Variables 
    phase_radius: Float
    beam_radius: Array
    position: Float


    def __init__(self: FresnelWavefront, beam_radius: Array, 
            wavelength: Float, offset: Array, amplitude: Array, 
            phase: Array, phase_radius: Float) -> FresnelWavefront:
        """
        Creates a wavefront with an empty amplitude and phase 
        arrays but of a given wavelength and phase offset. 

        Parameters
        ----------
        beam_radius : float 
            Radius of the beam at the initial optical plane.
        wavelength : float
            Wavelength of the monochromatic light.
        offset : ndarray
            Phase shift of the initial optical plane.         
        """
        super(wavelength, offset)
        self.beam_radius = beam_radius
        self.amplitude = amplitude
        self.phase = phase
        self.phase_radius = phase_radius
        # TODO: Determine if I need to implement some method of 
        # best fit for the phase_radius and beam_radius. 


    @functools.cached_property
    def rayleigh_distance(self: FresnelWavefront) -> None:
        """
        Calculates the rayleigh distance $z_{R}$ of the Gaussian beam.
        """
        self.rayleigh_distance = numpy.pi * self.beam_radius ** 2 \
            /self.wavelength


    def planar_to_planar(self: FresnelWavefront, 
            distance: Float) -> None:
        """
        Modifies the state of the wavefront by propagating a planar 
        wavefront to a planar wavefront. 

        Parameters
        ----------
        distance : Float
            The distance of the propagation in metres.
        """
        wavefront = self.amplitude * \
            numpy.exp(Imaginary * self.phase)

        new_wavefront = numpy.fft.ifft2(
            self.transfer_function(distance) * \
            numpy.fft.fft2(wavefront))

        amplitude = numpy.abs(new_wavefront)
        phase = self.calculate_phase(new_wavefront)
        self.update_phasor(amplitude, phase)        


    def waist_to_spherical(self: FresnelWavefront, 
            distance: Float) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        the waist of the gaussian beam to a spherical wavefront. 

        Parameters
        ----------
        distance : Float 
            The distance of the propagation in metres.
        """
        coefficient = 1 / Imaginary / self.wavelength / distance

        wavefront = self.amplitude * numpy.exp(Imaginary * self.phase)

        # TODO: Check that these are the correct algorithms to 
        # use and ask if we are going to actually require the 
        # reverse direction
        fourier_transform = jax.lax.cond(numpy.sign(distance) > 0, 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                numpy.fft.fft2(wavefront), 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                numpy.fft.ifft2(wavefront),
            wavefront, distance)

        new_wavefront = coefficient * fourier_transform
        phase = self.calculate_phase(new_wavefront)
        amplitude = jax.numpy.abs(new_wavefront)
        self.update_phasor(amplitude, phase)

        # TODO: Confirm that I need to update the pixel scale for 
        # this transformation.


    def spherical_to_waist(self: FresnelWavefront, 
            distance: Float) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        a spherical wavefront to the waist of the Gaussian beam. 

        Parameters
        ----------
        distance : Float
            The distance of propagation in metres
        """
        coefficient = 1 / Imaginary / self.wavelength / distance * \
            quadratic_phase_factor(distance)

        wavefront = self.amplitude * numpy.exp(Imaginary * self.phase)

        # TODO: Check that these are the correct algorithms to 
        # use and ask if we are going to actually require the 
        # reverse direction
        fourier_transform = jax.lax.cond(numpy.sign(distance) > 0, 
            lambda wavefront: numpy.fft.fft2(wavefront), 
            lambda wavefront: numpy.fft.ifft2(wavefront),
            wavefront)

        new_wavefront = coefficient * fourier_transform
        phase = self.calculate_phase(new_wavefront)
        amplitude = jax.numpy.abs(new_wavefront)
        self.update_phasor(amplitude, phase)

        # TODO: Confirm that I need to update the pixel scale for 
        # this transformation.


    def calculate_phase(self: FresnelWavefront, wavefront: Complex) -> Real:
        """
        The phase of the wavefront.

        Parameters
        ----------
        wavefront : Array[Complex]
            A complex array to retrieve the phases from.
        
        Returns
        -------
        : Array[Real]
            An array of phases calculated from the wavefront
        """
        return numpy.arccos(numpy.real(wavefront / numpy.abs(wavefront)))

    
    def transfer_function(self: FresnelWavefront, distance: Float) -> Array:
        """
        The optical transfer function (OTF) for the gaussian beam.
        Assumes propagation is along the axis. 

        Parameters
        ----------
        distance : Float
            The distance to propagate the wavefront along the beam 
            via the optical transfer function in metres.

        Returns
        -------
        : Float 
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
        xi = coordinates[0, :, :] / radius / self.wavelength
        eta = coordinates[1, :, :] / radius / self.wavelength       
        return numpy.exp(Imaginary * numpy.Pi * self.wavelength \
            * distance * (xi ** 2 + eta ** 2))


    def quadratic_phase_factor(self: FresnelWavefront, 
            distance: Float) -> Float:
        """
        Convinience function that simplifies many of the diffraction
        equations. Caclulates a quadratic phase factor associated with 
        the beam. 

        Parameters
        ----------
        distance : Float
            The distance of the propagation measured in metres. 

        Returns
        -------
        : Float
            The near-field quadratic phase accumulated by the beam
            from a propagation of distance.
        """      
        return Imaginary * numpy.pi * \
            (self.get_xycoords() ** 2).sum(axis=0) \
            / self.wavelength / distance


    @functools.cached_property
    def location_of_waist(self: FresnelWavefront) -> Float:
        """
        Calculates the position of the waist along the direction of 
        propagation based of the current state of the wave.

        Returns
        -------
        : Float
            The position of the waist in metres.
        """
        return - self.phase_radius / \
            (1 + (self.phase_radius / self.rayleigh_distance()) ** 2)


    def waist_radius(self: FresnelWavefront) -> Float:
        """
        The radius of the beam at the waist.

        Returns
        -------
        : Float
            The radius of the beam at the waist in metres.
        """
        return beam_radius / \
            numpy.sqrt(1 + (self.rayleigh_distance() / self.beam_radius) ** 2) 


    def pixel_scale(self: FresnelWavefront, position: Float) -> None:
        """
        The pixel scale at the position along the axis of propagation.
        Assumes that the wavefront is square. That is:
        ```
        x, y = self.amplitude.shape
        (x == y) == True
        ```

        Parameters
        ----------
        position : Float
            The position of the wavefront aling the axis of propagation
            in metres.
        """
        self.pixelscale = self.wavelength * numpy.abs(position) / \
            self.npix / self.pixelscale         


    def outside_to_outside(self: FresnelWavefront, distance: Float) -> None:
        """
        Propagation from outside the Rayleigh range to another 
        position outside the Rayleigh range. 

        Parameters
        ----------
        distance : Float
            The distance to propagate in metres.
        """
        self.waist_to_spherical(self.position + distance - \
            self.location_of_waist()) * self.waist_to_spherical(
            self.location_of_waist() - self.position)


    def outside_to_inside(self: FresnelWavefront, distance: Float) -> None:
        """
        Propagation from outside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        distance : Float
            The distance to propagate in metres.
        """
        self.planar_to_planar(self.position + distance - \
            self.location_of_waist()) * self.spherical_to_waist(
            self.location_of_waist() - self.position)


    def inside_to_inside(self: FresnelWavefront, distance: Float) -> None:
        """
        Propagation from inside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        distance : Float
            The distance to propagate in metres.
        """
        # Consider removing after checking the Jaxpr for speed. 
        # This is just an alias for another function. 
        self.planar_to_planar(distance)


    def inside_to_outside(self: FresnelWavefront, distance: Float) -> None:
        """
        Propagation from inside the Rayleigh range to outside the 
        rayleigh range. 

        Parameters
        ----------
        distance : Float
            The distance to propagate in metres.
        """
        self.waist_to_spherical(self.position + distance - \
            self.location_of_waist()) * planar_to_planar(
            self.location_of_waist() - self.position)


    @functools.partial(jax.vmap, in_axes=(None, 0))
    def is_inside(self: FresnelWavefront, distance: Float) -> Boolean:
        """ 
        Determines whether a point at along the axis of propagation 
        at distance away from the current position is inside the 
        rayleigh distance. 

        Parameters
        ----------
        distance : Float
            The distance to test in metres.

        Returns
        -------
        : Boolean
            true if the point is within the rayleigh distance false 
            otherwise.
        """
        return Float(numpy.abs(self.position + distance - \
            self.location_of_waist()) <= self.rayleigh_distance())
        

    def propagate(self: FresnelWavefront, distance: Float) -> None:
        """
        Propagates the wavefront approximated by a Gaussian beam 
        the amount specified by distance. Note that distance can 
        be negative.

        Parameters 
        ----------
        distance : Float
            The distance to propagate in metres.
        """
        # This works by considering the current position and distnace 
        # as a boolean array. The INDEX_GENERATOR converts this to 
        # and index according to the following table.
        #
        # sum((0, 0) * (1, 2)) == 0
        # sum((1, 0) * (1, 2)) == 1
        # sum((0, 1) * (1, 2)) == 2
        # sum((1, 1) * (1, 2)) == 3
        #
        # TODO: Test if a simple lookup is faster. 
        decision_vector = self.is_inside([0., distance])
        decision_index = numpy.sum(self.INDEX_GENERATOR * descision_vector)
 
        # Enters the correct function differentiably depending on 
        # the descision.
        jax.lax.switch(decision_index, 
            [self.inside_to_inside, self.inside_to_outside,
            self.outside_to_inside, self.outside_to_outside],
            distance) 
