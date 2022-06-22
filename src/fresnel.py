import dLux
import jax 
import jax.numpy as numpy
import equinox
import typing

# Type annotations
Array = numpy.array
Float64 = numpy.float64
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
    rayleigh_distance: Float64
    phase_radius: Float64
    beam_radius: Array
    position: Float64


    def __init__(self: FresnelWavefront, beam_radius: Array, 
            wavelength: Float64, offset: Array, amplitude: Array, 
            phase: Array) -> FresnelWavefront:
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
        self.rayleigh_distance = rayleigh_distance(beam_radius, wavelength)
        # Need to work out if the gaussian wavefront is modifying 
        # an existing wavefront. 
        # self.amplitude = global_phase(beam_radius, wavelength)
        # self.phase_radius = phase_radius(beam_radius, wavelength) 


    def rayleigh_distance(self: FresnelWavefront) -> None:
        """
        Calculates the rayleigh distance $z_{R}$ of the Gaussian beam.
        """
        self.rayleigh_distance = numpy.pi * self.beam_radius ** 2 \
            /self.wavelength


    def planar_to_planar(self: FresnelWavefront, 
            distance: Float64) -> None:
        """
        Modifies the state of the wavefront by propagating a planar 
        wavefront to a planar wavefront. 

        Parameters
        ----------
        distance : Float64
            The distance of the propagation in metres.
        """


    def waist_to_spherical(self: FresnelWavefront, 
            distance: Float64) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        the waist of the gaussian beam to a spherical wavefront. 

        Parameters
        ----------
        distance : Float64 
            The distance of the propagation in metres.
        """


    def spherical_to_waist(self: FresnelWavefront, 
            distance: Float64) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        a spherical wavefront to the waist of the Gaussian beam. 

        Parameters
        ----------
        distance : Float64
            The distance of propagation in metres
        """

    
    def transfer_function(self: FresnelWavefront, 
            distance: Float64) -> Float64:
        """
        The optical transfer function (OTF) for the gaussian beam.

        Parameters
        ----------
        distance : Float64
            The distance to propagate the wavefront along the beam 
            via the optical transfer function in metres.

        Returns
        -------
        : Float64 
            A phase representing the evolution of the wavefront over 
            the distance. 
        """


    def quadratic_phase_factor(self: FresnelWavefront, 
            radius: Float64, distance: Float64) -> Float64:
        """
        Convinience function that simplifies many of the diffraction
        equations. Caclulates a quadratic phase factor associated with 
        the beam. 

        Parameters
        ----------
        radius : Float64 
            The cylindrical radius at the current point of propagation
            measured in metres.
        distance : Float64
            The distance of the propagation measured in metres. 

        Returns
        -------
        : Float64
            The near-field quadratic phase accumulated by the beam
            from a propagation of distance.
        """


    def location_of_waist(self: FresnelWavefront) -> Float64:
        """
        Calculates the position of the waist along the direction of 
        propagation based of the current state of the wave.

        Returns
        -------
        : Float64
            The position of the waist in metres.
        """
        return - self.phase_radius / \
            (1 + (self.phase_radius / self.rayleigh_distance()) ** 2)


    def waist_radius(self: FresnelWavefront) -> Float64:
        """
        The radius of the beam at the waist.

        Returns
        -------
        : Float64
            The radius of the beam at the waist in metres.
        """
        return beam_radius / \
            numpy.sqrt(1 + (self.rayleigh_distance() / beam_radius) ** 2) 


    def pixel_scale(self: FresnelWavefront, position: Float64) -> Float64:
        """
        The pixel scale at the position along the axis of propagation.
        Assumes that the wavefront is square. That is:
        ```
        x, y = self.amplitude.shape
        (x == y) == True
        ```

        Parameters
        ----------
        position : Float64
            The position of the wavefront aling the axis of propagation
            in metres.

        Returns
        -------
        : Float64
            The pixel scale at position.
        """


    def outside_to_outside(self: FresnelWavefront, start: Float64, 
            finish: Float64) -> None:
        """
        Propagation from outside the Rayleigh range to another 
        position outside the Rayleigh range. 

        Parameters
        ----------
        start : Float64
            The initial position of the wavefront in metres.
        finish : Float64
            The final position of the wavefront in metres.
        """


    def outside_to_inside(self: FresnelWavefront, start: Float64,
            finish: Float64) -> None:
        """
        Propagation from outside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        start : Float64
            The initial position of the wavefront in metres.
        finish : Float64
            The final position of the wavefront in metres.
        """


    def inside_to_inside(self: FresnelWavefront, start: Float64,
            finish: Float64) -> None:
        """
        Propagation from inside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        start : Float64
            The initial position of the wavefront in metres.
        finish : Float64
            The final position of the wavefront in metres.
        """


    def inside_to_outside(self: FresnelWavefront, start: Float64,
            finish: Float64) -> None:
        """
        Propagation from inside the Rayleigh range to outside the 
        rayleigh range. 

        Parameters
        ----------
        start : Float64
            The initial position of the wavefront in metres.
        finish : Float64
            The final position of the wavefront in metres.
        """


    def propagate(self: FresnelWavefront, distance: Float64) -> None:
        """
        Propagates the wavefront approximated by a Gaussian beam 
        the amount specified by distance. Note that distance can 
        be negative.

        Parameters 
        ----------
        distance : Float64
            The distance to propagate in metres.
        """
        is_current_position_inside = numpy.abs(self.position - self.rayleigh_distance())
        is_
        


    
