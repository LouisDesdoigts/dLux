import dLux
import jax.numpy as numpy
import pytest
import typing


Array = typing.NewType("Array", numpy.ndarray)
Wavefront = typing.NewType("Wavefront", object)
PhysicalWavefront = typing.NewType("PhysicalWavefront", Wavefront)
GaussainWavefront = typing.NewType("GaussianWavefront", PhysicalWavefront)
WavefrontUtility = typing.NewType("WavefrontUtility", object)
PhysicalWavefrontUtility = typing.NewType("PhysicalWavefrontUtility", WavefrontUtilities)
GaussianWavefrontUtility = typing.NewType("GaussianWavefrontUtility", PhysicalWavefrontUtilities)


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
            .PhysicalWavefront(self.wavelength, self.offset)
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
    

class TestWavefront(object):
    """
    Tests the Wavefront class. Tests are written looking 
    for null and nan values, with only simple end to end tests.
    """    
    utility : WavefontUtility = WavefrontUtility()

    
    # TODO: These need to be moved into the superclass test file.
    def test_get_real(self):
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        # TODO: Implement more complex test cases
        full_real_wavefront = self.set_up()
        part_real_wavefront = self.set_up(phase = Pi * self.ONES / 4)
        none_real_wavefront = self.set_up(phase = Pi * self.ONES / 2)

        assert (full_real_wavefront.get_real() == 1.).all()
        assert (part_real_wavefront.get_real() == 1. / numpy.sqrt(2)).all()
        assert (none_real_wavefront.get_real() == 0.).all()        
        

    def test_get_imag(self):
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        # TODO: Implement more complex test cases. 
        none_imaginary_wavefront = self.set_up()
        part_imaginary_wavefront = self.set_up(phase = Pi * self.ONES / 4)
        full_imaginary_wavefront = self.set_up(phase = Pi * self.ONES / 2)

        assert (full_imaginary_wavefront.get_real() == 1.).all() 
        assert (part_imaginary_wavefront.get_real() == numpy.cos(Pi/4)).all() 
        assert (none_imaginary_wavefront.get_real() == 0.).all() 


    def test_multiply_amplitude(self):
        """
        Checks that the amplitude array has been updated, after 
        operations
        """
        # TODO: Check modification by array valued inputs.
        initial_wavefront = self.set_up()
        initial_amplitude = wavefront.get_amplitude()
        changed_wavefront.multiply_ampl(Pi)
        changed_amplitude = wavefront.get_amplitude()
        
        assert (initial_amplitude == 1.).all() 
        assert (changed_amplitude == Pi).all() 
        

    def test_add_phase(self):   
        """
        Checks that the phase array is correctly updated by the 
        operations. 
        """
        # TODO: Check modification by array valued inputs.
        initial_wavefront = self.set_up()
        initial_phase = wavefront.get_phase()
        changed_wavefront = wavefront.add_phase(Pi)
        changed_phase = wavefront.get_phase()
        
        assert (initial_phase == 0.).all()
        assert (changed_phase == Pi).all()


    def test_update_phasor(self):
        """
        Checks that the phasor is correctly updated by changes to
        both the phase and the amplitude. 
        """
        # TODO: Assumes that the imputs are real
        # TODO: Implement more complex example arrays
        new_amplitude = self.ONES * Pi
        new_phase = self.ONES * Pi
        wavefront = self.set_up()
        wavefront = wavefront.update_phasor(new_amplitude, new_phase)

        assert (wavefront.get_phase() == Pi).all()
        assert (wavefront.get_amplitude() == Pi).all()
        

    def test_wavefront_to_point_spread_function(self):
        """
        Test that the point spread function is correctly generated
        from the amplitude and phase arrays. Considered correct
        if the output is the amplitude ** 2 and modifying the phase 
        does not affect the PSF
        """
        wavefront = self.set_up(amplitude = 2. * self.ONES)
        uniform_zero_phase_psf = wavefront.wf2psf()
        wavefront.add_phase(Pi)
        uniform_pi_phase_psf = wavefront.wf2psf()
        wavefront.multiply_ampl(0.5 * self.GRID_ONE)
        variable_pi_phase_psf = wavefront.wf2psf()
        
        assert (uniform_zero_phase_psf == 4.).all()
        # TODO: This may be overkill and is a little blackboxy 
        assert (unifrom_pi_phase_psf == 4.).all()
        assert (variable_pi_phase_psf == self.GRID_ONE ** 2)


    def test_add_optical_path_difference(self):
        """
        Testing for correct behaviour when optical paths length 
        distances are added. 
        """
        # So if I were to add some phase I would want to be able to 
        # interfere this beam with another
        # TODO: Raise a github issue about this functionality
        # TODO: Raise a github issue about immutability vs mutability 
        # get some real evidence as to the performance re-creating 
        # vs modifying the classes.  
        initial_wavefront = self.set_up()
        initial_phase = wavefront.get_phase()
        changed_wavefront = wavefront.add_opd(self.WAVELENGTH / 2)
        changed_phase = wavefront.get_phase()

        assert (initial_phase == 0.).all()
        assert (changed_phase == Pi / 2).all()


    def test_normalise(self):
        """
        Checks that the normalise functionality is working. The 
        behaviour is considered functional if the maximum 
        value encountered is 1. and the minimum value encountered 
        is 0.
        """
        initial_wavefront = self.set_up(amplitude = 2 * GRID_ONE * GRID_TWO)
        changed_wavefront = initial_wavefront.normalise()
        changed_amplitude = changed_wavefront.get_amplitude()

        assert (changed_amplitude.max() == 1.)
        assert (changed_amplitude.min() == 0.)
        

    def test_get_pixel_position_vector(self):
        """
        The get_xs_vec() function is consider correct if the 
        minimum value -(npix - 1) / 2 and the maximum value is
        (npix - 1) / 2
        """
        initial_wavefront = self.set_up()
        pixel_coordinates = initial_wavefront.get_xs_vec(self.SIZE)
        
        assert (pixel_coordinates.max() == (self.SIZE - 1) / 2)
        assert (pixel_coorsinates.min() == -(self.SIZE - 1) / 2)
        #TODO: implement a check that the increment between the 
        # max and min values is uniform. 


    def test_get_pixel_position_grid(self):
        """
        The get_XXYY function is considered correct if it produces 
        an array that is of dimensions (2, self.SIZE, self.SIZE)
        as it uses the get_xs_vec() function under the hood.
        """
        initial_wavefront = self.set_up()
        pixel_position_grid = initial_wavefront.get_XXYY()
        
        assert pixel_position_grid.shape() == (2, self.SIZE, self.SIZE)


    def test_get_physical_position_gird(self):
        """
        The get_xycoords() function is considered correct if it 
        produces an array with the correct dimesions such that the 
        minimum and maximum values are plus and minus
        self.pixelscale * (npix - 1) / 2
        """
        initial_wavefront = self.set_up()
        changed_wavefront = dLux.CreateWavefront(self.SIZE, 1.)\
            ({"Wavefront": initial_wavefront})["Wavefront"]
        physical_coordinates = changed_wavefront.get_xycoords()
        
        assert (physical_coordinates.max() == (self.SIZE - 1) / (2 * self.SIZE))
        assert (physical_coordinates.min() == -(self.SIZE - 1) / (2 * self.SIZE))
        assert physical_coordinates.shape() == (2, self.SIZE, self.SIZE)

    
    # TODO: implement accessors in the GaussianWavefront class.
    # Mutators and Accessors
    def test_set_phase(self):
        """
        Functionality that is not currently supported, allows the 
        state to be immutably changed and viewed from outside the 
        namespace. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_phase(self.GRID_ONE)
        
        assert (changed_wavefront.get_phase() == self.GRID_ONE).all()
        assert (initial_wavefront.get_phase() == self.PHASE).all()
 

    def test_set_amplitude(self):
        """
        Functionality that is not currently supported, provides 
        immutable access to the state of the wavefront. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_amplitude(self.GRID_ONE)
        
        assert (changed_wavefront.get_amplitude() == self.GRID_ONE).all()
        assert (initial_wavefront.get_amplitude() == self.AMPLITUDE).all()


    def test_set_beam_waist(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the intial state is not modfied and
        a modified clone is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_beam_waist(2.)
        
        assert (changed_wavefront.get_beam_waist() == 2.)
        assert (initial_wavefront.get_beam_waist() == 1.)


    def test_set_wavelength(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial wavefront is not modified 
        and a modified wavefront is created. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_wavelength(2 * self.WAVELENGTH)

        assert (changed_wavefront.get_wavelength() == 2 * self.WAVELENGTH)
        assert (initial_wavefront.get_wavelength() == self.WAVELENGTH)


    def test_set_position(self):
        """
        Provides immutable access to the state of the wavefront.
        Considered cocrect if the initial instance is not modified 
        and a modified wavefront is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_position(1.)

        assert (initial_wavefront.get_position() = 0.)
        assert (changed_wavefront.get_position() = 1.)


    def test_set_phase_radius(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial instance is not modified
        and a modified wavefront is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_phase_radius(1.)

        # TODO: Work out a good default value for this. 
        assert initial_wavefront.get_phase_radius() == 0.
        assert changed_wavefront.get_phase_radius() == 1.
       

    # Properties 
    def test_rayleigh_distance_correct(self):
        """
        Checks that the rayleigh distance is calculated correctly
        based on the parameters that are input. 
        """
        # TODO: will not work because equinox enforces immutability
        # need to check that equinox is compatible with cached
        # properties
        wavefront = self.set_up()
        rayleigh_distance = wavefront.rayleigh_distance
        correct_rayleigh_distance = numpy.pi * \
            wavefront.get_beam_radius() ** 2 / \
            wavefront.get_wavelength()        

        assert rayleigh_distance == correct_rayleigh_distance


    def test_rayleigh_distance_cached(self):
        """
        Checks that after getting called this becomes a class 
        property stored in GaussianWavefront.__dict__
        """
        wavefront = self.set_up()
        wavefront.rayleigh_distance
        
        assert wavefront.__dict__["rayleigh_distance"] not None    
    

    def test_rayleigh_distance_updated(self):
        """
        Checks that if beam_radius and wavelength are changed then 
        the rayleigh distance is also updated.
        """
        # TODO: I can enforce this behaviour using setter methods
        # or mutators. These need to be implemented. 
        initial_wavefront = self.set_up()
        initial_wavefront.rayleigh_distance
        changed_wavefront = wavefront.set_beam_waist(2.)
        changed_wavefront.rayleigh_distance
        correct_rayleigh_distance = numpy.pi * \
            changed_wavefront.get_beam_radius() ** 2 / \
            changed_wavefront.get_wavelength() 

        assert correct_rayleigh_distance = changed_wavefront.rayleigh_distance
               

    def test_location_of_waist_correct(self):
        """
        Checks that the location of the waist is correctly determined.
        Simply runs the explicit calculations externally and compares.
        """
        wavefront = self.set_up()
        correct_location_of_waist = - wavefront.get_phase_radius() / \
            (1 + (wavefront.get_phase_radius() / \
                wavefront.rayleigh_distance) ** 2)

        assert (wavefront.location_of_waist == correct_location_of_waist)


    def test_location_of_waist_cached(self):
        """
        Checks that after the state is set it is added to the cache
        as a value. 
        """
        # TODO: This could also be assigned in the constructor
        # need to discus the style with @LouisDesdoigts.
        wavefront = self.set_up()
        wavefront.location_of_waist

        assert wavefront.__dict__["location_of_waist"] not None        
        

    def test_location_of_waist_updated(self):
        """
        Checks that if the phase_radius is updated or the 
        rayleigh_distance is updated then the location_of_waist is 
        also updated
        """
        # TODO: location of waist needs to be updated when the 
        # rayleigh distance is changed, This implies that it needs 
        # to be updated when the beam_radius and the wavelength are
        # changed

        # TODO: Fix how the cached property is implemented
        initial_wavefront = self.set_up()
        initial_location_of_waist = initial_wavefront.location_of_waist
        changed_beam_radius = initial_wavefront.set_beam_radius(2.)
        changed_phase_radius = initial_wavefront.set_phase_radius(2.)
        changed_wavelength = initial_wavefront.set_wavelength(2. * self.WAVELENGTH)

        assert initial_wavefront.location_of_waist == initial_location_of_waist
        assert changed_beam_radius.location_of_waist != initial_location_of_waist 
        assert changed_phase_radius.location_of_waist != initial_location_of_waist
        assert changed_wavelength != initial_location_of_waist 


    # TODO: Implement the waist radius as a chached property 
    def test_waist_radius_correct(self):
        """
        Directly confirms that the correct numerical calculations are 
        implemented, by the method
        """
        # TODO: fix the call of rayleigh_distance() to rayleigh_distance 
        # and also beam_radius to self.beam_radius
        wavefront = self.set_up()
        correct_waist_radius = wavefront.get_beam_radius() / \
            numpy.sqrt(1 + (wavefront.rayleigh_distance / \
            wavefront.beam_radius) ** 2)

        assert wavefront.waist_radius == correct_waist_radius        


    def test_waist_radius_cached(self):
        """
        This checks that the waist radius is added to the cache after 
        getting calculated.
        """
        wavefront = self.set_up()
        wavefront.waist_radius

        assert wavefront.__dict__["waist_radius"] not None


    def test_waist_radius_updated(self):
        """
        Tests that when properties influencing the waist_radius are 
        changed so too is the waist radius.
        """
        initial_wavefront = self.set_up()
        initial_waist_radius = initial_wavefront.waist_radius
        changed_wavelength = initial_wavefront.set_wavelength(2 * self.WAVELENGTH)
        changed_beam_radius = initial_wavefront.set_beam_radius(2.)

        assert initial_waist_radius != changed_wavelength.waist_radius 
        assert initial_waist_radius != changed_beam_radius.waist_radius


    # State modifying behaviours 
    def test_planar_to_planar_not_nan(self):
        """
        So I will just test the edge cases, distance = 0. and 
        distance = numpy.inf and the typical cases distance = 1.
        and distance = 10.
        """
        # TODO: Work out what the fuck is going on with update_phasor 
        # Should this be a mutable state operation as I have treated it
        
        # TODO: Work out what the fuck is going on with update_phasor
        zero_case = self.set_up().planar_to_planar(0.)
        infinte_case = self.set_up().planar_to_planar(numpy.inf)
        small_case = self.set_up().planar_to_planar(1.)
        large_case = self.set_up().planar_to_planar(10.)

        assert not numpy.isnan(zero_case).any()
        assert not numpy.isnan(infinite_case).any()
        assert not numpy.isnan(small_case).any()
        assert not numpy.isnan(large_case).any()


    def test_planar_to_planar_not_inf(self):
        """
        So I will test the same cases as above this time looking 
        for infinite values. Not sure how to consider the case of the 
        infinite propagation distance
        """
        zero_case = self.set_up().planar_to_planar(0.)
        infinte_case = self.set_up().planar_to_planar(numpy.inf)
        small_case = self.set_up().planar_to_planar(1.)
        large_case = self.set_up().planar_to_planar(10.)

        assert not numpy.isinf(zero_case).any()
        assert not numpy.isinf(infinite_case).any()
        assert not numpy.isinf(small_case).any()
        assert not numpy.isinf(large_case).any()


    # TODO: Discuss with @benjaminpope and @LouisDesdoigts how they 
    # want to do these tests if at all. Perhaps analytic cases.
    # For now I will leave this blanck.
    def test_planar_to_planar_correct(self):
        """
        To be implemented according to the TODO:
        """
        pass


    def test_waist_to_spherical_not_nan(self):
        """
        Tests the boundary cases distance = 0. and 
        distance = self.rayleigh_distance as well as two standard 
        cases.  
        """
        rayleigh_distance = self.set_up().rayleigh_distance
        zero_case = self.set_up().waist_to_spherical(0.)
        rayleigh_case = self.set_up().waist_to_spherical(rayleigh_distance)
        small_case = self.set_up().waist_to_spherical(0.01 * rayleigh_distance)
        large_case = self.set_up().waist_to_spherical(0.9 * rayleigh_distance)

        assert not numpy.isnan(zero_case).any()
        assert not numpy.isnan(rayleigh_case).any()
        assert not numpy.isnan(small_case).any()
        assert not numpy.isnan(large_case).any()


    def test_waist_to_spherical_not_inf(self):
        """
        Checks that the boundary values and typical values defined in
        the test above do not generate infinite values
        """
        rayleigh_distance = self.set_up().rayleigh_distance
        zero_case = self.set_up().waist_to_spherical(0.)
        rayleigh_case = self.set_up().waist_to_spherical(rayleigh_distance)
        small_case = self.set_up().waist_to_spherical(0.01 * rayleigh_distance)
        large_case = self.set_up().waist_to_spherical(0.9 * rayleigh_distance)

        assert not numpy.isinf(zero_case).any()
        assert not numpy.isinf(rayleigh_case).any()
        assert not numpy.isinf(small_case).any()
        assert not numpy.isinf(large_case).any()


    def test_waist_to_spherical_correct(self):
        """
        Not yet implemented; under review
        """
        pass


    def test_spherical_to_waist_not_nan(self):
        """
        Tests the boundary cases distance = 0. and 
        distance = self.rayleigh_distance as well as two standard 
        cases for nan values 
        """
        rayleigh_distance = self.set_up().rayleigh_distance
        zero_case = self.set_up().spherical_to_waist(0.)
        rayleigh_case = self.set_up().spherical_to_waist(rayleigh_distance)
        small_case = self.set_up().spherical_to_waist(0.01 * rayleigh_distance)
        large_case = self.set_up().spherical_to_waist(0.9 * rayleigh_distance)

        assert not numpy.isnan(zero_case).any()
        assert not numpy.isnan(rayleigh_case).any()
        assert not numpy.isnan(small_case).any()
        assert not numpy.isnan(large_case).any()


    def test_spherical_to_waist_not_inf(self):
        """
        Tests the boundary cases distance = 0. and 
        distance = self.rayleigh_distance as well as two standard 
        cases for numpy.inf values 
        """
        rayleigh_distance = self.set_up().rayleigh_distance
        zero_case = self.set_up().spherical_to_waist(0.)
        rayleigh_case = self.set_up().spherical_to_waist(rayleigh_distance)
        small_case = self.set_up().spherical_to_waist(0.01 * rayleigh_distance)
        large_case = self.set_up().spherical_to_waist(0.9 * rayleigh_distance)

        assert not numpy.isinf(zero_case).any()
        assert not numpy.isinf(rayleigh_case).any()
        assert not numpy.isinf(small_case).any()
        assert not numpy.isinf(large_case).any()


    def test_spherical_to_waist_correct(self):
        """
        under review    
        """
        pass

    # State independent behavious (static)
    def test_calculate_phase_correct(self):
        """
        Checks that the phase retrieval from a complex valued array 
        is correct. I check phases of Pi / 2, 0. and Pi / 4 as well 
        as a combination.
        """
        # TODO: Type annotation for Imaginary
        pi_on_two = 1j * numpy.ones((2, 2))
        pi_on_four = 1 / numpy.sqrt(2) * numpy.ones((2, 2)) + \
            1j / numpy.sqrt(2) * numpy.ones((2, 2))
        zero = numpy.ones((2, 2))
        # TODO: probably should make this more complex
        combination = numpy.array([[1j, 1], [1, 1j]])
        wavefront = self.set_up()

        assert (wavefront.calculate_phase(pi_on_two) == Pi / 2.).all()
        assert (wavefront.calculate_phase(pi_on_four) == Pi / 4.).all()
        assert (wavefront.calculate_phase(zero) == 0.).all()
        assert (wavefront.calculate_phase(combination) == \
            numpy.array([[Pi / 2., 0.], [0., Pi / 2.]])).all()
 
 
    def test_transfer_function_not_nan(self):
        """
        Check the boundary case distance = 0. and then two normal
        inputs a large and a small.
        """
        wavefront = self.set_up()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        larger = wavefront.transfer_function(1.)

        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()


    def test_transfer_function_not_inf(self):
        """
        Checks that the boundary case distance == 0. and then two 
        normal inputs do not produce infinte results
        """
        wavefront = self.set_up()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        larger = wavefront.transfer_function(1.)

        assert not numpy.ininf(zero).any()
        assert not numpy.ininf(small).any()
        assert not numpy.ininf(large).any()


    def test_transfer_function_correct(self):
        """
        In review
        """


    def test_quadratic_phase_factor_not_nan(self):
        """
        Checks the boundary case distance == 0. for nan inputs 
        as well as a small and a large typical use case
        """
        # TODO: I think that I forgot to wrap the calculation in 
        # a numpy.exp operation
        # TODO: I need to work out what is going on with the 
        # zero and infinite cases
        # TODO: fix the type annotations
        # TODO: Make distance > 0. a precondition
        wavefront = set_up()
        zero = wavefront.quadratic_phase_factor(0.)
        infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(infinte).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()        


    def test_quadratic_phase_factor_not_inf(self):
        """
        Tests the boundary cases distance == numpy.inf and distance 
        == 0. as well as a small and a large typical case for 
        infinite values
        """
        zero = wavefront.quadratic_phase_factor(0.)
        infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(infinte).any()
        assert not numpy.isinf(small).any()
        assert not numpy.isinf(large).any()  


    def test_quadratic_phase_factor_correct(self):
        """
        In review
        """

        
    def test_pixel_scale_not_nan(self):
        """
        Checks that the new pixel scale is not generated to be nan 
        by a negative, zero and positive use case
        """
        # TODO: fix assignment currently will not work. Consider 
        # using functools.cached_property
        # TODO: self does not have npix as a field. This needs to 
        # be reviewed with @LouisDesdoigts, for now add as a field 
        # to the GaussianWavefront class
        wavefront = set_up()
        negative = wavefront.pixel_scale(-0.01)
        zero = wavefront.pixel_scale(0.)
        positive = wavefront.pixel_scale(0.01)

        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive).any()        


    def test_pixel_scale_not_inf(self):
        """
        Tests the pixel scale for infinite values when passed a 
        negative, zero and positive position.
        """
        wavefront = set_up()
        negative = wavefront.pixel_scale(-0.01)
        zero = wavefront.pixel_scale(0.)
        positive = wavefront.pixel_scale(0.01)

        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive).any()       


    def test_outside_to_outside_not_nan(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        # TODO: Implement location of waist as cached_property and 
        # fix from location_of_waist() to location_of_waist
        # TODO: Check that position is actually outside of the 
        # waist. 
        # TODO: Consult with @benjaminpope if the inf testing is 
        # neccessary 
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().outside_to_outside(-numpy.inf)
        negative = set_up().outside_to_outside(-rayleigh_distance - 0.01)
        positive = set_up().outside_to_outside(rayleigh_distance + 0.01)
        positive_infinity = set_up().outside_to_outside(numpy.inf)
        
        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        assert not numpy.isnan(positive_infinity).any()     


    def test_outside_to_outside_not_inf(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().outside_to_outside(-numpy.inf)
        negative = set_up().outside_to_outside(-rayleigh_distance - 0.01)
        positive = set_up().outside_to_outside(rayleigh_distance + 0.01)
        positive_infinity = set_up().outside_to_outside(numpy.inf)
        
        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
        assert not numpy.isinf(positive_infinity).any()     
    

    def test_outside_to_outside_correct(self):
        """
        In review
        """


    def test_outside_to_inside_not_nan(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().outside_to_inside(-numpy.inf)
        negative = set_up().outside_to_inside(-rayleigh_distance - 0.01)
        positive = set_up().outside_to_inside(rayleigh_distance + 0.01)
        positive_infinity = set_up().outside_to_inside(numpy.inf)
        
        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        assert not numpy.isnan(positive_infinity).any()     


    def test_outside_to_inside_not_inf(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().outside_to_inside(-numpy.inf)
        negative = set_up().outside_to_inside(-rayleigh_distance - 0.01)
        positive = set_up().outside_to_inside(rayleigh_distance + 0.01)
        positive_infinity = set_up().outside_to_inside(numpy.inf)
        
        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
        assert not numpy.isinf(positive_infinity).any()     
    

    def test_outside_to_inside_correct(self):
        """
        In review
        """


    def test_inside_to_outside_not_nan(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().inside_to_outside(-numpy.inf)
        negative = set_up().inside_to_outside(-rayleigh_distance - 0.01)
        positive = set_up().inside_to_outside(rayleigh_distance + 0.01)
        positive_infinity = set_up().inside_to_outside(numpy.inf)
        
        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        assert not numpy.isnan(positive_infinity).any()     


    def test_inside_to_outside_not_inf(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().inside_to_outside(-numpy.inf)
        negative = set_up().inside_to_outside(-rayleigh_distance - 0.01)
        positive = set_up().inside_to_outside(rayleigh_distance + 0.01)
        positive_infinity = set_up().inside_to_outside(numpy.inf)
        
        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
        assert not numpy.isinf(positive_infinity).any()     
    

    def test_inside_to_outside_correct(self):
        """
        In review
        """


    def test_inside_to_inside_not_nan(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().inside_to_inside(-numpy.inf)
        negative = set_up().inside_to_inside(-rayleigh_distance - 0.01)
        positive = set_up().inside_to_inside(rayleigh_distance + 0.01)
        positive_infinity = set_up().inside_to_inside(numpy.inf)
        
        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        assert not numpy.isnan(positive_infinity).any()     


    def test_inside_to_inside_not_inf(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        rayleigh_distance = set_up().rayleigh_distance
        negative_infinity = set_up().inside_to_inside(-numpy.inf)
        negative = set_up().inside_to_inside(-rayleigh_distance - 0.01)
        positive = set_up().inside_to_inside(rayleigh_distance + 0.01)
        positive_infinity = set_up().inside_to_inside(numpy.inf)
        
        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
        assert not numpy.isinf(positive_infinity).any()     
    

    def test_inside_to_inside_correct(self):
        """
        In review
        """


    # TODO: Remove the Float call from is_inside
    # TODO: Implement a light set_up() so that not all arrays are 
    # created in functions like this. 
    def test_is_inside(self):
        """
        Branch coverage for a two dimensional is_inside call. That is 
        to say:
        
        false, false
        false, true
        true, false
        true, true
        """
        wavefront = set_up()
        rayleigh_distance = wavefront.rayleigh_distance

        false_false = wavefront.is_inside(
            rayleigh_distance * numpy.ones((2, )) + 1.)
        false_true = wavefront.is_inside(
            numpy.array([rayleigh_distance + 1., 0.]))
        true_false = wavefront.is_inside(
            numpy.array([0., rayleigh_distance + 1.]))
        true_true = wavefront.is_inside([0., 0.])

        assert (false_false == numpy.array([False, False])).all()
        assert (false_true == numpy.array([False, True])).all()
        assert (true_false == numpy.array([True, False])).all()
        assert (true_true == numpy.array([True, True])).all()


    # TODO: Change the sel.is_inside([0., distance]) to use 
    # self.position instead.
    def test_propagate_not_nan(self):
        """
        Tests full branch coverage and then the boundary cases 
        distance == -numpy.inf, distance == numpy.inf and distance 
        == 0. The branches covered are:
        
        inside_to_inside
        inside_to_outside
        outside_to_inside
        outside_to_outside

        NOTE: We are checking for nan results
        """
        # TODO: Implement blank() as a minimal wavefront fixture
        # NOTE: The outside_to_inside() ect. tests already tested 
        # negative values in the typical range so we only need to 
        # consider positive ones here
        rayleigh_distance = blank().rayleigh_distance

        negative_infinity = set_up().propagate(-numpy.inf)
        zero = set_up().propagate(0.)
        positive_infinity = set_up().propagate(numpy.inf)

        inside_to_inside = set_up().propagate(rayleigh_distance / 2.)
        inside_to_outside = set_up().propagate(rayleigh_distance + 1.)
        # TODO: Adopt this syntax for future use
        outside_to_inside = \
            set_up(position = -rayleigh_distance - 1.)\
            .propagate(rayleigh_distance)
        outside_to_outside = \
            set_up(position = -rayleigh_distance - 1.)  
            .propgate(2 * (rayleigh_distance + 1.))

        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive_infinity).any()

        assert not numpy.isnan(inside_to_inside).any()
        assert not numpy.isnan(inside_to_outside).any()
        assert not numpy.isnan(outside_to_inside).any()
        assert not numpy.isnan(outside_to_outside).any()
        

    def test_propagate_not_inf(self):
        """
        Tests full branch coverage and then the boundary cases 
        distance == -numpy.inf, distance == numpy.inf and distance 
        == 0. The branches covered are:
        
        inside_to_inside
        inside_to_outside
        outside_to_inside
        outside_to_outside

        Note: we are checking for inf results. 
        """
        rayleigh_distance = blank().rayleigh_distance

        negative_infinity = set_up().propagate(-numpy.inf)
        zero = set_up().propagate(0.)
        positive_infinity = set_up().propagate(numpy.inf)

        inside_to_inside = set_up().propagate(rayleigh_distance / 2.)
        inside_to_outside = set_up().propagate(rayleigh_distance + 1.)
        # TODO: Adopt this syntax for future use
        outside_to_inside = \
            set_up(position = -rayleigh_distance - 1.)\
            .propagate(rayleigh_distance)
        outside_to_outside = \
            set_up(position = -rayleigh_distance - 1.)  
            .propgate(2 * (rayleigh_distance + 1.))

        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive_infinity).any()

        assert not numpy.isinf(inside_to_inside).any()
        assert not numpy.isinf(inside_to_outside).any()
        assert not numpy.isinf(outside_to_inside).any()
        assert not numpy.isinf(outside_to_outside).any()


    def test_propagate_correct(self):
        """
        In review
        """


# TODO: Implement a GaussianWavefrontTemplate() that contains the 
# set_up and blank fixtures for future use.
# TODO: Reduce code duplication by moving the nan and inf tests 
# together. Discuss this with @benjaminpope and @LouisDesdoigts 
class TestGaussianWavefrontAutodiff():
    """
    This class is designed to automatically test that the autodiff 
    acts correctly working with the key functions of the 
    GaussianWavefront class
    """

    # TODO: Implement blank and full set_up fixtures.
    # NOTE: I may need to make internal wrapper functions for the 
    # gradient testing. 


    def grad_transfer_function_nan(self):
        """
        Checking that the gradient of the transfer function taken with 
        respect to the following attributes does not return nan.

        Attribute tested:
         - distance : Float

        The boundary cases -numpy.inf, 0. and numpy.inf are tested
        as well as some typical cases both positive and negative. 
        """
        # TODO: Talk to @LouisDesdoigts about taking gradient with 
        # respect to the wavelength.
        transfer_function_gradient = jax.grad(set_up().transfer_function)
        rayleigh_distance = blank().rayleigh_distance()        

        # NOTE: May require some more finesse because I am not passing
        # anything for self
        negative_infinity = transfer_function_gradient(-numpy.inf)
        zero = transfer_function_gradient(0.)
        positive_infinity = transfer_function_gradient(numpy.inf)

        negative = transfer_function_gradient(- rayleigh_distance)
        positive = transfer_function_gradient(rayleigh_distance)

        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive_infinity).any()

        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()


     def grad_transfer_function_inf(self):
        """
        Checking that the gradient of the transfer function taken with 
        respect to the following attributes does not return inf.

        Attribute tested:
         - distance : Float

        The boundary cases -numpy.inf, 0. and numpy.inf are tested
        as well as some typical cases both positive and negative. 
        """
        # TODO: Talk to @LouisDesdoigts about taking gradient with 
        # respect to the wavelength.
        transfer_function_gradient = jax.grad(set_up().transfer_function)
        rayleigh_distance = blank().rayleigh_distance()        

        # NOTE: May require some more finesse because I am not passing
        # anything for self
        negative_infinity = transfer_function_gradient(-numpy.inf)
        zero = transfer_function_gradient(0.)
        positive_infinity = transfer_function_gradient(numpy.inf)

        negative = transfer_function_gradient(- rayleigh_distance)
        positive = transfer_function_gradient(rayleigh_distance)

        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive_infinity).any()

        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()


    def grad_quadratic_phase_factor_nan(self):
        """
        """
        pass 


    def grad_quadratic_phase_factor_inf(self):
        """
        """
        pass 

    
    def grad_planar_to_planar_nan(self):
        """
        """
        pass 


    def grad_planar_to_planar_inf(self):
        """
        """
        pass 

    
    def grad_waist_to_spherical_nan(self):
        """
        """
        pass 


    def grad_waist_to_spherical_inf(self):
        """
        """
        pass


    def grad_spherical_to_waist_nan(self):
        """
        """
        pass 


    def grad_spherical_to_waist_inf(self):
        """
        """
        pass


    def grad_inside_to_inside_nan(self):
        """
        """
        pass 


    def grad_inside_to_inside_inf(self):
        """
        """
        pass


    def grad_inside_to_outside_nan(self):
        """
        """
        pass


    def grad_inside_to_outside_inf(self)
        """
        """
        pass


    def grad_outside_to_inside_nan(self):
        """
        """
        pass


    def grad_outside_to_inside_inf(self):
        """
        """
        pass


    def grad_outside_to_outside_nan(self):
        """
        """
        pass 


    def grad_outside_to_outside_inf(self):
        """
        """
        pass


    def grad_propagate(self):
        """
        """
        pass
