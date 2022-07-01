"""
tests/propagators.py
--------------------
Contains the unit tests for the `Propagator` objects coded in the 
`propagators.py` file. Tests are comprehensive unit tests based of 
the `UtilityUser` framework stored in the `tests/utilities.py` file.


Tests are organised one `TestX` class per `PropagatorX` class in 
`src/propagators.py` and can be run in the terminal using 
`pytest tests/propagators.py`. It is recommended that the `--quiet`
or `-q` flag is used to supress extra stack traces.
"""
__author__ = "Jordan Dennis"
__date__ = "02/07/2022"


import pytest
import jax.numpy as numpy
import dLux
import typing
from .utilities.py import *


Tester = typing.NewType("Tester", UtilityUser)


class TestPropagator(UtilityUser):
    """
    Tests the concrete methods of the abstract propagator class.
    The concrete methods are:
     - __init__()
     - _get_pixel_grid()
     - _get_pixel_positions()
     - is_inverse()

    Attributes
    ----------
    utility : PropagatorUtility
    """
    utility : PropagatorUtility = PropagatorUtility()


    def test_constructor(self : Tester) -> None:
        """
        Tests that the constructor correctly initialialises the 
        un-traced `inverse` attribute of the `Propagator` class.
        """
        false_propagator = self\
            .get_utility()\
            .construct(inverse = False)

        true_propagator = self\
            .get_utility()\
            .construct(inverse = True)

        assert false_constructor.is_inverse() == False
        assert true_propagator.is_inverse() == True


    def test_is_inverse(self : Tester) -> None:
        """
        Makes sure that the `is_inverse` function correctly reflects 
        the state of the `Propagator`
        """
        true_propagator = self\
            .get_utility()\
            .construct(inverse = True)

        false_propagator = self\
            .get_utility()\
            .construct(inverse = False)

        assert true_propagator.is_inverse() == True
        assert false_propagator.is_inverse() == False


    def test_get_pixel_grid(self : Tester) -> None:
        """
        Checks that the pixel grid is generated correctly in the 
        output plane. The implementation is considered correct if
        it has the correct maximum and minimum values and is the 
        correct shape.
        """
        utility = self.get_utility()

        pixel_grid = utility\
            .construct()\
            ._get_pixel_grid(
                utility.get_pixel_offset(), 
                utility.get_pixel_scale(), 
                utility.get_number_of_pixels())

        assert pixel_grid.max() == \
            utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.min() == \
            - utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.shape == (2, utility.get_number_of_pixels(),
            utility.get_number_of_pixels)        


    def test_get_pixel_positions(self : Tester) -> None:
        """
        Checks that the pixel positions are correctly generated.
        The pixel positions are considered correct if they have 
        the correct maximum, minimum and shape.
        """    
        utility = self.get_utility()

        pixel_positions = utility\
            .construct()\
            ._get_pixel_positions()    

        assert pixel_positions.max() == \
            utility.get_nuber_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.min() == \
            - utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.shape == (2, utility.get_number_of_pixels(),
            utility.get_number_of_pixels)    


class TestVariableSamplingPropagator(UtilityUser):
    """
    Tests the concrete methods of the `VariableSamplingPropagator`
    abstract class. The concrete methods are:
     - __init__()
     - _matrix_fourier_transform()
     - _fourier_transform()
     - _inverse_fourier_transform()
     - _normalising_factor()
     - get_pixel_scale_out()
     - get_pixels_out()
     - generate_twiddle_factors()

    Attributes
    ----------
    utility : VariableSamplingUtility
        A testing utility that generates safe test cases.

    Inherits
    --------
    : UtilityUser
        Provides acces to the `get_utitlity` method, broadcast 
        accross all test classes.
    """
    utility : VariableSamplingUtility = VariableSamplingUtility()


    def test_constructor(self : Tester) -> None:
        """
        Checks that all the appropriate fields are correctly 
        initialised by the constructor. These fields are:
         - pixel_scale_out : float
         - pixels_out : int
         - inverse : bool
        """
        propagator = self.get_utility().construct()

        assert propagator.is_inverse() == utility.get_inverse()
        assert propagator.get_pixels_out() == utility.get_pixels_out()
        assert propagator.get_pixel_scale_out() == \
            utility.get_pixel_scale_out()


    def test_get_pixel_scale_out(self : Tester) -> None:
        """
        Tests that the `get_pixel_scale_out` method correctly 
        reports the state of the object. 
        """
        OLD_PIXEL_SCALE = 0.7
        NEW_PIXEL_SCALE = 0.5

        old_propagtor = self\
            .get_utility()\
            .construct(pixel_scale_out = OLD_PIXEL_SCALE)

        new_propagator = self\
            .get_utility()\
            .construct(pixel_scale_out = NEW_PIXEL_SCALE)

        assert old_propagator.get_pixel_scale_out() == OLD_PIXEL_SCALE
        assert new_propagator.get_pixel_scale_out() == NEW_PIXEL_SCALE


    def test_get_pixels_out(self : Tester) -> None:
        """
        Tests that `get_pixels_out` correctly tracks the state of 
        the object. 
        """
        NEW_PIXELS = 100
        OLD_PIXELS = 50

        old_propagator = self\
            .get_utility()\
            .construct(pixels_out = OLD_PIXELS)

        new_propagator = self\
            .get_utility()\
            .construct(pixels_out = NEW_PIXELS)

        assert old_propagator.get_pixels_out() == OLD_PIXELS
        assert new_propagator.get_pixels_out() == NEW_PIXELS


    def test_generate_twiddle_factors(self : Tester) -> None:
        """
        Tests that the correct twiddle factors are getting 
        generated. This is done by direct comparison for a 
        simple case.
        """
        TWIDDLE_FACTORS : Array = 0.5 * numpy.array([
            [1. + 0.j, 1. + 0.j, 1. + 0.j, 1. + 0.j],
            [1. + 0.j, 0. - 1.j, -1. + 0.j, 0. + 1.j],
            [1. + 0.j, -1. + 0.j, 1. + 0.j, -1. + 0.j],
            [1. + 0.j, 0. + 0.j, -1. + 0.j, 0. - 1.j]])
        OFFSET : Array = numpy.array([2., 2.])
        PIXELS : int = 4
        PIXEL_SCALES : float = 1.
        SIGN : int = 1

        twiddle_factors = self\
            .get_utility()\
            .construct()\
            ._generate_twiddle_factors(OFFSETS, PIXEL_SCALES, 
                PIXESL, SIGN)

         is_correct = self\
            .get_utility()\
            .approx(twiddle_factors, TWIDDLE_FACTORS)\
            .all()

        assert is_correct


    def test_matrix_fourier_transform(self : Tester) -> None:
        """
        Tests that the discrete foruier transform is correct for
        a simple case. 
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = 1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = 1))\
            .all()

        assert is_correct


    def test_fourier_transform(self : Tester) -> None:
        """
        Tests that the Fourier transform of the wavefront is 
        correctly calculated. 
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = 1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = 1))\
            .all()

        assert is_correct

    def test_inverse_fourier_transform(self : Tester) -> None:
        """
        Tests that the inverse fourier transform is correctly 
        calculated.
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = -1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = -1))\
            .all()

        assert is_correct


class TestFixedSamplingPropagator(UtilityUser):
    def test_fourier_transform(self : Tester) -> None:
    def test_inverse_fourier_transfrom(self : Tester) -> None:
    def test_normalising_factor(self : Tester) -> None:


class TestPhysicalMFT(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_normalising_factor(self : Tester) -> None:
        """
        Tests that the normalising factor is correctly implemented. 
        """


    def test_get_focal_length(self : Tester) -> None:
    def test_number_of_fringes(self : Tester) -> None:
    def test_get_pixel_offsets(self : Tester) -> None:
    def test_propagate(self : Tester) -> None:
    def test_physical_mft(self : Tester) -> None:


class TestPhysicalFFT(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_get_focal_length(self : Tester) -> None:   
    def test_get_pixel_scale_out(self : Tester) -> None:
    def test_propagate(self : Tester) -> None:
    def test_physical_fft(self : Tester) -> None:


class TestPhysicalFresnel(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_get_focal_length(self : Tester) -> None:
    def test_number_of_fringes(self : Tester) -> None:
    def test_quadratic_phase(self : Tester) -> None:
    def test_thin_lens(self : Tester) -> None:
    def test_propagate(self : Tester) -> None:
    def test_physical_fresnel(self : Tester) -> None:


class TestAngularMFT(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_number_of_fringes(self : Tester) -> None:
    def test_get_pixel_offsets(self : Tester) -> None:
    def test_propagate(self : Tester) -> None:
    def test_angular_mft(self : Tester) -> None:


class TestAngularFFT(UtilityUser):
    def test_constructor(self : Tester) -> None:
    def test_pixel_scale_out(self : Tester) -> None:


class TestAngularFresnel(UtilityUser):
    pass

class TestGaussianPropagator(object):
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


    def test_outside_to_outside_not_nan(self):
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
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
