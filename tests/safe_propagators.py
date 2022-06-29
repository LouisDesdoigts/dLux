import pytest
import jax.numpy as numpy
import dLux
from .utilities.py import *


def TestGaussianPropagator(object):
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
