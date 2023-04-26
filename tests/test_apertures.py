import dLux 
import jax
import jax.numpy as np
import typing
from jax import Array
import pytest


jax.config.update("jax_debug_nans", True)

Aperture = typing.TypeVar("Aperture")
Spider = typing.TypeVar("Spider")


class BaseApertureTest():
    def _assert_valid_hard_aperture(aperture, msg=''):
        assert ((aperture == 1.) | (aperture == 0.)).all(), msg
        
    def _assert_valid_soft_aperture(aperture, msg=''):
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all(), msg
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all(), msg
        assert np.logical_and(aperture >= 0., aperture <= 1.).any(), msg

    def _test_aperture(self, aperture_fixture, kwargs, create_wavefront):
        
        # Test Constructors
        npixels = 16
        msg = f'Args: {kwargs}, on constructor {aperture_fixture}'
        aperture = aperture_fixture(**kwargs)
        diameter = 3*aperture._extent().max()

        # Test Transmission
        transmission = aperture.get_transmission(npixels, diameter)
        if kwargs['softening'] == 0.:
            TestAperturesCommonInterfaces._assert_valid_hard_aperture(transmission, msg)
        else:
            TestAperturesCommonInterfaces._assert_valid_soft_aperture(transmission, msg)
        
        # Test on Wavefront
        wf = aperture(create_wavefront(diameter=diameter))


class TestAperturesCommonInterfaces(BaseApertureTest):
    """
    For each type of aperture that has common properties, test it
    """
    # input is all the fixtures that need testing
    def test_all_apertures(self, 
        create_square_aperture : callable,
        create_rectangular_aperture : callable,
        create_circular_aperture : callable,
        create_hexagonal_aperture : callable,
        create_annular_aperture : callable,
        create_wavefront : callable,
        create_irregular_polygonal_aperture : callable):
        
        constructors = [create_square_aperture,
                        create_rectangular_aperture,
                        create_circular_aperture,
                        create_hexagonal_aperture,
                        create_annular_aperture,
                        create_irregular_polygonal_aperture,]
        
        # TODO might need to add error message for when this fails but it checks things very easily
        for ctor in constructors:
            self._test_single_aperture_class(ctor, create_wavefront)

    def _test_single_aperture_class(self, aperture_fixture, create_wavefront):

        # Non-rotatable
        not_rotatable_apertures = (dLux.apertures.CircularAperture,
                                   dLux.apertures.AnnularAperture)

        # Parameter inputs
        centres = [np.array([0., 0.]),
                   np.array([.1, .1])]
        rotations = [0, np.pi/3.]
        softenings = [0., 1.]
        occultings = [True, False]
        normalises = [False, True]

        kwargs = {}
        for centre in centres:
            for rotation in rotations:
                for softening in softenings:
                    for occulting in occultings:
                        for normalise in normalises:
                            kwargs["centre"] = centre
                            kwargs["softening"] = softening
                            kwargs["occulting"] = occulting
                            kwargs["normalise"] = normalise
                            
                            if not isinstance(aperture_fixture(),
                                not_rotatable_apertures):
                                kwargs["rotation"] = rotation
                            
                            self._test_aperture(aperture_fixture, kwargs,
                                create_wavefront)


# TODO: All aperture types need to be promoted to an aberrated in order to test _extent
class TestUniformSpider(object):
    """
    Contains the unit tests for the `UniformSpider` class.
    """


    def test_constructor(self, create_uniform_spider: callable) -> None:
        """
        Tests that the state is correctly initialised. 
        """
        # Test functioning
        spider = create_uniform_spider()

    
    def test_range_hard(self, create_uniform_spider: callable) -> None:
        """
        Checks that the apertures fall into the correct range.
        """
        npix = 16
        width = 2.
        pixel_scale = width / npix
        coordinates = dLux.utils.get_pixel_positions((npix,npix), 
                                                    (pixel_scale, pixel_scale))

        # Case Translated 
        spider = create_uniform_spider(centre=[1., 1.], softening=0.)
        aperture = spider._transmission(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Rotated 
        spider = create_uniform_spider(rotation=np.pi/4., softening=0.)
        aperture = spider._transmission(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Strained 
        spider = create_uniform_spider(shear=[.05, .05], softening=0.)
        aperture = spider._transmission(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Compression
        spider = create_uniform_spider(compression=[1.05, .95], softening=0.)
        aperture = spider._transmission(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()


    def test_range_soft(self, create_uniform_spider: callable) -> None:
        """
        Checks that the aperture falls into the correct range.
        """
        npix = 16
        width = 2.
        pixel_scale = width / npix
        coordinates = dLux.utils.get_pixel_positions((npix,npix), 
                                                    (pixel_scale, pixel_scale))

        # Case Translated 
        spider = create_uniform_spider(centre=[1., 1.], softening=0.)
        aperture = spider._transmission(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Rotated 
        spider = create_uniform_spider(rotation=np.pi/4., softening=0.)
        aperture = spider._transmission(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Strained 
        spider = create_uniform_spider(shear=[.05, .05], softening=0.)
        aperture = spider._transmission(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Compression
        spider = create_uniform_spider(compression=[1.05, .95], softening=0.)
        aperture = spider._transmission(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()


class TestAberratedAperture(object):
    """
    Checks that the aberrated aperture is functional. It does not
    test whether the aberrated aperture is correct.
    """


    def test_constructor(self, 
        create_aberrated_aperture: callable,
        create_circular_aperture: callable,
        create_static_aperture: callable) -> None:
        """
        Tests that it is possible to instantiate an AberratedAperture.
        Does not test if the AberratedAperture is correct.
        """
        # TODO: Make sure that the class asserts that the coeffs and 
        # the noll indexes have the same length.
        ap = create_aberrated_aperture()

        with pytest.raises(ValueError):
            occulting_ap = create_circular_aperture(occulting=True)
            create_aberrated_aperture(aperture=occulting_ap)
        
        with pytest.raises(TypeError):
            static_ap = create_static_aperture(npixels=16, diameter=1)
            create_aberrated_aperture(aperture=static_ap)
    
        with pytest.raises(dLux.exceptions.DimensionError):
            create_aberrated_aperture(noll_inds=np.arange(1, 7), 
                coefficients=np.zeros(3))


    def test_on_aperture(self: object, 
            create_aberrated_aperture: callable,
            create_circular_aperture: callable,
            create_wavefront: callable,
            create_hexagonal_aperture: callable,
            create_irregular_polygonal_aperture: callable) -> None:
        """
        Tests that the basis functions are evaluated atop the aperture.
        Applies mutliple different permutations.
        """
        npixels = 16

        # Test valid construction
        ap = create_circular_aperture()
        diameter = 3*ap._extent().max()
        aber_ap = create_aberrated_aperture(aperture=ap)
        wf = aber_ap(create_wavefront(diameter=diameter))

        # Test with normalise
        ap = create_circular_aperture(normalise=True)
        aber_ap = create_aberrated_aperture(aperture=ap)
        wf = aber_ap(create_wavefront(diameter=diameter))

        # Test Transmission
        transmission = aber_ap.get_transmission(npixels, diameter)
        basis = aber_ap.get_basis(npixels, diameter)
        opd = aber_ap.get_opd(npixels, diameter)

        # Regular Polygon Aperture (via hexagonal subclass)
        ap = create_hexagonal_aperture()
        aber_ap = create_aberrated_aperture(aperture=ap)
        diameter = 3*ap._extent().max()
        basis = aber_ap.get_basis(npixels, diameter)

        # Irregular Polygon Aperture
        ap = create_irregular_polygonal_aperture()
        aber_ap = create_aberrated_aperture(aperture=ap)
        diameter = 3*ap._extent().max()
        basis = aber_ap.get_basis(npixels, diameter)


class TestCompoundAperture():

    def test_constructor(self,
        create_compound_aperture: callable,
        create_aberrated_aperture: callable,
        create_circular_aperture: callable,
        create_static_aperture: callable) -> None:
        ap1 = create_circular_aperture()
        ap2 = create_circular_aperture(radius = 0.1, occulting=True)

        # Test valid construction
        comp_ap = create_compound_aperture([ap1, ap2])

        # Test with single aberrated
        aber_ap = create_aberrated_aperture()
        create_compound_aperture([aber_ap])

        # Test with multiple aberrated
        with pytest.raises(TypeError):
            create_compound_aperture([aber_ap, aber_ap])

        # Test with CompoundAperture
        with pytest.raises(TypeError):
            create_compound_aperture([comp_ap])
        
        # Test with non aperture type
        with pytest.raises(TypeError):
            create_compound_aperture([1.])

        # Test with static apertire
        with pytest.raises(TypeError):
            static_ap = create_static_aperture()
            create_compound_aperture([static_ap])


    def test_on_aperture(self,
        create_compound_aperture: callable,
        create_circular_aperture: callable,
        create_aberrated_aperture: callable,
        create_wavefront: callable):
        npixels = 16

        # Test applying
        ap = create_circular_aperture()
        diameter = 3*ap._extent().max()
        aber_ap = create_aberrated_aperture(aperture=ap)
        comp_ap = create_compound_aperture([aber_ap])
        wf = comp_ap(create_wavefront(diameter=diameter))

        # Test with normalise
        comp_ap = create_compound_aperture([ap], normalise=True)
        wf = comp_ap(create_wavefront(diameter=diameter))

        # Test methods
        transmission = comp_ap.get_transmission(npixels, diameter)
        coefficients = comp_ap.coefficients
        basis = comp_ap.get_basis(npixels, diameter)
        opd = comp_ap.get_opd(npixels, diameter)

        # Test Basis with no aberrated
        ap = create_circular_aperture()
        comp_ap = create_compound_aperture([ap])
        basis = comp_ap.get_basis(npixels, diameter)
        opd = comp_ap.get_opd(npixels, diameter)

        # Test getattr
        ap = create_circular_aperture()
        comp_ap = create_compound_aperture([ap])
        comp_ap.CircularAperture

        with pytest.raises(AttributeError):
            comp_ap._


class TestMultiAperture():

    def test_constructor(self,
        create_multi_aperture: callable,
        create_aberrated_aperture: callable,
        create_circular_aperture: callable,
        create_static_aperture: callable) -> None:
        ap1 = create_circular_aperture()
        ap2 = create_circular_aperture(radius = 0.1, occulting=True)

        # Test valid construction
        multi_ap = create_multi_aperture([ap1, ap2])

        # Test with single aberrated
        aber_ap = create_aberrated_aperture()
        create_multi_aperture([aber_ap])


    def test_on_aperture(self,
        create_multi_aperture: callable,
        create_compound_aperture: callable,
        create_circular_aperture: callable,
        create_aberrated_aperture: callable,
        create_wavefront: callable):
        npixels = 16

        # Test applying
        ap = create_circular_aperture()
        diameter = 3*ap._extent().max()
        aber_ap = create_aberrated_aperture(aperture=ap)
        comp_ap = create_compound_aperture([aber_ap])
        multi_ap = create_multi_aperture([aber_ap, comp_ap])
        wf = multi_ap(create_wavefront(diameter=diameter))

        # Test with normalise
        multi_ap = create_multi_aperture([ap], normalise=True)
        wf = multi_ap(create_wavefront(diameter=diameter))

        # Test methods
        transmission = multi_ap.get_transmission(npixels, diameter)
        coefficients = multi_ap.coefficients
        basis = multi_ap.get_basis(npixels, diameter)
        opd = multi_ap.get_opd(npixels, diameter)

        # Test Basis with no aberrated
        ap = create_circular_aperture()
        multi_ap = create_multi_aperture([ap])
        basis = multi_ap.get_basis(npixels, diameter)
        opd = multi_ap.get_opd(npixels, diameter)


class TestStaticAperture():

    def test_constructor(self, 
        create_static_aperture: callable,
        create_aberrated_aperture: callable,
        create_square_aperture: callable,
        create_static_aberrated_aperture: callable,
        create_compound_aperture: callable) -> None:

        create_static_aperture()

        # Test already static aperture
        with pytest.raises(TypeError):
            static_ap = create_static_aperture()
            create_static_aperture(aperture=static_ap)

        # Test Static Aberrated Aperture
        with pytest.raises(TypeError):
            static_aber_ap = create_static_aberrated_aperture()
            create_static_aperture(aperture=static_aber_ap)
        
        # Test on aberrated compound aperture
        aber_ap = create_aberrated_aperture()
        comp_ap = create_compound_aperture([aber_ap])
        with pytest.raises(TypeError):
            create_static_aperture(aperture=comp_ap)
        
        with pytest.raises(ValueError):
            static_ap = create_static_aperture(coordinates=np.array(1.))

        with pytest.raises(ValueError):
            static_ap = create_static_aperture(npixels=None)

    
    def test_on_aperture(self,
        create_static_aperture: callable,
        ):
        ap = create_static_aperture()
        shape = ap.shape
        transmission = ap.get_transmission()


class TestStaticAberratedAperture():

    def test_constructor(self, 
        create_static_aberrated_aperture: callable,
        create_aberrated_aperture: callable,
        create_square_aperture: callable,
        create_compound_aperture: callable,
        create_circular_aperture) -> None:

        create_static_aberrated_aperture()

        # Test already static aperture
        with pytest.raises(TypeError):
            static_ap = create_static_aberrated_aperture()
            create_static_aberrated_aperture(aperture=static_ap)

        # Test Static Aberrated Aperture
        with pytest.raises(TypeError):
            static_aber_ap = create_static_aberrated_aperture()
            create_static_aberrated_aperture(aperture=static_aber_ap)
        
        # Test on aberrated compound aperture
        ap = create_circular_aperture()
        comp_ap = create_compound_aperture([ap])
        with pytest.raises(TypeError):
            create_static_aberrated_aperture(aperture=comp_ap)
        
        with pytest.raises(ValueError):
            static_ap = create_static_aberrated_aperture(\
                coordinates=np.array(1.))

        with pytest.raises(ValueError):
            static_ap = create_static_aberrated_aperture(npixels=None)

    
    def test_on_aperture(self,
        create_static_aberrated_aperture: callable,
        create_aberrated_aperture: callable,
        create_wavefront: callable,
        create_circular_aperture: callable,
        create_multi_aperture: callable,
        ):
        diameter = 2.

        ap = create_circular_aperture()
        aber_ap = create_aberrated_aperture(aperture=ap)
        stat_ap = create_static_aberrated_aperture(aperture=aber_ap)
        wf = stat_ap(create_wavefront(diameter=diameter))

        ap = create_circular_aperture(normalise=True)
        aber_ap = create_aberrated_aperture(aperture=ap)
        stat_ap = create_static_aberrated_aperture(aperture=aber_ap)
        wf = stat_ap(create_wavefront(diameter=diameter))

        stat_ap.get_basis()
        stat_ap.get_opd()
        stat_ap.opd

        # Multi Aperture with multiple aberrations
        aber_ap = create_aberrated_aperture(aperture=ap)
        multi_ap = create_multi_aperture([aber_ap, aber_ap])
        stat_ap = create_static_aberrated_aperture(multi_ap)
        stat_ap._opd()


class TestApertureFactory():

    def test_constructor(self, create_aperture_factory):
        npixels = 16

        # Valid inputs
        ap = create_aperture_factory()
        create_aperture_factory(nsides=4)
        create_aperture_factory(secondary_ratio=0.1)
        create_aperture_factory(secondary_ratio=0.1, secondary_nsides=4)
        create_aperture_factory(nstruts=1, strut_ratio=0.1)

        # Invalid inputs
        with pytest.raises(ValueError):
            create_aperture_factory(nsides=1)
        
        with pytest.raises(ValueError):
            create_aperture_factory(secondary_nsides=1)
        
        with pytest.raises(ValueError):
            create_aperture_factory(secondary_ratio=-1.)
        
        with pytest.raises(ValueError):
            create_aperture_factory(aperture_ratio=0)
        
        with pytest.raises(ValueError):
            create_aperture_factory(strut_ratio=-1.)
        
        with pytest.raises(ValueError):
            create_aperture_factory(nstruts=1)