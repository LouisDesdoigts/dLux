import dLux 
import jax.numpy as np
import typing
# from jax import config
# config.update("jax_debug_nans", True)

Aperture = typing.TypeVar("Aperture")
Array = typing.TypeVar("Array")
Spider = typing.TypeVar("Spider")


class TestAperturesCommonInterfaces():
    """
    For each type of aperture that has common properties, test it
    """
    def _assert_valid_hard_aperture(aperture, msg=''):
        assert ((aperture == 1.) | (aperture == 0.)).all(), msg
        
    def _assert_valid_soft_aperture(aperture, msg=''):
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all(), msg
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all(), msg
        # there should also exist something in bounds (assuming edges are within coordinates)
        assert np.logical_and(aperture > 0., aperture < 1.).any(), msg
        
    # input is all the fixtures that need testing
    def test_all_apertures(self, create_square_aperture : callable,
                           create_rectangular_aperture : callable,
                           create_circular_aperture : callable,
                           create_hexagonal_aperture : callable,
                           create_annular_aperture : callable):
        
        constructors = [create_square_aperture,
                        create_rectangular_aperture,
                        create_circular_aperture,
                        create_hexagonal_aperture,
                        create_annular_aperture]
        
        # TODO might need to add error message for when this fails but it checks things very easily
        for ctor in constructors:
            self._test_single_aperture_class(ctor)
    
    def _test_single_aperture_class(self, aperture_fixture):
        
        coordinates = dLux.utils.get_pixel_coordinates(512, 2./512)
        
        x_offset = 1.
        y_offset = 1.
        centres = [np.array([0., 0.]),
                   np.array([x_offset, 0.]),
                   np.array([0., y_offset]),
                   np.array([x_offset, y_offset])]
        
        rotations = [0, np.pi/3., -np.pi/3.5, np.pi/2.]
        not_rotatable_apertures = (dLux.apertures.CircularAperture,
                                   dLux.apertures.AnnularAperture)
        
        base_kwargs = {"centre" : None,
                       "softening" : None,
                       "occulting" : None
                       }

        for centre in centres:
            for rotation in rotations:
                for softening in [True, False]:
                    for occulting in [True, False]:
                        actual_kwargs = base_kwargs
                        actual_kwargs["centre"] = centre
                        actual_kwargs["softening"] = softening
                        actual_kwargs["occulting"] = occulting
                        
                        if not isinstance(aperture_fixture(), not_rotatable_apertures):
                            actual_kwargs["rotation"] = rotation
                            
                        
                        aperture = aperture_fixture(**actual_kwargs)._aperture(coordinates)
                        
                        msg = f'{actual_kwargs}, on ctor {aperture_fixture}'
                        
                        if softening:
                            TestAperturesCommonInterfaces._assert_valid_soft_aperture(aperture, msg)
                        else:
                            TestAperturesCommonInterfaces._assert_valid_hard_aperture(aperture, msg)


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
        npix = 128
        width = 2.
        coordinates = dLux.utils.get_pixel_coordinates(npix, width / npix)

        # Case Translated 
        spider = create_uniform_spider(centre=[1., 1.], softening=False)
        aperture = spider._aperture(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Rotated 
        spider = create_uniform_spider(rotation=np.pi/4., softening=False)
        aperture = spider._aperture(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Strained 
        spider = create_uniform_spider(strain=[.05, .05], softening=False)
        aperture = spider._aperture(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Compression
        spider = create_uniform_spider(compression=[1.05, .95], softening=False)
        aperture = spider._aperture(coordinates)
        assert ((aperture == 1.) | (aperture == 0.)).all()


    def test_range_soft(self, create_uniform_spider: callable) -> None:
        """
        Checks that the aperture falls into the correct range.
        """
        npix = 128
        width = 2.
        coordinates = dLux.utils.get_pixel_coordinates(npix, width / npix)

        # Case Translated 
        spider = create_uniform_spider(centre=[1., 1.], softening=False)
        aperture = spider._aperture(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Rotated 
        spider = create_uniform_spider(rotation=np.pi/4., softening=False)
        aperture = spider._aperture(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Strained 
        spider = create_uniform_spider(strain=[.05, .05], softening=False)
        aperture = spider._aperture(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()

        # Case Compression
        spider = create_uniform_spider(compression=[1.05, .95], softening=False)
        aperture = spider._aperture(coordinates)
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all()
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all()


class TestAberratedAperture(object):
    """
    Checks that the aberrated aperture is functional. It does not
    test whether the aberrated aperture is correct.
    """


    def test_constructor(self, create_aberrated_aperture: callable) -> None:
        """
        Tests that it is possible to instantiate an AberratedAperture.
        Does not test if the AberratedAperture is correct.
        """
        # TODO: Make sure that the class asserts that the coeffs and 
        # the noll indexes have the same length.
        create_aberrated_aperture()


    def test_on_aperture(self: object, 
            create_aberrated_aperture: callable,
            create_circular_aperture: callable) -> None:
        """
        Tests that the basis functions are evaluated atop the aperture.
        Applies mutliple different permutations.
        """
        width = 2.
        npix = 128
        coordinates = dLux.utils.get_pixel_coordinates(npix, width / npix)

        ap = create_circular_aperture()

        aber_ap = create_aberrated_aperture(aperture=ap)._basis(coordinates)
        ap = ap._aperture(coordinates)

        abers = np.where(ap == 0., aber_ap, 0.)
        assert (abers == 0.).all()

