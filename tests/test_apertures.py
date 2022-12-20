import dLux 
import jax.numpy as np
import typing

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
        # there should also exist something in bounds (assuming edges are within coords)
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
        
        coords = dLux.utils.get_pixel_coordinates(512, 2./512)
        
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
                            
                        
                        aperture = aperture_fixture(**actual_kwargs)._aperture(coords)
                        
                        msg = f'{actual_kwargs}, on ctor {aperture_fixture}'
                        
                        if softening:
                            TestAperturesCommonInterfaces._assert_valid_soft_aperture(aperture, msg)
                        else:
                            TestAperturesCommonInterfaces._assert_valid_hard_aperture(aperture, msg)




class TestUniformSpiderUtility(UtilityUser):
    """
    Contains the unit tests for the `UniformSpider` class.

    Parameters:
    -----------
    utility: Utility
        Helper functions and default values for easier 
        comparisons. 
    """
    utility: Utility = UniformSpiderUtility()


    def test_constructor(self) -> None:
        """
        Tests that the state is correctly initialised. 
        """
        spider = self.utility.construct()

        assert spider.number_of_struts == self.utility.number_of_struts
        assert spider.width_of_struts == self.utility.width_of_struts
        assert spider.softening == np.inf
        assert spider.x_offset == self.utility.x_offset
        assert spider.y_offset == self.utility.y_offset
        assert spider.rotation == self.utility.rotation

        # Case: Extra Strut
        number_of_struts = 5
        spider = self.utility.construct(number_of_struts = number_of_struts)

        assert spider.number_of_struts == number_of_struts

        # Case: Fatter Struts
        width_of_struts = .1
        spider = self.utility.construct(width_of_struts = width_of_struts)

        assert spider.width_of_struts == width_of_struts

        # Case: Rotated
        rotation = np.pi / 4.
        spider = self.utility.construct(rotation = rotation)

        assert spider.rotation == rotation

        # Case: Translated x
        x_offset = 1.
        spider = self.utility.construct(x_offset = x_offset)

        assert spider.x_offset == x_offset

        # Case: Translated y
        y_offset = 1.
        spider = self.utility.construct(y_offset = y_offset)

        assert spider.y_offset == y_offset

    
    def test_range_hard(self) -> None:
        """
        Checks that the apertures fall into the correct range.
        """
        coords = self.utility.get_coordinates()

        # Case Translated X
        x_offset = 1.
        aperture = self\
            .utility\
            .construct(x_offset = x_offset)\
            ._aperture(coords)

        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Translated Y
        y_offset = 1.
        aperture = self\
            .utility\
            .construct(y_offset = y_offset)\
            ._aperture(coords)

        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Rotated 
        rotation = np.pi / 2.
        aperture = self\
            .utility\
            .construct(rotation = rotation)\
            ._aperture(coords)

        assert ((aperture == 1.) | (aperture == 0.)).all()


    def test_range_soft(self) -> None:
        """
        Checks that the aperture falls into the correct range.
        """
        coords = self.utility.get_coordinates()

        # Case Translated X
        x_offset = 1.
        aperture = self\
            .utility\
            .construct(x_offset = x_offset)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()

        # Case Translated Y
        y_offset = 1.
        aperture = self\
            .utility\
            .construct(y_offset = y_offset)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()

        # Case Rotated 
        rotation = np.pi / 2.
        aperture = self\
            .utility\
            .construct(rotation = rotation)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()

