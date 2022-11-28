import jax.numpy as np
import typing
from apertures import ApertureUtility 
from utilities import UtilityUser

class UniformSpiderUtility(ApertureUtility):
    """
    Contains the default values for the constructor of the 
    `UniformSpider` class.

    Parameters:
    -----------
    number_of_struts: int 
        The number of supporting struts in the spider. 
    width_of_struts: float, meters
        The thickness of each strut. 
    theta: float, radians
        The global rotation of the struts. 
    """
    number_of_struts: int = 4
    width_of_struts: float = .05
    theta: float = 0.


    def construct(self,
            width_of_struts: float = None,
            number_of_struts: int = None,
            softening: bool = None,
            x_offset: float = None,
            y_offset: float = None, 
            theta: float = None) -> Spider:
        """
        Return a ready-to-test `UniformSpider` instance. 

        Parameters:
        -----------
        width_of_struts: float, meters
            The width of each strut measured in meters.
        number_of_struts: int
            The number of struts that the spider has. 
        softening: bool
            True if the aperture is soft edged and false otherwise.
        x_offset: float, meters 
            The distance along the x-axis that the spider is to be moved. 
        y_offset: float, meters 
            The distance along the y-axis that the spider is to be moved. 
        theta: float, radians 
            The global rotation of the spider. 
        """
        return dl.UniformSpider(
            width_of_struts = self.width_of_struts if not width_of_struts else width_of_struts,
            number_of_struts = self.number_of_struts if not number_of_struts else number_of_struts,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            theta = self.theta if not theta else theta)


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
        assert spider.theta == self.utility.theta

        # Case: Extra Strut
        number_of_struts = 5
        spider = self.utility.construct(number_of_struts = number_of_struts)

        assert spider.number_of_struts == number_of_struts

        # Case: Fatter Struts
        width_of_struts = .1
        spider = self.utility.construct(width_of_struts = width_of_struts)

        assert spider.width_of_struts == width_of_struts

        # Case: Rotated
        theta = np.pi / 4.
        spider = self.utility.construct(theta = theta)

        assert spider.theta == theta

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
        theta = np.pi / 2.
        aperture = self\
            .utility\
            .construct(theta = theta)\
            ._aperture(coords)

        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Occulting
        occulting = True
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
            ._aperture(coords)

        assert ((aperture == 1.) | (aperture == 0.)).all()

        # Case Not Occulting
        occulting = False
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
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
        theta = np.pi / 2.
        aperture = self\
            .utility\
            .construct(theta = theta)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()

        # Case Occulting
        occulting = True
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()

        # Case Not Occulting
        occulting = False
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
            ._aperture(coords)

        assert (aperture <= 1.).all()
        assert (aperture >= 0.).all()
