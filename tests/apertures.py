class ApertureUtility(Utility):
    """
    Contains the default constructor values for all of
    the parameters common to every aperture.

    Parameters:
    -----------
    occulting: bool = False
        The default occulting setting is false.
    softening: bool = False
        The default softening setting is False.
    x_offset: float = 0.
        The default aperture is centred. 
    y_offset: float = 0. 
        The default aperture is centred. 
    """
    occulting: bool = False
    softening: bool = False
    x_offset: float = 0.
    y_offset: float = 0.


class SquareApertureUtility(Utility):
    """
    Contains useful default values for the constructor
    of a `SquareAperture`.

    parameters:
    -----------
    width: float = 0. (meters)
        The default aperture is one meter wide. 
    """
    width: float = 1.


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            width: float = None) -> Aperture:
        """
        Construct's an instance of \`SquareAperture\` making 
        it easy to toggle a single parameter. 

        Parameters:
        -----------
        occulting: bool = None
            True if light passes through the aperture. 
            False if light is blocked by the aperture. 
        softening: bool = None
            True is the aperture has soft pixels on the 
            edges else false. 
        x_offset: float = None, meters
            The distance along the x axis that the aperture 
            is moved from the centre of the coordinate system. 
        y_offset: float = None, meters
            The distance along the y axis that the aperture 
            is moved from the centre of the coordinate system.
        width: float = None, meters
            The width of the aperture.
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            width = self.width if not width else width)


class RectangularApertureUtility(Utility):
    """
    Contains useful default values for the constructor
    of a `RectangularAperture`.

    Parameters:
    -----------
    width: float = 0. (meters)
        The default aperture is one meter wide. 
    length: float = .5, meters 
        The defualt aperture is half a meter long. 
    """
    occulting: bool = False
    softening: bool = False
    x_offset: float = 0.
    y_offset: float = 0.
    width: float = 1.
    length: float = .5


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            width: float = None,
            length: float = None) -> Aperture:
        """
        Construct's an instance of \`SquareAperture\` making 
        it easy to toggle a single parameter. 

        Parameters:
        -----------
        occulting: bool = None
            True if light passes through the aperture. 
            False if light is blocked by the aperture. 
        softening: bool = None
            True is the aperture has soft pixels on the 
            edges else false. 
        x_offset: float = None, meters
            The distance along the x axis that the aperture 
            is moved from the centre of the coordinate system. 
        y_offset: float = None, meters
            The distance along the y axis that the aperture 
            is moved from the centre of the coordinate system.
        width: float = None, meters
            The width of the aperture.
        length: float = None, meters 
            The length of the aperture.
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            self.x_offset if not x_offset else x_offset,
            self.y_offset if not y_offset else y_offset,
            self.width if not width else width)
class CircularApertureUtility(Utility):
class AnnularApertureUtility(Utility):

class TestSquareAperture(UtilityUser):
    def test_constructor():
        # Case default

        # Case Translated X

        # Case Translated Y

        # Case Rotated Clockwise

        # Case Rotated Anticlockwise

        # Case Soft

        # Case Hard 

        # Case Occulting

        # Case Non-occulting


    def test_range_hard():
        # Case Translated X

        # Case Translated Y

        # Case Rotated 

        # Case Occulting

        # Case Not Occulting


    def test_range_soft():
        # Case Translated X

        # Case Translated Y

        # Case Rotated 

        # Case Occulting

        # Case Not Occulting


class TestRectangularAperture(UtilityUser):
    def test_constructor():
        # Case default

        # Case Translated X

        # Case Translated Y

        # Case Rotated 

        # Case Soft

        # Case Hard 

        # Case Occulting

        # Case Non-occulting


    def test_range_hard():
        # Case Translated X

        # Case Translated Y

        # Case Rotated 

        # Case Occulting

        # Case Not Occulting


    def test_range_soft():
        # Case Translated X

        # Case Translated Y

        # Case Rotated 

        # Case Occulting

        # Case Not-occulting


class TestCircularAperture(UtilityUser):
    def test_constructor():
        # Case default

        # Case Translated X

        # Case Translated Y

        # Case Soft

        # Case Hard

        # Case Occulting

        # Case Non-occulting


    def test_range_hard():
        # Case Translated X

        # Case Translated Y

        # Case Occulting

        # Case Non-occulting


    def test_range_soft():
        # Case Translated X

        # Case Translated Y

        # Case Occulting

        # Case Non-occulting


class TestAnnularAperture(UtilityUser):
    def test_constructor():
        # Case default

        # Case Translated X

        # Case Translated Y

        # Case Soft

        # Case Hard

        # Case Occulting

        # Case Non-occulting


    def test_range_hard():
        # Case Translated X

        # Case Translated Y

        # Case Occulting

        # Case Non-occulting


    def test_range_soft():
        # Case Translated X

        # Case Translated Y

        # Case Occulting

        # Case Non-occulting

