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


    def get_coordinates(self) -> Array:
        """
        Returns:
        --------
        coords: Array, meters
            A default coordinate array that the aperture can be generated
            over. 
        """
        return dl.utils.get_coordinates(128, 4. / 128)


class SquareApertureUtility(ApertureUtility):
    """
    Contains useful default values for the constructor
    of a `SquareAperture`.

    parameters:
    -----------
    width: float = 0. meters
        The default aperture is one meter wide.
    theta: float = 0., radians
        The default aperture is not rotated.
    """
    width: float = 1.
    theta: float = 0.


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            width: float = None,
            theta: float = None) -> Aperture:
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
        theta: float = None, radians
            The rotationof the apertue with respect to the 
            coordinate system. 
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            width = self.width if not width else width,
            theta = self.theta if not theta else theta)


class RectangularApertureUtility(ApertureUtility):
    """
    Contains useful default values for the constructor
    of a `RectangularAperture`.

    Parameters:
    -----------
    width: float = 0. (meters)
        The default aperture is one meter wide. 
    length: float = .5, meters 
        The defualt aperture is half a meter long. 
    theta: float = 0., radians
        The default aperture is not rotated. 
    """
    width: float = 1.
    length: float = .5
    theta: float = 0.


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            width: float = None,
            length: float = None,
            theta: float = None) -> Aperture:
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
        theta: float = None, radians
            The rotation of the aperture.
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            width = self.width if not width else width,
            length = self.length if not length else length, 
            theta = self.theta if not theta else theta)


class CircularApertureUtility(ApertureUtility):
    """
    Contains the default parameters for the `CircularAperture`. 

    Parameters:
    -----------
    radius: float = 1., meters
        The default aperture is one meter wide.
    """
    radius: float = 1.


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            radius: float = None) -> Aperture:
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
        radius: float = None, meters
            The radius of the aperture.
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            radius = self.radius if not radius else radius)
    

class AnnularApertureUtility(ApertureUtility):
    """
    Contains the parameters of the `AnnularAperture` that 
    are used as default. 

    Parameters:
    -----------
    rmin: float = .5, meters
        The default inner radius is half a meter. 
    rmax: float = 1., meters
        The default outer radius is a full meter.
    """
    rmin: float = .5 
    rmax: float = 1.


    def construct(self, 
            occulting: bool = None, 
            softening: bool = None,
            x_offset: float = None, 
            y_offset: float = None, 
            rmin: float = None,
            rmax: float = None) -> Aperture:
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
        rmin: float = None, meters
            The inner radius of the aperture.
        rmax: float = None, meters
            The outer radius of the aperture.
        """
        return dl.SquareAperture(
            occulting = self.occulting if not occulting else occulting,
            softening = self.softening if not softening else softening,
            x_offset = self.x_offset if not x_offset else x_offset,
            y_offset = self.y_offset if not y_offset else y_offset,
            rmin = self.rmin if not rmin else rmin,
            rmax = self.rmax if not rmax else rmax)


class TestSquareAperture(UtilityUser):
    """
    Provides unit tests for the \`Square Aperture\` class. 

    Parameters:
    -----------
    utility: SquareApertureUtility
        Provides default parameter values and coordinate systems. 
    """
    utility: SquareApertureUtility = SquareApertureUtility()


    def test_constructor(self) -> None:
        """
        Checks that all of the fields are correctly set. 
        """
        # Case default
        sq_ap = self.utility.construct()

        assert sq_ap.occulting == self.utility.occulting
        assert sq_ap.softening == self.utility.softening
        assert sq_ap.x_offset == self.utility.x_offset
        assert sq_ap.y_offset == self.utility.y_offset
        assert sq_ap.width == self.utility.width
        assert sq_ap.theta 

        # Case Translated X
        x_offset = 1.
        sq_ap = self.utility.construct(x_offset = x_offset)

        assert sq_ap.x_offset == x_offset

        # Case Translated Y
        y_offset = 1.
        sq_ap = self.utility.construct(y_offset = y_offset)

        assert sq_ap.y_offset == y_offset

        # Case Rotated Clockwise
        theta = np.pi / 2.
        sq_ap = self.utility.construct(theta = theta)

        assert sq_ap.theta == theta


    def test_range_hard(self) -> None:
        """
        Checks that the apertures fall into the correct range.
        """
        coords = self.get_coordinates()

        # Case Translated X
        x_offset = 1.
        aperture = self\
            .utility\
            .construct(x_offset = x_offset)\
            ._aperture(coords)

        assert (aperture == 1 || aperture == 0).all()

        # Case Translated Y
        y_offset = 1.
        aperture = self\
            .utility\
            .construct(y_offset = y_offset)\
            ._aperture(coords)

        assert (aperture == 1 || aperture == 0).all()

        # Case Rotated 
        theta = np.pi / 2.
        aperture = self\
            .utility\
            .construct(theta = theta)\
            ._aperture(coords)

        assert (aperture == 1 || aperture == 0).all()

        # Case Occulting
        occulting = True
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
            ._aperture(coords)

        assert (aperture == 1 || aperture == 0).all()

        # Case Not Occulting
        occulting = False
        aperture = self\
            .utility\
            .construct(occulting = occulting)\
            ._aperture(coords)

        assert (aperture == 1 || aperture == 0).all()


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

