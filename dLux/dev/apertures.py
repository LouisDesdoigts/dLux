class PolygonalAperture(RotatableAperture):
    """
    A general representation of a pefect polygonal aperture. 
    Each side of the aperture should be the same length. There
    are some pre-existing implementations for some of the more 
    common cases. This is designed for the exceptions that are 
    less common. 

    Parameters:
    -----------
    nsides: Int
        The number of sides.
    rmax: Float
        The radius of the smallest circle that can fully contain the 
        aperture. 
    x_offset : float, meters
        The centre of the coordinate system along the x-axis.
    y_offset : float, meters
        The centre of the coordinate system along the y-axis. 
    softening: bool 
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool 
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    theta: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    nsides: Int
    rmax: Float


    def __init__(
            self        : Layer,
            x_offset    : Float,
            y_offset    : Float,
            theta       : Float,
            rmax        : Float,
            nsides      : Int,
            occulting   : bool,
            softening   : bool) -> Layer:
        """
        """
        self.rmax = np.asarray(rmax).astype(float)
        self.nsides = int(nsides)
        super().__init__(x_offset, y_offset, theta, occulting, softening)


    def _perp_dist_from_line(
            self    : Layer, 
            point   : Array, 
            grad    : Float, 
            coords  : Array) -> Array:
        """
        """
        x, y = coords[0], coords[1]
        x1, y1 = point[0], point[1]
        return (y - y1 - grad * (x - x1)) / np.sqrt(1 + grad ** 2)


    def _grad_from_two_points(
            self    : Layer,
            point_1 : Array,
            point_2 : Array)-> Array:
        """
        """
        x1, y1 = point_1[0], point_1[1]
        x2, y2 = point_2[0], point_2[1]
        return (y2 - y1) / (x2 - x1)


    def _aperture(self: Layer, coords: Array) -> Array:
        """
        """

#class PolygonalAperture(Aperture):
#    """
#    """
#    x : float
#    y : float
#    alpha : float
#
#
#    def __init__(self : Layer, pixels : int, pixel_scale : float, 
#            vertices : Matrix, theta : float = 0., phi : float = 0., 
#            magnification : float = 1.) -> Layer:
#        """
#        Parameters
#        ----------
#        pixels : int
#            The number of pixels that the entire compound aperture
#            is to be generated over. 
#        pixel_scale : float
#    
#        vertices : float
#    
#        theta : float, radians
#            The angle that the aperture is rotated from the positive 
#            x-axis. By default the horizontal sides are parallel to 
#            the x-axis.
#        phi : float, radians
#            The angle that the y-axis is rotated away from the 
#            vertical. This results in a sheer. 
#        magnification : float
#            A factor by which to enlarge or shrink the aperture. 
#            Should only be very small amounts in typical use cases.
#        """
#        x, y, alpha = self._vertices(vertices)
#        x_offset, y_offset = self._offset(vertices)
#        self.x = np.asarray(x).astype(float)
#        self.y = np.asarray(y).astype(float)
#        self.alpha = np.asarray(alpha).astype(float)
#        super().__init__(pixels, x_offset, y_offset, theta, phi,
#            magnification, pixel_scale)
#
#
#    def _wrap(self : Layer, array : Vector, order : Vector) -> tuple:
#        """
#        Re-order an array and duplicate the first element as an additional
#        final element. Satisfies the postcondition `wrapped.shape[0] ==
#        array.shape[0] + 1`. This is just a helper method to simplify 
#        external object and is not physically important (Only invoke 
#        this method if you know what you are doing)
#
#        Parameters
#        ----------
#        array : Vector
#            The 1-dimensional vector to sort and append to. Must be one 
#            dimesnional else unexpected behaviour can occur.
#        order : Vector
#            The new order for the elements of `array`. Will be accessed 
#            by invoking `array.at[order]` hence `order` must be `int`
#            `dtype`.
#
#        Returns
#        -------
#        wrapped : Vector
#            `array` with `array[0]` appended to the end. The dimensions
#            of `array` are also expanded twofold so that the final
#            shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
#            This is just for the vectorisation demanded later in the 
#            code.
#        """
#        _array = np.zeros((array.shape[0] + 1,))\
#            .at[:-1]\
#            .set(array.at[order].get())\
#            .reshape(-1, 1, 1)
#        return _array.at[-1].set(_array[0])
#        
#
#    def _vertices(self : Layer, vertices : Matrix) -> tuple:
#        """
#        Generates the vertices that are compatible with the rest of 
#        the transformations from the raw data vertices.
#
#        Parameters
#        ----------
#        vertices : Matrix, meters
#            The vertices loaded from the WebbPSF module. 
#
#        Returns
#        -------
#        x, y, angles : tuple 
#            The vertices in normalised positions and wrapped so that 
#            they can be used in the generation of the compound aperture.
#            The `x` is the x coordinates of the vertices, the `y` is the 
#            the y coordinates and `angles` is the angle of the vertex. 
#        """
#        _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
#        _y = (vertices[:, 1] - np.mean(vertices[:, 1]))
#
#        _angles = np.arctan2(_y, _x)
#        _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)
#
#        # By default the `np.arctan2` function returns values within the 
#        # range `(-np.pi, np.pi)` but comparisons are easiest over the 
#        # range `(0, 2 * np.pi)`. This is where the logic implemented 
#        # above comes from. 
#
#        order = np.argsort(_angles)
#
#        x = self._wrap(_x, order)
#        y = self._wrap(_y, order)
#        angles = self._wrap(_angles, order).at[-1].add(2 * np.pi)
#
#        # The final `2 * np.pi` is designed to make sure that the wrap
#        # of the first angle is within the angular coordinate system 
#        # associated with the aperture. By convention this is the
#        # range `angle[0], angle[0] + 2 * np.pi` what this means in 
#        # practice is that the first vertex appearing in the array 
#        # is used to chose the coordinate system in angular units. 
#
#        return x, y, angles
#
#
#    def _offset(self : Layer, vertices : Matrix) -> tuple:
#        """
#        Get the offsets of the coordinate system.
#
#        Parameters
#        ----------
#        vertices : Matrix 
#            The unprocessed vertices loaded from the JWST data file.
#            The correct shape for this array is `vertices.shape == 
#            (2, number_of_vertices)`. 
#        pixel_scale : float, meters
#            The physical size of each pixel along one of its edges.
#
#        Returns 
#        -------
#        x_offset, y_offset : float, meters
#            The x and y offsets in physical units. 
#        """
#        x_offset = np.mean(vertices[:, 0])
#        y_offset = np.mean(vertices[:, 1])
#        return x_offset, y_offset
#
#
#    # TODO: number_of_pixels can be moved out as a parameter
#    def _rad_coordinates(self : Layer) -> tuple:
#        """
#        Generates the vectorised coordinate system associated with the 
#        aperture.
#
#        Parameters
#        ----------
#        phi_naught : float 
#            The angle substending the first vertex. 
#
#        Returns 
#        -------
#        rho, theta : tuple[Tensor]
#            The stacked coordinate systems that are typically passed to 
#            `_segments` to generate the segments.
#        """
#        cartesian = self._coordinates()
#        positions = cartesian_to_polar(cartesian)
#
#        # rho = positions[0] * self.get_pixel_scale()
#        rho = positions[0]        
#
#        theta = positions[1] 
#        theta += 2 * np.pi * (positions[1] < 0.)
#        theta += 2 * np.pi * (theta < self.alpha[0])
#
#        rho = np.tile(rho, (6, 1, 1))
#        theta = np.tile(theta, (6, 1, 1))
#        return rho, theta
#
#
#    def _edges(self : Layer, rho : Tensor, theta : Tensor) -> Tensor:
#        """
#        Generate lines connecting adjacent vertices.
#
#        Parameters
#        ----------
#        rho : Tensor, meters
#            Represents the radial distance of every point from the 
#            centre of __this__ aperture. 
#        theta : Tensor, Radians
#            The angle associated with every point in the final bitmap.
#
#        Returns
#        -------
#        edges : Tensor
#            The edges represented as a Bitmap with the points inside the 
#            edge marked as 1. and outside 0. The leading axis contains 
#            each unique edge and the corresponding matrix is the bitmap.
#        """
#        # This is derived from the two point form of the equation for 
#        # a straight line (eq. 1)
#        # 
#        #           y_2 - y_1
#        # y - y_1 = ---------(x - x_1)
#        #           x_2 - x_1
#        # 
#        # This is rearranged to the form, ay - bx = c, where:
#        # - a = (x_2 - x_1)
#        # - b = (y_2 - y_1)
#        # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
#        # we can then drive the transformation to polar coordinates in 
#        # the usual way through the substitutions; y = r sin(theta), and 
#        # x = r cos(theta). The equation is now in the form 
#        #
#        #                  c
#        # r = ---------------------------
#        #     a sin(theta) - b cos(theta) 
#        #
#        a = (self.x[1:] - self.x[:-1])
#        b = (self.y[1:] - self.y[:-1])
#        c = (a * self.y[:-1] - b * self.x[:-1])
#
#        linear = c / (a * np.sin(theta) - b * np.cos(theta))
#        #return rho < (linear * self.get_pixel_scale())
#        return rho < linear
#
#
#    def _wedges(self : Layer, theta : Tensor) -> Tensor:
#        """
#        The angular bounds of each segment of an individual hexagon.
#
#        Parameters
#        ----------
#        theta : Tensor, Radians
#            The angle away from the positive x-axis of the coordinate
#            system associated with this aperture. Please note that `theta`
#            May not start at zero. 
#
#        Returns 
#        -------
#        wedges : Tensor 
#            The angular bounds associated with each pair of vertices in 
#            order. The leading axis of the Tensor steps through the 
#            wedges in order arround the circle. 
#        """
#        return (self.alpha[:-1] < theta) & (theta < self.alpha[1:])
#
#
#    def _segments(self : Layer, theta : Tensor, rho : Tensor) -> Tensor:
#        """
#        Generate the segments as a stacked tensor. 
#
#        Parameters
#        ----------
#        theta : Tensor
#            The angle of every pixel associated with the coordinate system 
#            of this aperture. 
#        rho : Tensor
#            The radial positions associated with the coordinate system 
#            of this aperture. 
#
#        Returns 
#        -------
#        segments : Tensor 
#            The bitmaps corresponding to each vertex pair in the vertices.
#            The leading dimension contains the unique segments. 
#        """
#        edges = self._edges(rho, theta)
#        wedges = self._wedges(theta)
#        return (edges & wedges).astype(float)
#        
#
#    def _aperture(self : Layer) -> Matrix:
#        """
#        Generate the BitMap representing the aperture described by the 
#        vertices. 
#
#        Returns
#        -------
#        aperture : Matrix 
#            The Bit-Map that represents the aperture. 
#        """
#        rho, theta = self._rad_coordinates()
#        segments = self._segments(theta, rho)
#        return segments.sum(axis=0)
#
