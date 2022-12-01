class HexagonalAperture(RotableAperture):
    """
    Generate a hexagonal aperture, parametrised by rmax. 
    
    Attributes
    ----------
    rmax : float, meters
        The infimum of the radii of the set of circles that fully 
        enclose the hexagonal aperture. In other words the distance 
        from the centre to one of the vertices. 
    """
    rmax : float


    def __init__(self   : Layer, 
            x_offset    : float, 
            y_offset    : float, 
            theta       : float, 
            rmax        : float,
            softening   : bool,
            occulting   : bool) -> Layer:
        """
        Parameters
        ----------
        x_offset : float, meters
            The centre of the coordinate system along the x-axis.
        y_offset : float, meters
            The centre of the coordinate system along the y-axis. 
        theta : float, radians
            The rotation of the coordinate system of the aperture 
            away from the positive x-axis. Due to the symmetry of 
            ring shaped apertures this will not change the final 
            shape and it is recomended that it is just set to zero.
        rmax : float, meters
            The distance from the center of the hexagon to one of
            the vertices. . 
        softening: bool
            True if the aperture is soft edged else False.
        occulting: bool
            True is the aperture is occulting else False. An occulting 
            Aperture is zero inside and one outside. 
        """
        super().__init__(x_offset, y_offset, theta, rmax, softening, occulting)
        self.rmax = np.asarray(rmax).astype(float)


    def get_rmax(self : Layer) -> float:
        """
        Returns
        -------
        max_radius : float, meters
            The distance from the centre of the hexagon to one of 
            the vertices.
        """
        return self.rmax


    def _aperture(self : Layer, coords: Array) -> Array:
        """
        Generates an array representing the hard edged hexagonal 
        aperture. 

        Parameters:
        -----------
        coords: Array, meters
            The coordinates over which to generate the aperture. 

        Returns
        -------
        aperture : Array
            The aperture represented as a binary float array of 0. and
            1. representing no transmission and transmission 
            respectively.
        """
        x_centre, y_centre = self.get_centre()
        maximum_radius = self.get_rmax()

        x *= 2 / number_of_pixels
        y *= 2 / number_of_pixels

        rectangle = (np.abs(x) <= maximum_radius / 2.) \
            & (np.abs(y) <= (maximum_radius * np.sqrt(3) / 2.))

        left_triangle = (x <= - maximum_radius / 2.) \
            & (x >= - maximum_radius) \
            & (np.abs(y) <= (x + maximum_radius) * np.sqrt(3))

        right_triangle = (x >= maximum_radius / 2.) \
            & (x <= maximum_radius) \
            & (np.abs(y) <= (maximum_radius - x) * np.sqrt(3))

        hexagon = rectangle | left_triangle | right_triangle
        return np.asarray(hexagon).astype(float)


