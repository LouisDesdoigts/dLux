from dLux import Aperture, CompoundAperture
from matplotlib import pyplot
from dLux.utils import (get_positions_vector, get_pixel_positions)
from constants import *

config.update("jax_enable_x64", True)

Array = TypeVar("Array")
Layer = TypeVar("Layer")
Tensor = TypeVar("Tensor")
Matrix = TypeVar("Matrix")
Vector = TypeVar("Vector")

MAX_DIFF = 4


def factorial(n : int) -> int:
    """
    Calculate n! in a jax friendly way. Note that n == 0 is not a 
    safe case.  

    Parameters
    ----------
    n : int
        The integer to calculate the factorial of.

    Returns
    n! : int
        The factorial of the integer
    """
    return jax.lax.exp(jax.lax.lgamma(n + 1.))


class JWSTPrimaryApertureSegment(Aperture):
    """
    A dLux implementation of the JWST primary aperture segment.
    The segments are sketched and drawn below:

                            +---+
                           *     *
                      +---+  B1   +---+
                     *     *     *     *
                +---+  C6   +---+  C1   +---+
               *     *     *     *     *     *
              +  B6   +---+  A1   +---+  B2   +
               *     *     *     *     *     *
                +---+  A6   +---+  A2   +---+
               *     *     *     *     *     *
              +  C5   +---+       +---+  C2   +
               *     *     *     *     *     * 
                +---+  A5   +---+  A3   +---+
               *     *     *     *     *     *   
              +  B5   +---+  A4   +---+  B3   +
               *     *     *     *     *     *
                +---+  C4   +---+  C3   +---+
                     *     *     *     *
                      +---+  B4   +---+
                           *     *         
                            +---+    

    The data for the vertices is retrieved from WebbPSF and the 
    syntax for using the class is as follows:

    >>> npix = 1008 # Total number of pixels for the entire primary
    >>> appix = 200 # Pixels for this specific aperture. 
    >>> C1 = JWSTPrimaryApertureSegment("C1-1", npix, appix)
    >>> aperture = C1._aperture()

    If you want to only model one mirror then appix and npix can be 
    set to the same. The assumption is that the entire aperture is 
    going to be modelled. 

    To use the aperture to generate an orthonormal basis on the not 
    quite a hexagon we use the following code. 

    >>> basis = Basis(C1(), nterms)._basis()

    To learn the rotation, shear and other parameters of the mirror 
    we can provide this functionality to the constructor of the 
    aperture. For example:
    
    >>> C1 = JWSTPrimaryApertureSegment(
    ...     segement : str = "C1-1",
    ...     number_of_pixels : int = npix,
    ...     aperture_pixels : int = appix,
    ...     rotation : float = 0.1,
    ...     shear : float = 0.1,
    ...     magnification = 1.001)
    >>> basis = Basis(npix, nterms, C1)._basis()   

    The generation of zernike polynomials and there orthonormalisation
    is an archilies heal of the project, currently runnig much slower 
    than equivalent implementations and there is ongoing work into 
    optimising this but for now you are unfortunate.  

    Attributes
    ----------
    segement : str
        The string code for the segment that is getting modelled. 
        The format for the string is the "Ln" where "L" is the 
        letter code and "n" is the number. The format "Ln-m" will 
        also work where m is an integer mapping of "Ln" to the 
        integers. The way it works is A1, A2, A3, A4, A5 and A6 map 
        to 1, 2, 3, 4, 5 and 6. B1, B2, B3, B4, B5 and B6 map to 
        7, 9, 11, 13, 15, 17 and Cn map to the remaining numbers.
    x_offset : float, meters
        The x offset of the centre. This is automatically calculated
        in the consturctor but can be changed and optimised.
    y_offset : float, meters
        The y offset of the centre. This is automatically calculated 
        in the constructor but can be changed and optimised. 
    theta : float, Radians
        The angle of rotation from the positive x axis.
    phi : float
        The angle of shear. If the initial shape of the aperture is
        as shown in fig. 1, then the sheered aperture is as shown 
        in fig. 2.
                                  |
                                +---+
                               *  |  *
                          <---+---+---+--->
                               *  |  *
                                +---+                       
                                  |
                    fig 1. The unsheered aperture.

                                  |  / 
                                  +---+
                                * |/  *
                          <---+---+---+--->
                              *  /| *
                              +---+                       
                               /  |
                    fig 2. The sheered aperture. 

    magnification : float
        The multiplicative factor indicating the size of the aperture
        from the initial.
    pixel_scale : float, meters per pixel
        The is automatically calculated. DO NOT MODIFY THIS VALUE.  
    x : Vector[float], meters
        The x coordinates of the vertices. It is not recomended that 
        gradients be taken with respect to these. Also note that these
        are measured with respect to the centre of the coordinate 
        system with the origin at `x_offset`, `y_offset`.
    y : Vector[float]
        The y coordinates of the vertices. It is not recommended that
        gradients be take with respect to these. As with `x` these 
        are centred on the coordinate system with the origin at 
        `x_offset`, `y_offset`.
    alpha : Vector[float], radians.   
        The angular position of each vertex counterclockwise around
        the shape.            
    """
    segment : str 
    x : float
    y : float
    alpha : float


    def __init__(self : Layer, segment : str, pixels : int,
            pixel_scale : float, theta : float = 0., phi : float = 0., 
            magnification : float = 1.) -> Layer:
        """
        Parameters
        ----------
        segement : str
            The James Webb primary mirror section to modify. The 
            format of this string is "Ln", where "L" is a letter 
            "A", "B" or "C" and "n" is a number. See the class
            documentation for more detail.
        pixels : int
            The number of pixels that the entire compound aperture
            is to be generated over. 
        theta : float, radians
            The angle that the aperture is rotated from the positive 
            x-axis. By default the horizontal sides are parallel to 
            the x-axis.
        phi : float, radians
            The angle that the y-axis is rotated away from the 
            vertical. This results in a sheer. 
        magnification : float
            A factor by which to enlarge or shrink the aperture. 
            Should only be very small amounts in typical use cases.
        """
        vertices = self._load(segment)
        x, y, alpha = self._vertices(vertices)
        x_offset, y_offset = self._offset(vertices)
        self.segment = str(segment)
        self.x = np.asarray(x).astype(float)
        self.y = np.asarray(y).astype(float)
        self.alpha = np.asarray(alpha).astype(float)
        super().__init__(pixels, x_offset, y_offset, theta, phi,
            magnification, pixel_scale)


    # TODO: Does this functionality really need to be separate. 
    # consider moving into the function below.              
    def _wrap(self : Layer, array : Vector, order : Vector) -> tuple:
        """
        Re-order an array and duplicate the first element as an additional
        final element. Satisfies the postcondition `wrapped.shape[0] ==
        array.shape[0] + 1`. This is just a helper method to simplify 
        external object and is not physically important (Only invoke 
        this method if you know what you are doing)

        Parameters
        ----------
        array : Vector
            The 1-dimensional vector to sort and append to. Must be one 
            dimesnional else unexpected behaviour can occur.
        order : Vector
            The new order for the elements of `array`. Will be accessed 
            by invoking `array.at[order]` hence `order` must be `int`
            `dtype`.

        Returns
        -------
        wrapped : Vector
            `array` with `array[0]` appended to the end. The dimensions
            of `array` are also expanded twofold so that the final
            shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
            This is just for the vectorisation demanded later in the 
            code.
        """
        _array = np.zeros((array.shape[0] + 1,))\
            .at[:-1]\
            .set(array.at[order].get())\
            .reshape(-1, 1, 1)
        return _array.at[-1].set(_array[0])
        

    def _vertices(self : Layer, vertices : Matrix) -> tuple:
        """
        Generates the vertices that are compatible with the rest of 
        the transformations from the raw data vertices.

        Parameters
        ----------
        vertices : Matrix, meters
            The vertices loaded from the WebbPSF module. 

        Returns
        -------
        x, y, angles : tuple 
            The vertices in normalised positions and wrapped so that 
            they can be used in the generation of the compound aperture.
            The `x` is the x coordinates of the vertices, the `y` is the 
            the y coordinates and `angles` is the angle of the vertex. 
        """
        _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
        _y = (vertices[:, 1] - np.mean(vertices[:, 1]))

        _angles = np.arctan2(_y, _x)
        _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)

        # By default the `np.arctan2` function returns values within the 
        # range `(-np.pi, np.pi)` but comparisons are easiest over the 
        # range `(0, 2 * np.pi)`. This is where the logic implemented 
        # above comes from. 

        order = np.argsort(_angles)

        x = self._wrap(_x, order)
        y = self._wrap(_y, order)
        angles = self._wrap(_angles, order).at[-1].add(2 * np.pi)

        # The final `2 * np.pi` is designed to make sure that the wrap
        # of the first angle is within the angular coordinate system 
        # associated with the aperture. By convention this is the
        # range `angle[0], angle[0] + 2 * np.pi` what this means in 
        # practice is that the first vertex appearing in the array 
        # is used to chose the coordinate system in angular units. 

        return x, y, angles


    def _offset(self : Layer, vertices : Matrix) -> tuple:
        """
        Get the offsets of the coordinate system.

        Parameters
        ----------
        vertices : Matrix 
            The unprocessed vertices loaded from the JWST data file.
            The correct shape for this array is `vertices.shape == 
            (2, number_of_vertices)`. 
        pixel_scale : float, meters
            The physical size of each pixel along one of its edges.

        Returns 
        -------
        x_offset, y_offset : float, meters
            The x and y offsets in physical units. 
        """
        x_offset = np.mean(vertices[:, 0])
        y_offset = np.mean(vertices[:, 1])
        return x_offset, y_offset


    # TODO: number_of_pixels can be moved out as a parameter
    def _rad_coordinates(self : Layer) -> tuple:
        """
        Generates the vectorised coordinate system associated with the 
        aperture.

        Parameters
        ----------
        phi_naught : float 
            The angle substending the first vertex. 

        Returns 
        -------
        rho, theta : tuple[Tensor]
            The stacked coordinate systems that are typically passed to 
            `_segments` to generate the segments.
        """
        cartesian = self._coordinates()
        positions = cartesian_to_polar(cartesian)

        # rho = positions[0] * self.get_pixel_scale()
        rho = positions[0]        

        theta = positions[1] 
        theta += 2 * np.pi * (positions[1] < 0.)
        theta += 2 * np.pi * (theta < self.alpha[0])

        rho = np.tile(rho, (6, 1, 1))
        theta = np.tile(theta, (6, 1, 1))
        return rho, theta


    def _edges(self : Layer, rho : Tensor, theta : Tensor) -> Tensor:
        """
        Generate lines connecting adjacent vertices.

        Parameters
        ----------
        rho : Tensor, meters
            Represents the radial distance of every point from the 
            centre of __this__ aperture. 
        theta : Tensor, Radians
            The angle associated with every point in the final bitmap.

        Returns
        -------
        edges : Tensor
            The edges represented as a Bitmap with the points inside the 
            edge marked as 1. and outside 0. The leading axis contains 
            each unique edge and the corresponding matrix is the bitmap.
        """
        # This is derived from the two point form of the equation for 
        # a straight line (eq. 1)
        # 
        #           y_2 - y_1
        # y - y_1 = ---------(x - x_1)
        #           x_2 - x_1
        # 
        # This is rearranged to the form, ay - bx = c, where:
        # - a = (x_2 - x_1)
        # - b = (y_2 - y_1)
        # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
        # we can then drive the transformation to polar coordinates in 
        # the usual way through the substitutions; y = r sin(theta), and 
        # x = r cos(theta). The equation is now in the form 
        #
        #                  c
        # r = ---------------------------
        #     a sin(theta) - b cos(theta) 
        #
        a = (self.x[1:] - self.x[:-1])
        b = (self.y[1:] - self.y[:-1])
        c = (a * self.y[:-1] - b * self.x[:-1])

        linear = c / (a * np.sin(theta) - b * np.cos(theta))
        #return rho < (linear * self.get_pixel_scale())
        return rho < linear


    def _wedges(self : Layer, theta : Tensor) -> Tensor:
        """
        The angular bounds of each segment of an individual hexagon.

        Parameters
        ----------
        theta : Tensor, Radians
            The angle away from the positive x-axis of the coordinate
            system associated with this aperture. Please note that `theta`
            May not start at zero. 

        Returns 
        -------
        wedges : Tensor 
            The angular bounds associated with each pair of vertices in 
            order. The leading axis of the Tensor steps through the 
            wedges in order arround the circle. 
        """
        # A wedge simply represents the angular bounds of the aperture
        # I have demonstrated below with a hexagon but understand that
        # these bounds are _purely_ angular (see fig 1.)
        #
        #               +-------------------------+
        #               |                ^^^^^^^^^|
        #               |     +--------+^^^^^^^^^^|
        #               |    /        /^*^^^^^^^^^|
        #               |   /        /^^^*^^^^^^^^|
        #               |  /        /^^^^^*^^^^^^^|
        #               | +        +^^^^^^^+^^^^^^|
        #               |  *              /       |
        #               |   *            /        |
        #               |    *          /         |
        #               |     +--------+          |
        #               +-------------------------+
        #               figure 1: The angular bounds 
        #                   between the zeroth and the 
        #                   first vertices. 
        #
        return (self.alpha[:-1] < theta) & (theta < self.alpha[1:])


    def _segments(self : Layer, theta : Tensor, rho : Tensor) -> Tensor:
        """
        Generate the segments as a stacked tensor. 

        Parameters
        ----------
        theta : Tensor
            The angle of every pixel associated with the coordinate system 
            of this aperture. 
        rho : Tensor
            The radial positions associated with the coordinate system 
            of this aperture. 

        Returns 
        -------
        segments : Tensor 
            The bitmaps corresponding to each vertex pair in the vertices.
            The leading dimension contains the unique segments. 
        """
        edges = self._edges(rho, theta)
        wedges = self._wedges(theta)
        return (edges & wedges).astype(float)
        

    def _aperture(self : Layer) -> Matrix:
        """
        Generate the BitMap representing the aperture described by the 
        vertices. 

        Returns
        -------
        aperture : Matrix 
            The Bit-Map that represents the aperture. 
        """
        # TODO: consider storing the vertices as a parameter to 
        # avoid reloading them every time. 
        rho, theta = self._rad_coordinates()
        segments = self._segments(theta, rho)
        return segments.sum(axis=0)


    def _load(self : Layer, segment : str):
        """
        Load the desired segment from the WebbPSF data. 

        Parameters
        ----------
        segment : str
            The segment that is desired to load. Should be in the 
            form "Ln". See the class doc string for more detail.

        Returns
        -------
        vertices : Matrix, meters
            The vertice information in any order with the shape
            (2, 6).
        """
        return dict(JWST_PRIMARY_SEGMENTS)[segment]



class JWSTPrimaryAperture(CompoundAperture):
    """
    Generates the full James Webb primary aperture. 
    """

    def __init__(self : Layer, number_of_pixels : int) -> Layer:
        """
        Generate the full primary aperture of the James Webb space 
        telescope. This constructor initialises default values for 
        the segements. This means that they are not rotated magnified
        or sheared. 

        Parameters
        ----------
        number_of_pixels : int
            The number of pixels to display the the entire aperture
            over.
        """
        pixel_scale = self._pixel_scale(number_of_pixels)

        SEGMENTS = [
            "A1-1", "A2-2", "A3-3", "A4-4", "A5-5", "A6-6", 
            "B1-7", "B2-9", "B3-11", "B4-13", "B5-15", "B6-17", 
            "C1-8", "C2-10", "C3-12", "C4-14", "C5-16", "C6-18"]

        apertures = dict()
        for segment in SEGMENTS:
            apertures[segment] = JWSTPrimaryApertureSegment(segment, 
                    number_of_pixels, pixel_scale)
        
        super().__init__(number_of_pixels, pixel_scale, apertures)
        

    def _pixel_scale(self : Layer, npix : int) -> float:
        """
        The physical dimesnions of a pixel along one edge. 

        Parameters
        ----------
        npix : int
            The number of pixels along one edge of the output image 

        Returns
        -------
        pixel_scale : float, meters
            The physical length along one edge of a pixel. 
        """
        B1 = JWST_PRIMARY_SEGMENTS[6][1]
        B4 = JWST_PRIMARY_SEGMENTS[9][1]
        height = B1[:, 1].max() - B4[:, 1].min()
        return height / npix


class Basis(eqx.Module):
    """
    _Abstract_ class representing a basis fixed over an aperture 
    that is used to optimise and learn aberations in the aperture. 
    
    Attributes
    ----------
    npix : int
        The number of pixels along the edge of the square array 
        representing each term in the basis.
    nterms : int
        The number of basis vectors to generate. This is determined
        by passing along the noll indices until the number is 
        reached (inclusive).
    aperture : Layer
        The aperture over which to generate the basis. This should 
        be an implementation of the abstract base class `Aperture`.
    """
    npix : int
    nterms : int    
    aperture : Layer


    def __init__(self : Layer, nterms : int,
            aperture : Layer) -> Layer:
        """
        Parameters
        ----------
        nterms : int
            The number of basis terms to generate. This determines the
            length of the leading dimension of the output Tensor.
        aperture : Layer
            The aperture to generate the basis over. This must be 
            an implementation of the abstract subclass `Aperture`. 
        """
        self.npix = int(aperture.get_npix())
        self.nterms = int(nterms)
        self.aperture = aperture


    def save(self : Layer, file_name : str, n : int) -> None:
        """
        Save the basis to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the basis terms to.
        n : int
            The number of terms in the basis to generate in the save.
        """
        basis = self._basis()
        with open(file_name, "w") as save:
            save.write(basis)


    @functools.partial(jax.vmap, in_axes=(None, 0))
    def _noll_index(self : Layer, j : int) -> tuple:
        """
        Decode the jth noll index of the zernike polynomials. This 
        arrises because the zernike polynomials are parametrised by 
        a pair numbers, e.g. n, m, but we want to impose an order.
        The noll indices are the standard way to do this see [this]
        (https://oeis.org/A176988) for more detail. The top of the 
        mapping between the noll index and the pair of numbers is 
        shown below:

        n, m Indices
        ------------
        (0, 0)
        (1, -1), (1, 1)
        (2, -2), (2, 0), (2, 2)
        (3, -3), (3, -1), (3, 1), (3, 3)

        Noll Indices
        ------------
        1
        3, 2
        5, 4, 6
        9, 7, 8, 10

        Parameters
        ----------
        j : int
            The noll index to decode.
        
        Returns
        -------
        n, m : tuple
            The n, m parameters of the zernike polynomial.
        """
        # To retrive the row that we are in we use the formula for 
        # the sum of the integers:
        #  
        #  n      n(n + 1)
        # sum i = -------- = x_{n}
        # i=0        2
        # 
        # However, `j` is a number between x_{n - 1} and x_{n} to 
        # retrieve the 0th based index we want the upper bound. 
        # Applying the quadratic formula:
        # 
        # n = -1/2 + sqrt(1 + 8x_{n})/2
        #
        # We know that n is an integer and hence of x_{n} -> j where 
        # j is not an exact solution the row can be found by taking 
        # the floor of the calculation. 
        #
        # n = (-1/2 + sqrt(1 + 8j)/2) // 1
        #
        # All the odd noll indices map to negative m integers and also 
        # 0. The sign can therefore be determined by -(j & 1). 
        # This works because (j & 1) returns the rightmost bit in 
        # binary representation of j. This is equivalent to -(j % 2).
        # 
        # The m indices range from -n to n in increments of 2. The last 
        # thing to do is work out how many times to add two to -n. 
        # This can be done by banding j away from the smallest j in 
        # the row. 
        #
        # The smallest j in the row can be calculated using the sum of
        # integers formula in the comments above with n = (n - 1) and
        # then adding one. Let this number be (x_{n - 1} + 1). We can 
        # then subtract j from it to get r = (j - x_{n - 1} + 1)
        #
        # The odd and even cases work differently. I have included the 
        # formula below:
        # odd : p = (j - x_{n - 1}) // 2 
       
        # even: p = (j - x_{n - 1} + 1) // 2
        # where p represents the number of times 2 needs to be added
        # to the base case. The 1 required for the even case can be 
        # generated in place using ~(j & 1) + 2, which is 1 for all 
        # even numbers and 0 for all odd numbers.
        #
        # For odd n the base case is 1 and for even n it is 0. This 
        # is the result of the bitwise operation j & 1 or alternatively
        # (j % 2). The final thing is adding the sign to m which is 
        # determined by whether j is even or odd hence -(j & 1).
        n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
        smallest_j_in_row = n * (n + 1) / 2 + 1 
        number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
        sign_of_shift = -(j & 1) + ~(j & 1) + 2
        base_case = (n & 1)
        m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
        return n, m


    def _radial_zernike(self : Layer, n : int, m : int,
            rho : Matrix) -> Tensor:
        """
        The radial zernike polynomial.

        Parameters
        ----------
        n : int
            The first index number of the zernike polynomial to forge
        m : int 
            The second index number of the zernike polynomial to forge.
        rho : Matrix
            The radial positions of the aperture. Passed as an argument 
            for speed.

        Returns
        -------
        radial : Tensor
            An npix by npix stack of radial zernike polynomials.
        """
        m, n = np.abs(m), np.abs(n)
        upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1
        rho = np.tile(rho, (MAX_DIFF, 1, 1))

        murder_weapon = (np.arange(MAX_DIFF) < upper)

        k = np.arange(MAX_DIFF) * murder_weapon
        coefficients = (-1) ** k * factorial(n - k) / \
            (factorial(k) * \
                factorial(((n + m) / 2).astype(int) - k) * \
                factorial(((n - m) / 2).astype(int) - k))
        radial = coefficients.reshape(MAX_DIFF, 1, 1) *\
            rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1) *\
            murder_weapon.reshape(MAX_DIFF, 1, 1)
         
        return radial.sum(axis=0)


    def _zernikes(self : Layer, coordinates : Tensor) -> Tensor:
        """
        Calculate the zernike basis on a square pixel grid. 

        Parameters
        ----------
        number : int
            The number of zernike basis terms to calculate.
            This is a static argument to jit because the array
            size depends on it.
        pixels : int
            The number of pixels along one side of the zernike image
            for each of the n zernike polynomials.
        coordinates : Tensor
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        zernike : Tensor 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        j = np.arange(1, self.nterms + 1).astype(int)
        n, m = self._noll_index(j)
        coordinates = cartesian_to_polar(coordinates)

        # NOTE: The idea is to generate them here at the higher level 
        # where things will not change and we will be done. 
        rho = coordinates[0]
        theta = coordinates[1]

        aperture = (rho <= 1.).astype(int)

        # In the calculation of the noll coefficient we must define 
        # between the m == 0 and and the m != 0 case. I have done 
        # this in place by casting the logical operation to an int. 

        normalisation_coefficients = \
            (1 + (np.sqrt(2) - 1) * (m != 0).astype(int)) \
            * np.sqrt(n + 1)

        radial_zernikes = np.zeros((self.nterms,) + rho.shape)
        for i in np.arange(self.nterms):
            radial_zernikes = radial_zernikes\
                .at[i]\
                .set(self._radial_zernike(n[i], m[i], rho))

        # When m < 0 we have the odd zernike polynomials which are 
        # the radial zernike polynomials multiplied by a sine term.
        # When m > 0 we have the even sernike polynomials which are 
        # the radial polynomials multiplies by a cosine term. 
        # To produce this result without logic we can use the fact
        # that sine and cosine are separated by a phase of pi / 2
        # hence by casting int(m < 0) we can add the nessecary phase.
        out_shape = (self.nterms, 1, 1)

        theta = np.tile(theta, out_shape)
        m = m.reshape(out_shape)
        phase_mod = (m < 0).astype(int) * np.pi / 2
        phase = np.cos(np.abs(m) * theta - phase_mod)

        normalisation_coefficients = \
            normalisation_coefficients.reshape(out_shape)
        
        return normalisation_coefficients * radial_zernikes \
            * aperture * phase 


    def _orthonormalise(self : Layer, aperture : Matrix, 
            zernikes : Tensor) -> Tensor:
        """
        The hexike polynomials up until `number_of_hexikes` on a square
        array that `number_of_pixels` by `number_of_pixels`. The 
        polynomials can be restricted to a smaller subset of the 
        array by passing an explicit `maximum_radius`. The polynomial
        will then be defined on the largest hexagon that fits with a 
        circle of radius `maximum_radius`. 
        
        Parameters
        ----------
        aperture : Matrix
            An array representing the aperture. This should be an 
            `(npix, npix)` array. 
        number_of_hexikes : int = 15
            The number of basis terms to generate. 
        zernikes : Tensor
            The zernike polynomials to orthonormalise on the aperture.
            This tensor should be `(nterms, npix, npix)` in size, where 
            the first axis represents the noll indexes. 

        Returns
        -------
        hexikes : Tensor
            The hexike polynomials evaluated on the square arrays
            containing the hexagonal apertures until `maximum_radius`.
            The leading dimension is `number_of_hexikes` long and 
            each stacked array is a basis term. The final shape is:
            ```py
            hexikes.shape == (number_of_hexikes, number_of_pixels, number_of_pixels)
            ```
        """
        pixel_area = aperture.sum()
        basis = np.zeros(zernikes.shape).at[0].set(aperture)
        
        for j in np.arange(1, self.nterms):
            intermediate = zernikes[j] * aperture

            coefficient = -1 / pixel_area * \
                (zernikes[j] * basis[1 : j + 1] * aperture)\
                .sum(axis = (1, 2))\
                .reshape(j, 1, 1) 

            intermediate += (coefficient * basis[1 : j + 1])\
                .sum(axis = 0)
            
            basis = basis\
                .at[j]\
                .set(intermediate / \
                    np.sqrt((intermediate ** 2).sum() / pixel_area))
        
        return basis


    def basis(self : Layer) -> Tensor:
        """
        Generate the basis. Requires a single run after which,
        the basis is cached and can be used with no computational 
        cost.  

        Parameters
        ----------
        aperture : Matrix
            The aperture over which the basis is to be generated. 
        coordinates : Matrix, meters, radians 
            The coordinate system over which to generate the aperture.

        Returns
        -------
        basis : Tensor
            The basis polynomials evaluated on the square arrays
            containing the apertures until `maximum_radius`.
            The leading dimension is `n` long and 
            each stacked array is a basis term. The final shape is:
            `(n, npix, npix)`
        """
        coordinates = self.aperture._coordinates()
        aperture = self.aperture._aperture()
        zernikes = self._zernikes(coordinates)
        return self._orthonormalise(aperture, zernikes)


class CompoundBasis(eqx.Module):
    """
    Interfaces with compound apertures to generate basis over them.
    """
    bases : list

 
    def __init__(self : Layer, nterms : int, 
            compound_aperture : Layer) -> Layer:
        """
        Parameters
        ----------
        nterms : int
            The number of basis terms to generate over each mirror.
            pass a list of integers in the order that the apertures
            appear in compound_aperture.
        compound_aperture : Layer
            The compound aperture to generate a basis over. 
        """
        apertures = compound_aperture.apertures.values()
        if isinstance(nterms, list):
            bases = [Basis(nterm, aperture) \
                for nterm, aperture in zip(nterms, apertures)]
        else:
            bases = [Basis(nterms, aperture) for aperture in apertures]
        self.bases = bases


    def basis(self : Layer) -> Tensor:
        """
        Generate a basis over a compound aperture.

        Returns 
        -------
        basis : Tensor
            The basis represented as `(napp, nterms, npix, npix)`
            array
        """
        return np.stack([basis.basis() for basis in self.bases])
