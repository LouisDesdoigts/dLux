import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dLux as dl
import jax

jax.config.update("jax_debug_nans", True)

nolls: float = np.arange(3, 9, dtype = int)
coeffs: float = np.ones((6,), dtype = float)
aperture: object = dl.CircularAperture(1.)
basis: object = dl.AberratedAperture(nolls, coeffs, aperture)
npix: int = 128
width: float = 2.
    def noll_index(self: ApertureLayer, j: int) -> tuple:
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


    def jth_radial_zernike(self: ApertureLayer, n: int, m: int) -> callable:
       """
       The radial zernike polynomial.

       Parameters
       ----------
       n : int
           The first index number of the zernike polynomial to forge
       m : int 
           The second index number of the zernike polynomial to forge.

       Returns
       -------
       radial : Array
           An npixels by npixels stack of radial zernike polynomials.
       """
       MAX_DIFF = 5
       m, n = np.abs(m), np.abs(n)
       upper = ((np.abs(n) - np.abs(m)) / 2).astype(int) + 1

       k = np.arange(MAX_DIFF)
       mask = (k < upper).reshape(MAX_DIFF, 1, 1)
       coefficients = (-1) ** k * dLux.utils.math.factorial(n - k) / \
           (dLux.utils.math.factorial(k) * \
               dLux.utils.math.factorial(((n + m) / 2).astype(int) - k) * \
               dLux.utils.math.factorial(((n - m) / 2).astype(int) - k))

       def _jth_radial_zernike(rho: list) -> list:
           rho = np.tile(rho, (MAX_DIFF, 1, 1))
           coefficients_out = coefficients.reshape(MAX_DIFF, 1, 1)
           rads = rho ** (n - 2 * k).reshape(MAX_DIFF, 1, 1)
           return (coefficients_out * mask * rads).sum(axis = 0)
               
       return _jth_radial_zernike


    def jth_polar_zernike(self: ApertureLayer, n: int, m: int) -> callable:
       """
       Generates a function representing the polar component 
       of the jth Zernike polynomial.

       Parameters
       ----------
       n: int 
           The first index number of the Zernike polynomial.
       m: int 
           The second index number of the Zernike polynomials.

       Returns
       -------
       polar: Array
           The polar component of the jth Zernike polynomials.
       """
       is_m_zero = (m != 0).astype(int)
       norm_coeff = (1 + (np.sqrt(2) - 1) * is_m_zero) * np.sqrt(n + 1)

       # When m < 0 we have the odd zernike polynomials which are 
       # the radial zernike polynomials multiplied by a sine term.
       # When m > 0 we have the even sernike polynomials which are 
       # the radial polynomials multiplies by a cosine term. 
       # To produce this result without logic we can use the fact
       # that sine and cosine are separated by a phase of pi / 2
       # hence by casting int(m < 0) we can add the nessecary phase.

       phase_mod = (m < 0).astype(int) * np.pi / 2

       def _jth_polar_zernike(theta: list) -> list:
           return norm_coeff * np.cos(np.abs(m) * theta - phase_mod)

       return _jth_polar_zernike  


    def jth_zernike(self: ApertureLayer, j: int) -> callable:
        """
        Calculate the zernike basis on a square pixel grid. 
     
        Parameters
        ----------
        noll_index: int
            The noll index corresponding to the zernike to generate.
            The first ten zernikes have been computed analytically 
            and are available via the `PreCompZernikeBasis` class. 
            This is only for doing zernike terms that are of higher 
            order and not centered.
     
        Returns
        -------
        zernike : Array 
            The zernike polynomials evaluated until number. The shape
            of the output tensor is number by pixels by pixels. 
        """
        n, m = self.noll_index(j)
     
        def _jth_zernike(coordinates: list) -> list:
            polar_coordinates = dLux.utils.cartesian_to_polar(coordinates)
            rho = polar_coordinates[0]
            theta = polar_coordinates[1]
            aperture = rho <= 1.
            _jth_rad_zern = self.jth_radial_zernike(n, m)
            _jth_pol_zern = self.jth_polar_zernike(n, m)
            return aperture * _jth_rad_zern(rho) * _jth_pol_zern(theta)
        
        return _jth_zernike 


    def jth_polike(self: ApertureLayer, j: int, n: int) -> callable:
        """
        The jth polike as a function. 
     
        Parameters
        ----------
        j: int
            The noll index of the requested zernike.
        n: int
            The number of sides on the regular polygon.
     
        Returns
        -------
        hexike: callable
            A function representing the jth hexike that is evaluated 
            on a cartesian coordinate grid. 
        """
        _jth_zernike = self.jth_zernike(j)
     
        def _jth_polike(coordinates: Array) -> Array:
            polar = dLux.utils.cartesian_to_polar(coordinates)
            rho = polar[0]
            alpha = np.pi / n
            phi = polar[1] + alpha 
            wedge = np.floor((phi + alpha) / (2. * alpha))
            u_alpha = phi - wedge * (2 * alpha)
            r_alpha = np.cos(alpha) / np.cos(u_alpha)
            return 1 / r_alpha * _jth_zernike(coordinates / r_alpha)
     
        return _jth_polike

basis.get_basis(npix, width)
