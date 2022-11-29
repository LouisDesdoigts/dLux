import dLux as dl
from typing import TypeVar
import functools
import jax.numpy as np
import equinox as eqx
import jax
from dLux.utils import (get_positions_vector, get_pixel_positions)

__all__ = ['Basis', "CompoundBasis"]

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


@functools.partial(jax.jit, static_argnums=2)
def jit_safe_slice(arr: list, entry: tuple, lengths: tuple) -> list:
    """
    Take a slice of the array that has the shape `lengths`.
    The top left corner of this slice is at `entry` in `arr`.
    This re-`jit`s the function for each new `lengths` given. 

    Parameters:
    -----------
    arr: list
        The array/tensor to slice.
    entry: tuple
        The set of coordinates to enter the array at. 
    lengths: tuple
        The length along each dimension to slice. 

    Returns:
    --------
    slice: list
        The requested slice. 
    """
    return jax.lax.dynamic_slice(arr, entry, lengths)

class Basis(eqx.Module):
    """
    _Abstract_ class representing a basis fixed over an aperture 
    that is used to optimise and learn aberations in the aperture. 
    
    Attributes
    ----------
    nterms : int
        The number of basis vectors to generate. This is determined
        by passing along the noll indices until the number is 
        reached (inclusive).
    aperture : Layer
        The aperture over which to generate the basis. This should 
        be an implementation of the abstract base class `Aperture`.
    is_computed: bool
        A simple caching mechanism. If `is_computed` is `True` then
        the aperture basis has been calculated and is stored. 
    """
    nterms: int    
    aperture: Layer


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
        self.nterms = int(nterms)
        self.aperture = aperture


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

        # So the zernikes need to be generated on the smallest circle that contains 
        # the aperture. This code makes a normalised set of coordinates centred 
        # on the centre of the aperture with 1. at the largest extent. 
        coordinates = self.aperture.compute_aperture_normalised_coordinates(coordinates)

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
        
        middle_zernike = normalisation_coefficients * radial_zernikes \
            * aperture * phase 

        return middle_zernike 


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
        shape = zernikes.shape
        width = shape[-1]
        basis = np.zeros(shape).at[0].set(aperture)

        for j in np.arange(1, self.nterms):
            intermediate = zernikes[j] * aperture
            coefficient = np.zeros((nterms, 1, 1), dtype=float)
            mask = (np.arange(1, self.nterms) > j + 1).reshape((-1, 1, 1))

            coefficient = -1 / pixel_area * \
                (zernikes[j] * basis[1:] * aperture * mask)\
                .sum(axis = (1, 2))\
                .reshape(-1, 1, 1) 

            intermediate += (coefficient * basis[1:] * mask).sum(axis = 0)
            
            basis = basis\
                .at[j]\
                .set(intermediate / \
                    np.sqrt((intermediate ** 2).sum() / pixel_area))
        
        return basis


    # NOTE: I can tell that this function is not going to be jitable.
    # I say this because there are side effects from within 
    # `_compute` that `jax` will block. This destroys the cache 
    # mechanism, which I can now get rid of anyway.
    @property
    def basis(self : Layer, coordinates : Array) -> Tensor:
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
        -------9
        basis : Tensor
            The basis polynomials evaluated on the square arrays
            containing the apertures until `maximum_radius`.
            The leading dimension is `n` long and 
            each stacked array is a basis term. The final shape is:
            `(n, npix, npix)`
        """
        aperture = self.aperture._aperture(coordinates)
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


    def basis(self : Layer, coordinates : Array) -> Tensor:
        """
        Generate a basis over a compound aperture.
        
        Parameters
        ----------
        coordinates : Matrix, meters, radians 
            The coordinate system over which to generate the aperture.

        Returns 
        -------
        basis : Tensor
            The basis represented as `(napp, nterms, npix, npix)`
            array
        """
        return np.sum(np.stack([basis_vector.basis(coordinates) for basis_vector in self.bases]), axis=0)

import matplotlib as mpl
import matplotlib.pyplot as plt 

mpl.rcParams["text.usetex"] = True

pixels = 128
nterms = 6

coordinates = dl.utils.get_pixel_coordinates(pixels, 2. / pixels)
print(coordinates.shape)
sq_aperture = dl.SquareAperture(0., 0., 0., 1., False, False)
sq_basis = Basis(nterms, sq_aperture)
sq_basis_vecs = sq_basis.basis(coordinates) 

fig, axes = plt.subplots(2, 3)
axes[0][0].set_title("$j = 0$")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
_map = axes[0][0].imshow(sq_basis_vecs[0])
fig.colorbar(_map, ax=axes[0][0])
axes[0][1].set_title("$j = 1$")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
_map = axes[0][1].imshow(sq_basis_vecs[1])
fig.colorbar(_map, ax=axes[0][1])
axes[0][2].set_title("$j = 2$")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
_map = axes[0][2].imshow(sq_basis_vecs[2])
fig.colorbar(_map, ax=axes[0][2])
axes[1][0].set_title("$j = 3$")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
_map = axes[1][0].imshow(sq_basis_vecs[3])
fig.colorbar(_map, ax=axes[1][0])
axes[1][1].set_title("$j = 4$")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
_map = axes[1][1].imshow(sq_basis_vecs[4])
fig.colorbar(_map, ax=axes[1][1])
axes[1][2].set_title("$j = 5$")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
_map = axes[1][2].imshow(sq_basis_vecs[5])
fig.colorbar(_map, ax=axes[1][2])
plt.show()
