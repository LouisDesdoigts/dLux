import jax 
import abc 
import jax.numpy as np
import equinox as eqx
import dLux as dl
import typing 


Array = typing.TypeVar("Array")
Layer = typing.TypeVar("Layer")


zernikes: list = [
    lambda rho, theta: np.ones(rho.shape, dtype=float),
    lambda rho, theta: 2. * rho * np.sin(theta),
    lambda rho, theta: 2. * rho * np.cos(theta),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.sin(2. * theta),
    lambda rho, theta: np.sqrt(3.) * (2. * rho ** 2 - 1.),
    lambda rho, theta: np.sqrt(6.) * rho ** 2 * np.cos(2. * theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.sin(3. * theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * (3. * rho ** 3 - 2. * rho) * np.sin(theta),
    lambda rho, theta: np.sqrt(8.) * rho ** 3 * np.cos(3. * theta)
]


class AbberatedAperture(dl.OpticalLayer, abc.ABC):
    """
    An abstract base class representing an `Aperture` defined
    with a basis. The basis is a set of polynomials that are 
    orthonormal over the surface of the aperture (usually). 
    These can be used to represent any aberation on the surface
    of the aperture. In general, the basis should only be defined 
    on apertures that have a surface such as a mirror or phase 
    plate ect. It isn't really possible to have aberrations on 
    an opening. This rule may be broken to learn the atmosphere 
    above a telescope but whether or not this is a good idea 
    remains to be seen.

    Parameters:
    -----------
    basis_funcs: list[callable]
        A list of functions that represent the basis. The exact
        polynomials that are represented will depend on the shape
        of the aperture. 
    aperture: Aperture
        The aperture on which the basis is defined. Must be a 
        subclass of the `Aperture` class.
    coeffs: list[floats]
        The coefficients of the basis terms. By learning the 
        coefficients only the amount of time that is required 
        for the learning process is significantly reduced.
    """
    basis_funcs: List[callable]
    aperture: Aperture
    coeffs: Array


    @abc.abstractmethod
    def __init__(self   : Layer, 
            noll_inds   : List[int],
            aperture    : Aperture, 
            coeffs      : Array) -> Layer:
        """
        Parameters:
        -----------
        noll_inds: List[int]
            The noll indices are a scheme for indexing the Zernike
            polynomials. Normally these polynomials have two 
            indices but the noll indices prevent an order to 
            these pairs. All basis can be indexed using the noll
            indices based on `n` and `m`. 
        aperture: Aperture
            The aperture that the basis is defined on. The shape 
            of this aperture defines what the polynomials are. 
        coeffs: Array
            The coefficients of the basis vectors. 
        """


    @abc.abstractmethod
    def _basis(self: Layer, coords: Array) -> Array:
        """
        Generate the basis vectors over a set of coordinates.  

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. 
        """


    def _opd(self: Layer, coords: Array) -> Array:
        """
        Calculate the optical path difference that is caused 
        by the basis and the aberations that it represents. 

        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        opd: Array
            The optical path difference associated with much of 
            the path. 
        """
        basis: Array = self._basis(coords)
        opd: Array = np.dot(basis.T, self.coeffs)
        return opd


    @abc.abstractmethod
    def __call__(self, params_dict: dict) -> dict:
        """
        Apply the aperture and the abberations to the wavefront.  

        Parameters:
        -----------
        params: dict
            A dictionary containing the key "Wavefront".

        Returns:
        --------
        params: dict 
            A dictionary containing the key "wavefront".
        """


class AberratedCircularAperture(AberratedAperture):
    """
    Parameters:
    -----------
    zernikes: Array
        An array of `jit` compiled zernike basis functions 
        that operate on a set of coordinates. In particular 
        these coordinates correspond to a normalised set 
        of coordinates that are centered at the the centre 
        of the circular aperture with 1. occuring along the 
        radius. 
    coeffs: Array
        The coefficients of the Zernike terms. 
    aperture: Layer
        Must be an instance of `CircularAperture`. This 
        is applied alongside the basis. 
    """
    zernikes: Array
    coeffs: Array
    aperture: CircularAperture


    def __init__(self   : Layer, 
            noll_inds   : list, 
            coeffs      : list, 
            aperture    : CircularAperture):
        """
        Parameters:
        -----------
        noll_inds: Array 
            The noll indices of the zernikes that are to be mapped 
            over the aperture.
        coeffs: Array 
            The coefficients associated with the zernikes. These 
            should be ordered by the noll index of the zernike 
            that they refer to.
        aperture: CircularAperture
            A `CircularAperture` within which the aberrations are 
            being studied. 
        """
        self.zernikes = [zernikes[ind] if ind < 10 
            for ind in noll_inds else jth_zernike(ind)]
        self.coeffs = np.asarray(coeffs).astype(float)
        self.aperture = aperture

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.CircularAperture)


    def _basis(self: Layer, coords: Array) -> Array:
        """
        Parameters:
        -----------
        coords: Array, meters
            The paraxial coordinate system on which to generate
            the array. 

        Returns:
        --------
        basis: Array
            The basis vectors associated with the aperture. 
            These vectors are stacked in a tensor that is,
            `(nterms, npix, npix)`. Normally the basis is 
            cropped to be just on the aperture however, this 
            step is not necessary except for in visualisation. 
            It has been removed to save some time in the 
            calculations. 
        """
        pol_coords: list = dl.cartesian_to_polar(coords)
        rho: list = pol_coords[0]
        theta: list = pol_coords[1]
        basis: list = np.stack([z(rho, theta) for z in zernikes])
        return basis


    def __call__(self, params_dict: dict) -> dict:
        """
        Apply the aperture and the abberations to the wavefront.  

        Parameters:
        -----------
        params: dict
            A dictionary containing the key "Wavefront".

        Returns:
        --------
        params: dict 
            A dictionary containing the key "wavefront".
        """
        wavefront: object = params_dict["Wavefront"]
        coords: Array = wavefront.pixel_positions()
        opd: Array = self._opd(coords)
        aperture: Array = self.aperture._aperture(coords)
        params_dict["Wavefront"] = wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)
        return params_dict




