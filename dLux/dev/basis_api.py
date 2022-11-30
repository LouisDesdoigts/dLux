import jax 
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


class HardCodedLowOrderZernikeBasisAndAperture(eqx.Module):
    """
    A selection of the low order (< 10) zernike terms. 
    This class is the fastest when it comes to applying 
    the basis terms. 

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
    aperture: Layer


    def __init__(self, noll_inds: list, coeffs: list, aperture: list):
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
        aperture: Layer
            A `CircularAperture` within which the aberrations are 
            being studied. 
        """
        self.zernikes = [zernikes[ind] for ind in noll_inds]
        self.coeffs = np.asarray(coeffs).astype(float)
        self.aperture = aperture

        assert len(noll_inds) == len(coeffs)
        assert isinstance(aperture, dl.CircularAperture)


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
        cart_coords: list = wavefront.pixel_positions()
        pol_coords: list = dl.cartesian_to_polar(car_coords)
        rho: list = pol_coords[0]
        theta: list = pol_coords[1]
        basis: list = np.stack([z(rho, theta) for z in zernikes])
        aperture: Array = self.aperture._aperture(cart_coords)
        opd: list = np.dot(basis.T, self.coeffs) * aperture
        params_dict["Wavefront"] = wavefront\
            .add_opd(opd)\
            .multiply_amplitude(aperture)
        return params_dict
