import jax 
import jax.numpy as np
import equinox as eqx


zernikes: list = [
    lambda rho, theta: 1.,
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


class PreCompZernikeBasis(eqx.Module):
    zernikes: list

    def __init__(self, noll_inds: list):
        self.zernikes = [zernikes[ind] for ind in noll_inds]


    def __call__(self, params_dict: dict) -> dict:
        wavefront: object = params_dict["Wavefront"]
        cart_coords: list = wavefront.pixel_positions()
        pol_coords: list = dl.cartesian_to_polar(car_coords)
        rho: list = pol_coords[0]
        theta: list = pol_coords[1]
        basis: list = np.stack([z(rho, theta) for z in zernikes])

        # So maybe I am missing something. I need to make that 
        # Github issue. 
        
        
