import jax.numpy as np
import dLux

__all__ = ["get_GE", "get_RGE", "get_RWGE", "get_radial_mask"]


Array = np.ndarray


def get_GE(array : Array) -> Array:
    """
    Calcuates the spatial gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calcuate the gradient energy for.

    Returns
    -------
    array : Array
        The array of gradient energies.
    """
    grads_vec = np.gradient(array)
    return np.hypot(grads_vec[0], grads_vec[1])


def get_RGE(array : Array, epsilon : float = 1e-8) -> Array:
    """
    Calcuates the spatial radial gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calcuate the radial gradient energy for.
    epsilon : float
        A small value added to the radial values to help with gradient
        stability.

    Returns
    -------
    array : Array
        The array of radial gradient energies.
    """
    positions = dLux.utils.coordinates.get_pixel_positions(array.shape[0])
    grads_vec = np.gradient(array)

    xnorm = positions[1]*grads_vec[0]
    ynorm = positions[0]*grads_vec[1]
    return np.square(xnorm + ynorm)


def get_RWGE(array : Array, epsilon : float = 1e-8) -> Array:
    """
    Calcuates the spatial radially weighted gradient energy of the array.

    Parameters
    ----------
    array : Array
        The array to calcuate the radially weighted gradient energy for.
    epsilon : float
        A small value added to the radially weighted values to help with
        gradient stability.

    Returns
    -------
    array : Array
        The array of radial radially weighted energies.
    """
    positions = dLux.utils.coordinates.get_pixel_positions(array.shape[0])
    radii = dLux.utils.coordinates.get_polar_positions(array.shape[0])[0]
    radii_norm = positions/(radii + epsilon)
    grads_vec = np.gradient(array)

    xnorm = radii_norm[1]*grads_vec[0]
    ynorm = radii_norm[0]*grads_vec[1]
    return np.square(xnorm + ynorm)


def get_radial_mask(npixels : int,
                    rmin    : Array,
                    rmax    : Array) -> Array:
    """
    Calcautes a binary radial mask, masking out radii below rmin, and above
    rmax.

    Parameters
    ----------
    npixels : int
        The linear size of the array.
    rmin : Array
        The inner radius to mask out.
    rmax : Array
        The outer radius to mask out.

    Returns
    -------
    mask: Array
        A mask with the the values below rmin and above rmax masked out.
    """
    radii = dLux.utils.coordinates.get_polar_positions(npixels)[0]
    return np.asarray((radii < rmax) & (radii > rmin), dtype=float)