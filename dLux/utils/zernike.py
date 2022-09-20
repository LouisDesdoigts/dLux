"""
This code has been ported from the poppy zernike.py file to work with jax
"""

__all__ = ["zernike_basis", "zern_name"]

import jax.numpy as np
from math import factorial

def zernike_basis(nterms=15, npix=512, rho=None, theta=None, **kwargs):
    """
    Return a cube of Zernike terms from 1 to N each as a 2D array
    showing the value at each point. (Regions outside the unit circle on which
    the Zernike is defined are initialized to zero.)

    Parameters
    -----------
    nterms : int, optional
        Number of Zernike terms to return, starting from piston.
        (e.g. ``nterms=1`` would return only the Zernike piston term.)
        Default is 15.
    npix : int
        Desired pixel diameter for circular pupil. Only used if `rho`
        and `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.

    Other parameters are passed through to `poppy.zernike.zernike`
    and are documented there.
    """
    if rho is not None and theta is not None:
        # both are required, but validated in zernike1
        shape = rho.shape
        use_polar = True
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    else:
        shape = (npix, npix)
        use_polar = False
    zern_output = np.zeros([nterms, npix, npix])

    if use_polar:
        for j in range(nterms):
            zern_output = zern_output.at[j].set(zernike1(j + 1, rho=rho, theta=theta, **kwargs))
    else:
        for j in range(nterms):
            zern_output = zern_output.at[j].set(zernike1(j + 1, npix=npix, **kwargs))
    return zern_output

def zernike1(j, **kwargs):
    """ Return the Zernike polynomial Z_j for pupil points {r,theta}.

    For this function the desired Zernike is specified by a single index j.
    See zernike for an equivalent function in which the polynomials are
    ordered by two parameters m and n.

    Note that there are multiple contradictory conventions for labeling Zernikes
    with one single index. We follow that of Noll et al. JOSA 1976.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of
        Noll et al. JOSA 1976

    Additional arguments are defined as in `poppy.zernike.zernike`.

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta)
    """
    n, m = noll_indices(j)
    return zernike(n, m, **kwargs)

def noll_indices(j):
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    """

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    # n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    n = (np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1).astype(int)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign
    return n, m

def zernike(n, m, npix=100, rho=None, theta=None, outside=0.,
            noll_normalize=True, **kwargs):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:

        zernike(n, m, npix)
            where npix specifies a pupil diameter in pixels.
            The returned pupil will be a circular aperture
            with this diameter, embedded in a square array
            of size npix*npix.

        zernike(n, m, rho=r, theta=theta)
            Which explicitly provides the desired pupil coordinates
            as arrays r and theta. These need not be regular or contiguous.

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix : int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")

    if theta is None and rho is None:
        x = (np.arange(npix) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    if not np.all(rho.shape == theta.shape):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    aperture = (rho <= 1)

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture
    zernike_result = zernike_result.astype(float)
    zernike_result = zernike_result.at[np.where(rho > 1.)].set(outside)
    
    return zernike_result

def _is_odd(integer):
    """Helper for testing if an integer is odd by bitwise & with 1."""
    return integer & 1

def R(n, m, rho):
    """Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordnates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
        
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial(int((n + m) / 2) - k) *\
                        factorial(int((n - m) / 2) - k)))
            output += coef * rho ** (n - 2 * k)
            
        return output

def zern_name(i):
    """Return a human-readable text name corresponding to some Zernike term as specified
    by `j`, the index
    Only works up to term 22, i.e. 5th order spherical aberration.
    """
    names = ['Null', 'Piston', 'Tilt X', 'Tilt Y',
             'Focus', 'Astigmatism 45', 'Astigmatism 0',
             'Coma Y', 'Coma X',
             'Trefoil Y', 'Trefoil X',
             'Spherical', '2nd Astig 0', '2nd Astig 45',
             'Tetrafoil 0', 'Tetrafoil 22.5',
             '2nd coma X', '2nd coma Y', '3rd Astig X', '3rd Astig Y',
             'Pentafoil X', 'Pentafoil Y', '5th order spherical']

    if i < len(names):
        return names[i]
    else:
        return "Z%d" % i