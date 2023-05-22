from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import dLux.utils as dlu
from jax import Array, vmap
import matplotlib.pyplot as plt
import dLux


Source = lambda : dLux.sources.BaseSource
Optics = lambda : dLux.core.BaseOptics


__all__ = ["AlphaCen"] #, "MixedAlphaCen"]


class AlphaCen(Source()):
    """
    
    """
    separation     : float
    x_position     : tuple
    y_position     : tuple
    position_angle : float
    log_flux       : float
    contrast       : float
    wavelengths    : Array
    weights        : Array

    def __init__(self, 
        nwavels = 5,
        x_position = 0., # arcseconds
        y_position = 0., # arcseconds
        separation = 10., # arcseconds
        position_angle = 90, # Degrees
        weights = None,
        log_flux = 5, # Photons TODO: Find true flux
        A_mag = 1.33,
        B_mag = 0.01,
        ):
        """
        
        """
        # Positional Parameters
        self.x_position = x_position
        self.y_position = y_position
        self.separation = separation
        self.position_angle = position_angle

        # Flux & Constrast
        self.log_flux = log_flux
        self.contrast = 10**((A_mag - B_mag)/2.5)

        # Spectrum (Uniform)
        self.wavelengths = np.linspace(530e-9, 640e-9, nwavels)
        if weights is None:
            self.weights = np.ones((2, nwavels))/nwavels
        else:
            self.weights = weights


    def normalise(self):
        """
        
        """
        return self.multiply('weights', 1/self.weights.sum(1)[:, None])
    

    @property
    def xy_positions(self):
        """
        
        """
        # Calculate
        r = self.separation/2
        phi = dlu.deg_to_rad(self.position_angle)
        sep_vec = np.array([r*np.sin(phi), r*np.cos(phi)])

        # Add to Position vectors
        pos_vec = np.array([self.x_position, self.y_position])
        output_vec = np.array([pos_vec + sep_vec, pos_vec - sep_vec])
        return dlu.arcsec_to_rad(output_vec)


    @property
    def raw_fluxes(self):
        """
        This casts log flux to be the total (not mean) flux.
        """
        flux = (10 ** self.log_flux) / 2
        flux_A = 2 * self.contrast * flux / (1 + self.contrast)
        flux_B = 2 * flux / (1 + self.contrast)
        return np.array([flux_A, flux_B])
    
    
    @property
    def norm_weights(self):
        """
        
        """
        return self.weights/self.weights.sum(1)[:, None]


    def model(self      : Source(),
              optics    : Optics()) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Get Values
        weights = self.norm_weights
        fluxes = self.raw_fluxes
        positions = self.xy_positions

        # vmap propagator
        source_propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        propagator = vmap(source_propagator, in_axes=(None, 0))

        # Model PSF
        input_weights = weights * fluxes[:, None]
        psfs = propagator(self.wavelengths, positions)
        psfs *= input_weights[..., None, None]
        return psfs.sum((0, 1))


def get_mixed_alpha_cen_spectra(
    nwavels    : int, 
    min_wavel  : float = 545e-9, 
    max_wavels : float = 645e-9
    ):
    """
    
    """
    # Import Here to prevent issues with google colab install, for example
    import pysynphot as S

    alpha_cen_a_spectrum: float = S.Icat("phoenix",
        5790, # Surface temp (K)
        0.2,  # Metalicity (Unit?)
        4.0,) # Surface gravity (unit?)
    alpha_cen_a_spectrum.convert('flam')
    alpha_cen_a_spectrum.convert('m')

    alpha_cen_b_spectrum: float = S.Icat("phoenix",
        5260, # Surface temp (K)
        0.23, # Metalicity (Unit?)
        4.37) # Surface gravity (unit?)
    alpha_cen_b_spectrum.convert('flam')
    alpha_cen_b_spectrum.convert('m')

    spot_spectrum: float = S.Icat("phoenix",
        4000, # Surface temp (K)
        0.23, # Metalicity (Unit?)
        4.37) # Surface gravity (unit?)
    spot_spectrum.convert('flam')
    spot_spectrum.convert('m')

    # Full spectrum
    wavelengths = np.linspace(545e-9, 645e-9, nwavels)
    Aspec = alpha_cen_a_spectrum.sample(wavelengths)
    Bspec = alpha_cen_b_spectrum.sample(wavelengths)
    Sspec = spot_spectrum.sample(wavelengths)

    Aspec /= Aspec.max()
    Bspec /= Bspec.max()
    Sspec /= Sspec.max()

    return np.array([Aspec, Bspec, Sspec]), wavelengths


