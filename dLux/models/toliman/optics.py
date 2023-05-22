from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import dLux.utils as dlu
from jax import Array, vmap
import dLux


Optics = lambda : dLux.core.BaseOptics
MixedAlphaCen = lambda : dLux.models.MixedAlphaCen

__all__ = ["TolimanOptics"]

OpticalLayer = lambda : dLux.optical_layers.OpticalLayer
AngularOptics = lambda : dLux.optics.AngularOptics


class TolimanOptics(AngularOptics()):
    """
    A model of the Toliman optical system.

    Its default parameters are:

    """

    def __init__(self, 

        wf_npixels = 256,
        psf_npixels = 256,
        psf_oversample = 2,
        psf_pixel_scale = 0.375, # arcsec

        mask = None,

        radial_orders    : Array = None,
        noll_indices     : Array = None,
        coefficients = None,
        # amplitude : float = 0.,
        # seed : int = 0,

        m1_diameter = 0.125,
        m2_diameter = 0.032,
        
        nstruts = 3,
        strut_width = 0.002,
        strut_rotation=-np.pi/2

        ) -> TolimanOptics:
        """
        Constructs a simple model of the Toliman Optical Systems

        In this class units are different:
        - psf_pixel_scale is in unit of arcseconds
        """

        # Diameter
        diameter = m1_diameter

        # # Generate Aperture
        # if zernikes is not None:
        # Set coefficients
        # if amplitude != 0.:
        #     # coefficients = /np.zeros(len(zernikes))
        # # else:
        #     coefficients = amplitude * jr.normal(jr.PRNGKey(seed), 
        #         (len(zernikes),))
        # else:
        #     coefficients = None

        # Generate Aperture
        aperture = dLux.apertures.ApertureFactory(
            npixels         = wf_npixels,
            radial_orders   = radial_orders,
            noll_indices    = noll_indices,
            coefficients    = coefficients,
            secondary_ratio = m2_diameter/m1_diameter,
            nstruts         = nstruts,
            strut_ratio     = strut_width/m1_diameter)

        # Generate Mask
        if mask is None:
            path = ("/Users/louis/PhD/Software/dLux/dLux/models/toliman/"
                "diffractive_pupil.npy")
            mask = dlu.scale_array(np.load(path), wf_npixels, order=1)
            
            # Enforce full binary
            mask = mask.at[np.where(mask <= 0.5)].set(0.)
            mask = mask.at[np.where(mask > 0.5)].set(1.)

            # Enforce full binary
            mask = dlu.phase_to_opd(mask * np.pi, 585e-9)

        # Propagator Properties
        psf_npixels = int(psf_npixels)
        psf_oversample = float(psf_oversample)
        psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, mask=mask, psf_npixels=psf_npixels, 
            psf_oversample=psf_oversample, psf_pixel_scale=psf_pixel_scale)


    def _apply_aperture(self, wavelength, offset):
        """
        Overwrite so mask can be stored as array
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf += self.mask
        return wf