from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import dLux.utils as dlu
from jax import Array, vmap
import matplotlib.pyplot as plt
import dLux


Optics = lambda : dLux.core.BaseOptics
MixedAlphaCen = lambda : dLux.models.MixedAlphaCen

__all__ = ["SimpleToliman", "TolimanSpikes", "Toliman"]


class SimpleToliman(Optics()):
    """
    A model of the Toliman optical system.

    Its default parameters are:

    """
    diameter        : Array
    aperture        : Union[Array, TransmissiveOptic()]
    mask            : Union[Array, AberrationLayer()]
    aberrations     : Union[Array, AberrationLayer()]
    psf_npixels     : int
    psf_oversample  : float
    psf_pixel_scale : float

    def __init__(self, 

        wf_npixels = 256,
        psf_npixels = 256,
        psf_oversample = 2,
        psf_pixel_scale = 0.375, # arcsec

        mask = None,
        zernikes = None,
        amplitude : float = 0.,
        seed : int = 0,

        m1_diameter = 0.13, # Double check this
        m2_diameter = 0.032,
        
        nstruts = 3,
        strut_width = 0.002,
        strut_rotation=-np.pi/2

        ) -> SimpleToliman:
        """
        Constructs a simple model of the Toliman Optical Systems

        In this class units are different:
        - psf_pixel_scale is in unit of arcseconds
        """

        # Diameter
        self.diameter = m1_diameter

        # Generate Aperture
        self.aperture = dLux.apertures.ApertureFactory(
            npixels         = wf_npixels,
            secondary_ratio = m2_diameter/m1_diameter,
            nstruts         = nstruts,
            strut_ratio     = strut_width/m1_diameter,
            name            = "Aperture").transmission

        # Generate Mask
        if mask is None:
            phase_mask = np.load("pupil.npy")

            # Scale mask
            mask = dlu.scale_array(phase_mask, wf_npixels, order=1)

            # Enforce full binary
            small = np.where(mask <= 0.5)
            big = np.where(mask > 0.5)
            mask = mask.at[small].set(0.).at[big].set(np.pi)

            opd_mask = dlu.phase_to_opd(phase_mask, 595e-9)
            self.mask = dLux.optics.AddOPD(opd_mask)
        
        # Allow for arbitrary mask layers
        else:
            self.mask = mask

        # Generate Aberrations
        if zernikes is None:
            self.aberrations = None
        else:
            # Set coefficients
            if amplitude == 0.:
                coefficients = np.zeros(len(zernikes))
            else:
                coefficients = amplitude * jr.normal(jr.PRNGKey(seed), 
                    (len(zernikes),))
            
            # Construct Aberrations
            self.aberrations = dLux.aberrations.AberrationFactory(
                npixels      = wf_npixels,
                zernikes     = zernikes,
                coefficients = coefficients,
                name         = "Aberrations")

        # Propagator Properties
        # Test default float input
        self.psf_npixels = int(psf_npixels)
        self.psf_oversample = float(psf_oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)

        super().__init__()


    def _construct_wavefront(self       : Optics(),
                             wavelength : Array,
                             offset     : Array = np.zeros(2)) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        
        Returns
        -------
        wavefront : Wavefront
            The wavefront object to propagate through the optics.
        """
        wf_constructor = dLux.wavefronts.Wavefront
        
        # Construct and tilt
        wf = wf_constructor(self.aperture.shape[-1], self.diameter, wavelength)
        return wf.tilt_wavefront(offset)


    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        # Construct and tilt
        wf = dLux.wavefronts.Wavefront(self.aperture.shape[-1], self.diameter, 
            wavelength)
        wf = wf.tilt_wavefront(offset)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply mask
        wf *= self.mask

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcseconds_to_radians(pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale_radians)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


    def plot_aperture(self):
        """
        
        """
        from matplotlib import colormaps
        cmap = colormaps['inferno']
        cmap.set_bad('k',0.8)

        inv_support = np.where(self.aperture < 0.5)

        if self.aberrations is None:
            aperture = self.aperture.at[inv_support].set(np.nan)
        else:
            aberrations = self.aberrations.get_opd()
            aperture = aberrations.at[inv_support].set(np.nan)

        mask = self.mask.opd.at[inv_support].set(np.nan)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Aperture")
        plt.imshow(aperture * 1e9, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label(r'OPD [nm]')

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask * 1e9, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label(r'OPD [nm]')
        plt.show()


class TolimanSpikes(SimpleToliman):
    """
    A model of the Toliman optical system.

    Its default parameters are:

    """
    grating_depth  : float
    grating_period : float
    spike_npixels  : int

    def __init__(self, 

        wf_npixels = 256,
        psf_npixels = 256,
        psf_oversample = 2,
        psf_pixel_scale = 0.375, # arcsec
        spike_npixels = 512,

        mask = None,
        zernikes = None,
        amplitude : float = 0.,
        seed : int = 0,

        m1_diameter = 0.13, # Double check this
        m2_diameter = 0.032,
        
        nstruts = 3,
        strut_width = 0.002,
        strut_rotation=-np.pi/2,

        grating_depth = 100., # nm
        grating_period = 300, # um

        ) -> SimpleToliman:
        """
        Constructs a simple model of the Toliman Optical Systems

        In this class units are different:
        - psf_pixel_scale is in unit of arcseconds
        grating depth is in nm
        grating period is in um
        """

        # Diameter
        self.grating_depth = grating_depth
        self.grating_period = grating_period
        self.spike_npixels = spike_npixels

        super().__init__(
            wf_npixels      = wf_npixels, 
            psf_npixels     = psf_npixels, 
            psf_oversample  = psf_oversample, 
            psf_pixel_scale = psf_pixel_scale, 
            mask            = mask, 
            zernikes        = zernikes, 
            amplitude       = amplitude, 
            seed            = seed, 
            m1_diameter     = m1_diameter, 
            m2_diameter     = m2_diameter, 
            nstruts         = nstruts, 
            strut_width     = strut_width, 
            strut_rotation  = strut_rotation
            )


    def model_spike(self, wavelengths, offset, weights, angles, sign, center):
        """
        
        """
        propagator = vmap(self.model_spike_mono, (0, None, 0, None, None))
        psfs = propagator(wavelengths, offset, angles, sign, center)
        psfs *= weights[..., None, None]
        return psfs.sum(0)


    def model_spike_mono(self, wavelength, offset, angle, sign, center):
        """
        
        """
        # Construct and tilt
        wf = dLux.wavefronts.Wavefront(self.aperture.shape[-1], self.diameter, 
            wavelength)

        # Addd offset and tilt
        wf = wf.tilt_wavefront(offset - sign * angle)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        shift = sign * center
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        wf = wf.shifted_MFT(self.spike_npixels, pixel_scale, shift=shift)

        # Return PSF
        return wf.psf


    def get_diffraction_angles(self, wavelenghts):
        """
        
        """
        period = self.grating_period * 1e-6 # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2) # Radians
        return dlu.radians_to_arcseconds(angles)
    

    def model_spikes(self, wavelengths, offset, weights):
        """
        
        """
        # Get center shift values
        period = self.grating_period * 1e-6 # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2) # Radians
        # angles = get_diffraction_angles(wavelengths)
        true_pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale = dlu.arcseconds_to_radians(true_pixel_scale)
        center = angles.mean(0)//pixel_scale

        # Model
        signs = np.array([[-1, +1], [+1, +1], [-1, -1], [+1, -1]])
        propagator = vmap(self.model_spike, (None, None, None, None, 0, None))
        return propagator(wavelengths, offset, weights, angles, signs, center)


    def full_model(self, source, cent_nwavels=5):
        """
        Returns the diffraction spikes of the PSF

        source should be an MixedAplhaCen object
        """
        if not isinstance(source, MixedAlphaCen()):
            raise TypeError("source must be a MixedAlphaCen object")
        
        # Get Values
        wavelengths = source.wavelengths
        weights = source.norm_weights
        fluxes = source.raw_fluxes
        positions = source.xy_positions
        fratio = source.mixing

        # Calculate relative fluxes
        # TODO: Translate grating depth to central vs corner flux
        # Requires some experimental  mathematics
        # Probably requires both period and depth
        central_flux = 0.8
        corner_flux = 0.2

        # Model Central
        # TODO: Downsample central wavelengths and weights
        central_wavelegths = wavelengths
        central_weights = weights
        propagator = vmap(self.propagate, in_axes=(None, 0, 0))
        central_psfs = propagator(
            central_wavelegths, 
            positions, 
            central_weights)
        central_psfs *= central_flux * fluxes[:, None, None]

        # Model spikes
        propagator = vmap(self.model_spikes, in_axes=(None, 0, 0))
        spikes = propagator(wavelengths, positions, weights)
        spikes *= corner_flux * fluxes[:, None, None, None] / 4

        # Return
        return central_psfs.sum(0), spikes.sum(0)


class Toliman(dLux.core.BaseInstrument):
    source : None
    optics : None
    
    def __init__(self, optics, source):
        self.optics = optics
        self.source = source
        super().__init__()
    
    def __getattr__(self, key):
        if hasattr(self.source, key):
            return getattr(self.source, key)
        elif hasattr(self.optics, key):
            return getattr(self.optics, key)
        else:
            raise AttributeError(f"Neither source nor optics have attribute "
                f"{key}")
    
    def normalise(self):
        return self.set('source', self.source.normalise())
    
    def model(self):
        return self.optics.model(self.source)

    def full_model(self):
        return self.optics.full_model(self.source)
    
    def perturb(self, X, parameters):
        for parameter, x in zip(parameters, X):
            self = self.add(parameter, x)
        return self