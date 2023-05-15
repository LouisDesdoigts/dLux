from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import dLux.utils as dlu
from jax import Array, vmap
import matplotlib.pyplot as plt
import dLux


Optics = lambda : dLux.core.BaseOptics
MixedAlphaCen = lambda : dLux.models.MixedAlphaCen

__all__ = ["TolimanOptics", "ApplyBasisCLIMB", "TolimanSpikes", "Toliman"]

OpticalLayer = lambda : dLux.optical_layers.OpticalLayer
BaseBasisOptic = lambda : dLux.optical_layers.BaseBasisOptic
AngularOptics = lambda : dLux.optics.AngularOptics


class ApplyBasisCLIMB(BaseBasisOptic()):
    """
    Adds an array of binary phase values to the input wavefront from a set of
    continuous basis vectors. This uses the CLIMB algorithm in order to
    generate the binary values in a continous manner as described in the
    paper Wong et al. 2021. The basis vectors are taken as an Optical Path
    Difference (OPD), and applied to the phase of the wavefront. The ideal
    wavelength parameter described the wavelength that will have a perfect
    anti-phase relationship given by the Optical Path Difference.

    Note: Many of the methods in the class still need doccumentation.
    Note: This currently only outputs 256 pixel arrays and uses a 3x oversample,
    therefore requiring a 768 pixel basis array.

    Attributes
    ----------
    basis: Array
        Arrays holding the continous pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    ideal_wavelength : Array
        The target wavelength at which a perfect anti-phase relationship is
        applied via the OPD.
    """
    # basis            : Array
    # coefficients     : Array
    ideal_wavelength : Array


    def __init__(self             : OpticalLayer(),
                 basis            : Array,
                 ideal_wavelength : Array,
                 coefficients     : Array = None) -> OpticalLayer():
        """
        Constructor for the ApplyBasisCLIMB class.

        Parameters
        ----------
        basis : Array
            Arrays holding the continous pre-calculated basis vectors. This must
            be a 3d array of shape (nterms, npixels, npixels), with the final
            two dimensions matching that of the wavefront at time of
            application. This is currently required to be a nx768x768 shaped
            array. 
        ideal_wavelength : Array
            The target wavelength at which a perfect anti-phase relationship is
            applied via the OPD.
        coefficients : Array = None
            The Array of coefficients to be applied to each basis vector. This
            must be a one dimensional array with leading dimension equal to the
            leading dimension of the basis vectors. Default is None which
            initialises an array of zeros.
        """
        super().__init__(basis=basis, coefficients=coefficients)
        # self.basis            = np.asarray(basis, dtype=float)
        self.ideal_wavelength = np.asarray(ideal_wavelength, dtype=float)
        # self.coefficients     = np.array(coefficients).astype(float) \
        #             if coefficients is not None else np.zeros(len(self.basis))

        # # Inputs checks
        # assert self.basis.ndim == 3, \
        # ("basis must be a 3 dimensional array, ie (nterms, npixels, npixels).")
        # assert self.basis.shape[-1] == 768, \
        # ("Basis must have shape (n, 768, 768).")
        # assert self.coefficients.ndim == 1 and \
        # self.coefficients.shape[0] == self.basis.shape[0], \
        # ("coefficients must be a 1 dimensional array with length equal to the "
        # "First dimension of the basis array.")
        # assert self.ideal_wavelength.ndim == 0, ("ideal_wavelength must be a "
        #                                          "scalar array.")


    def __call__(self : OpticalLayer(), wavefront : Wavefront) -> Wavefront:
        """
        Generates and applies the binary OPD array to the wavefront in a
        differentiable manner.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the binary OPD applied.
        """
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent, ppsz=wavefront.npixels)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavelength)
        return wavefront.add_opd(opd)


    @property
    def applied_shape(self):
        return tuple(np.array(self.basis.shape[-2:])//3)


    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel


    def phase_to_opd(self, phase, wavel):
        return phase*wavel/(2*np.pi)


    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)


    def get_total_opd(self):
        return self.get_opd(self.basis, self.coefficients)


    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent)
        return binary_phase


    def lsq_params(self, img):
        xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),
                             np.linspace(0,1,img.shape[1]))
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
        return matrix, xx, yy, A


    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix,img.ravel())


    def area(self, img, epsilon = 1e-15):
        a,b,c = self.lsq(img)
        a = np.where(a==0,epsilon,a)
        b = np.where(b==0,epsilon,b)
        c = np.where(c==0,epsilon,c)
        x1 = (-b-c)/(a) # don't divide by zero
        x2 = -c/(a) # don't divide by zero
        x1, x2 = np.min(np.array([x1,x2])), np.max(np.array([x1,x2]))
        x1, x2 = np.max(np.array([x1,0])), np.min(np.array([x2,1]))

        dummy = x1 + (-c/b)*x2-(0.5*a/b)*x2**2 - (-c/b)*x1+(0.5*a/b)*x1**2

        # Set the regions where there is a defined gradient
        dummy = np.where(dummy>=0.5,dummy,1-dummy)

        # Colour in regions
        dummy = np.where(np.mean(img)>=0,dummy,1-dummy)

        # rescale between 0 and 1?
        dummy = np.where(np.all(img>0),1,dummy)
        dummy = np.where(np.all(img<=0),0,dummy)

        # undecided region
        dummy = np.where(np.any(img==0),np.mean(dummy>0),dummy)

        # rescale between 0 and 1
        dummy = np.clip(dummy, 0, 1)

        return dummy

    def CLIMB(self, wf, ppsz = 256):
        psz = ppsz * 3
        dummy = np.array(np.split(wf, ppsz))
        dummy = np.array(np.split(np.array(dummy), ppsz, axis = 2))
        subarray = dummy[:,:,0,0]

        flat = dummy.reshape(-1, 3, 3)
        vmap_mask = vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin



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
        zernikes = None,
        amplitude : float = 0.,
        seed : int = 0,

        m1_diameter = 0.13, # Double check this
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

        # Generate Aperture
        if zernikes is not None:
            # Set coefficients
            if amplitude == 0.:
                coefficients = np.zeros(len(zernikes))
            else:
                coefficients = amplitude * jr.normal(jr.PRNGKey(seed), 
                    (len(zernikes),))
        else:
            coefficients = None

        # Generate Aperture
        aperture = dLux.apertures.ApertureFactory(
            npixels         = wf_npixels,
            noll_indices    = zernikes,
            coefficients    = coefficients,
            secondary_ratio = m2_diameter/m1_diameter,
            nstruts         = nstruts,
            strut_ratio     = strut_width/m1_diameter)

        # Generate Mask
        if mask is None:
            path = "/Users/louis/PhD/Software/dLux/dLux/models/toliman/diffractive_pupil.npy"
            phase_mask = np.load(path)

            # Scale mask
            mask = dlu.scale_array(phase_mask, wf_npixels, order=1)

            # Enforce full binary
            small = np.where(mask <= 0.5)
            big = np.where(mask > 0.5)
            mask = mask.at[small].set(0.).at[big].set(np.pi)

            opd_mask = dlu.phase_to_opd(phase_mask, 595e-9)
            mask = dLux.optical_layers.AddOPD(opd_mask)

        # Propagator Properties
        # Test default float input
        psf_npixels = int(psf_npixels)
        psf_oversample = float(psf_oversample)
        psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, mask=mask, psf_npixels=psf_npixels, 
            psf_oversample=psf_oversample, psf_pixel_scale=psf_pixel_scale)


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


class TolimanSpikes(TolimanOptics):
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

        ) -> TolimanOptics:
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


class Toliman(dLux.instruments.BaseInstrument):
    source : None
    optics : None
    
    def __init__(self, optics, source):
        self.optics = optics
        self.source = source
        super().__init__()
    
    def __getattr__(self, key):
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        # if key in self.sources.keys():
        #     return self.sources[key]
        raise AttributeError(f"{self.__class__.__name__} has no attribute "
        f"{key}.")
    
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