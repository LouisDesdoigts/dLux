import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from jax import vmap
from equinox import Module

__all__ = ['Layer', 'OpticalSystem']

class Layer(Module):
    """ Base Layer class, Equinox Modle
    
    Only supports square arrays (n, n).
    
    Each child class can either be some optical or array operation, transform
    or neural network like operation. Thier __call__() function must be 
    differentiable in order for things to work and follow this formatting:
    
        def __call__(self, wavelength, wavefront, pixelscale, offset):
            # Do things
            return wavefront, pixelscale
    
    wavefront must be input as an array with shape (size_in, size_in) and it 
    must be returned with shape (size_out, size_out).
    
    The parameters size_in, size_out must be set in the __init__() class of
    children classes.
    
    Parameters
    ----------
    size_in: int, equinox.static_field
        defines the linear size of the input wavefront to the __call__()
        function
    size_out: int, equinox.static_field
        defines the linear size of the output wavefront to the __call__()
        function
    """
    size_in: int = static_field()
    size_out: int = static_field()
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        
class Wavefront(Module):
    """
    Class to store the optical information and paramaters
    """
    wavefront: ndarray
    wavel: float
    offset: ndarray
    pixelscale: float
    planetype: str
    
    def __init__(self, wavel, offset):
        self.wavel = wavel
        self.offset = offset
        self.planetype = "Pupil"
        self.pixelscale = None
        self.wavefront = None

    def wf2psf(self):
        return np.abs(self.wavefront)**2
    
    def get_XXYY(self, shift=0.):
        npix = self.wavefront.shape[0]
        xs = np.arange(-npix//2, npix//2) + shift
        YY, XX = np.meshgrid(xs, xs)
        return np.array([XX, YY])
    
    def get_xycoords(self, shift=0.):
        return self.pixelscale * self.get_XXYY(shift=shift)
    
class OpticalSystem(Module):
    """ Optical System class, Equinox Modle
    
    DOCSTRING NOT COMPLETE
    
    A Class to store and apply properties external to the optical system
    Ie: stellar positions and specturms
    
    positions: (Nstars, 2) array
    wavels: (Nwavels) array
    weights: (Nwavel)/(Nwavels, Nstars) array
    
    
    Notes:
     - Take in layers in order to re-intialise the model every call?
    
    
    General images output shape: (Nimages, Nstars, Nwavels, Npix, Npix)
    
     - Currently doesnt allow temporal variation in specturm becuase im lazy
     - Currently doesnt allow temporal variation in flux becuase im lazy
    
    ToDo: Add getter methods for acessing weights and fluxes attributes that
    use np.squeeze to remove empy axes
    
    """
    layers: list
    detector_layers: list
    wavels: ndarray
    positions: ndarray
    fluxes: ndarray
    weights: ndarray
    dithers: ndarray
    
    # Determined from inputs
    Nstars:  int = static_field()
    Nwavels: int = static_field()
    Nims:    int = static_field()
    
    # To Do - add asset conditions to ensure that everything is formatted correctly 
    # To Do - pass in positions for multiple images, ignoring dither (ie multi image)
    def __init__(self, layers, wavels=None, positions=None, fluxes=None, 
                       weights=None, dithers=None, detector_layers=None):
        # Required Inputs
        self.layers = layers
        self.wavels = np.array(wavels)
        
        # Set to default values
        self.positions = np.zeros(2) if positions is None else np.array(positions)
        self.fluxes = np.ones(1) if fluxes is None else np.array(fluxes)
        self.dithers = np.zeros([1, 2]) if dithers is None else dithers
        self.detector_layers = [] if detector_layers is None else detector_layers
        
        # Determined from inputs
        self.Nstars = len(self.positions)
        self.Nwavels = len(self.wavels)
        self.Nims = len(self.dithers)
        
        # Format weights
        if wavels is None:
            self.weights = np.ones([1, 1, 1, 1, 1])
        elif weights is None:
            # Each star has the same uniform spectrum
            self.weights = np.ones([1, 1, len(wavels), 1, 1])
        elif len(weights.shape) == 1:
            # Each star has the same non-uniform spectrum
            self.weights = np.expand_dims(weights, axis=(-1, 0, 1, 3))
        else:
            # Each star has a different non-uniform spectrum
            self.weights = np.expand_dims(weights, axis=(-2, -1, 0))

    def debug_prop(self, wavel, offset=np.zeros(2)):        
        """
        I believe this is diffable but there is no reason to force it to be
        """
        params_dict = {"Wavefront": Wavefront(wavel, offset)}
        intermeds = []
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
            intermeds.append(params_dict)
        return params_dict["Wavefront"].wf2psf(), intermeds
            
        
        
    """################################"""
    ### DIFFERENTIABLE FUNCTIONS BELOW ###
    """################################"""
    
    
    
    def __call__(self):
        """
        Maps the wavelength and position calcualtions across multiple dimesions
        
        To Do: Reformat the vmaps such that we only vmap over wavelengths and
        positions in order to simplify the dimensionality
        """
        
        # Mapping over wavelengths
        propagate_single = vmap(self.propagate_mono, in_axes=(0, None))
        
        # Then over the positions for each image
        # Mapping over the stars in a single image first (zeroth axis of positions)
        propagate_multi = vmap(propagate_single, in_axes=(None, 0))
        
        # Mapping over each image (first axis of positions)
        propagator = vmap(propagate_multi, in_axes=(None, 1))

        # Generate positions 
        dithered_positions = self.dither_positions(self.dithers, self.positions)
        
        # Calc PSFs and apply weights/fluxes
        psfs = propagator(self.wavels, dithered_positions)
        psfs = self.weight_psfs(psfs)
        
        # Sum into images and remove empty dims for single image props
        psf_images = np.squeeze(psfs.sum([1, 2]))
        
        # Vmap operation over each image
        detector_vmap = vmap(self.apply_detector_layers, in_axes=0)
        images = detector_vmap(psf_images)
        
        return images

    def propagate_mono(self, wavel, offset=np.zeros(2)):        
        """
        
        """
        params_dict = {"Wavefront": Wavefront(wavel, offset)}
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
        return params_dict["Wavefront"].wf2psf()
    
    def dither_positions(self, dithers, positions):
        """
        Function to do an outer sum becuase it is more flexible than
        Formatting array shapes and using +
        Turns out I might actually be learning something!
        Output shape: (Nstars, Ndithers, 2)
        
        # dithered_positions = self.positions() + self.dithers.T # Outer sum operation
        # dithered_positions = self.positions + np.expand_dims(self.dithers, axis=(1))
        """
        outer = vmap(vmap(lambda a, b: a + b, in_axes=(0, None)), in_axes=(None, 0))
        dithered_positions = outer(dithers, positions) 
        return dithered_positions
    
    def weight_psfs(self, psfs):
        """
        Normalise Weights, and format weights/fluxes
        Psfs output shape: (Nims, Nstars, Nwavels, npix, npix)
        We want weights shape: (1, 1, Nwavels, 1, 1)
        We want fluxes shape: (1, Nstars, 1, 1, 1)
        """
        # Normliase along each stellar wavelength
        weights_in = self.weights / np.expand_dims(self.weights.sum(2), axis=2)   
        
        # Expand dimension of Fluxes to match PSFs shape
        fluxes_in = np.expand_dims(self.fluxes, axis=(0, -1, -2, -3)) # Expand to correct dims
        
        # Apply weights and fluxes
        psfs *= weights_in
        psfs *= fluxes_in
        
        return psfs
        
    def propagate_single(self, wavels, offset=np.zeros(2), weights=1.):
        """
        Only propagates a single star, allowing wavelength input
        sums output to single array
        
        Wavels must be an array and the same shape as weights if provided
        """
        
        # Mapping over wavelengths
        prop_wf_map = vmap(self.propagate_mono, in_axes=(0, None))
        
        # Apply spectral weighting
        psfs = weights * prop_wf_map(wavels, offset)/len(wavels)
        
        # Sum into single psf
        psf = psfs.sum(0)
        return psf
    
    def apply_detector_layers(self, image):
        """
        
        """
        for i in range(len(self.detector_layers)):
            image = self.detector_layers[i](image)
        return image