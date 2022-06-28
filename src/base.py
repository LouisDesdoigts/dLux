import jax
import jax.numpy as numpy
from jax import vmap
import equinox
from copy import deepcopy


class OpticalSystem(equinox.Module):
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
    use numpy.squeeze to remove empy axes
    
    """
    # Helpers, Determined from inumpyuts, not real params
    Nstars:  int
    Nwavels: int
    Nims:    int
    
    wavels:          numpy.ndarray
    positions:       numpy.ndarray
    fluxes:          numpy.ndarray
    weights:         numpy.ndarray
    dithers:         numpy.ndarray
    layers:          list
    detector_layers: list
    
    # To Do - add asset conditions to ensure that everything is formatted correctly 
    # To Do - pass in positions for multiple images, ignoring dither (ie multi image)
    def __init__(self, layers, wavels=None, positions=None, fluxes=None, 
                       weights=None, dithers=None, detector_layers=None):
        
        # Required Inumpyuts
        self.layers = layers
        self.wavels = numpy.array(wavels).astype(float)
        
        # Set to default values
        self.positions = numpy.zeros([1, 2]) if positions is None else np.array(positions)
        self.fluxes = numpy.ones(len(self.positions)) if fluxes is None else np.array(fluxes)
        self.weights = numpy.ones(len(self.wavels)) if weights is None else np.array(weights)
        self.dithers = numpy.zeros([1, 2]) if dithers is None else dithers
        self.detector_layers = [] if detector_layers is None else detector_layers
        
        if self.fluxes.shape == ():
            self.fluxes = numpy.array([self.fluxes])
        
        # Determined from inumpyuts - treated as static
        self.Nstars =  len(self.positions)
        self.Nims =    len(self.dithers)
        self.Nwavels = 0 if wavels is None else len(self.wavels)
        
        # Check Inumpyut shapes
        assert self.positions.shape[-1] == 2, """Inumpyut positions must be 
        of shape (Nstars, 2)"""
        
        assert self.fluxes.shape[0] == self.Nstars, """Inumpyut fluxes must be
        match inumpyut positions."""
        
        weight_shape = self.weights.shape
        if len(weight_shape) == 1 and weights is not None:
            assert weight_shape[0] == self.Nwavels, """Inumpyuts weights shape 
            must be either (len(wavels)) or  (len(positions), len(wavels)), 
            got shape: {}""".format(self.weights.shape)
        elif len(weight_shape) == 2:
            assert weight_shape == [self.Nstars, self.Nwavels], """Inumpyuts 
            weights shape must be either (len(wavels)) or  (len(positions), 
            len(wavels))"""

    def debug_prop(self, wavel, offset=numpy.zeros(2)):        
        """
        I believe this is diffable but there is no reason to force it to be
        """
        # params_dict = {"Wavefront": Wavefront(wavel, offset)}
        params_dict = {"Wavefront": PhysicalWavefront(wavel, offset)}
        intermeds = []
        layers_applied = []
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
            intermeds.append(deepcopy(params_dict))
            layers_applied.append(self.layers[i].__str__())
        return params_dict["Wavefront"].wf2psf(), intermeds, layers_applied
            
        
        
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
        
        # Then over the positions 
        propagator = vmap(propagate_single, in_axes=(None, 0))

        # Generate inumpyut positions vector
        dithered_positions = self.dither_positions()
        
        # Calculate PSFs
        psfs = propagator(self.wavels, dithered_positions)
        
        # Reshape output into images
        psfs = self.reshape_psfs(psfs)
        
        # Weight PSFs and sum into images
        psfs = self.weight_psfs(psfs).sum([1, 2])
        
        # Vmap operation over each image
        detector_vmap = vmap(self.apply_detector_layers, in_axes=0)
        images = detector_vmap(psfs)
        
        return numpy.squeeze(images)

    def propagate_mono(self, wavel, offset=numpy.zeros(2)):        
        """
        
        """
        # params_dict = {"Wavefront": Wavefront(wavel, offset)}
        params_dict = {"Wavefront": PhysicalWavefront(wavel, offset)}
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
        return params_dict["Wavefront"].wf2psf()
    
    def propagate_single(self, wavels, offset=numpy.zeros(2), weights=1.):
        """
        Only propagates a single star, allowing wavelength inumpyut
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
    
    def reshape_psfs(self, psfs):
        """
        
        """
        numpyix = psfs.shape[-1]
        return psfs.reshape([self.Nims, self.Nstars, self.Nwavels, numpyix, npix])
    
    def dither_positions(self):
        """
        Dithers the inumpyut positions, returned with shape (Npsfs, 2)
        """
        Npsfs = self.Nstars * self.Nims
        shaped_pos = self.positions.reshape([1, self.Nstars, 2])
        shaped_dith = self.dithers.reshape([self.Nims, 1, 2])
        dithered_positions = (shaped_pos + shaped_dith).reshape([Npsfs, 2])
        return dithered_positions
    
    
    def weight_psfs(self, psfs):
        """
        Normalise Weights, and format weights/fluxes
        Psfs output shape: (Nims, Nstars, Nwavels, numpyix, npix)
        We want weights shape: (1, 1, Nwavels, 1, 1)
        We want fluxes shape: (1, Nstars, 1, 1, 1)
        """
        # Get values
        Nims = self.Nims
        Nstars = self.Nstars
        Nwavels = self.Nwavels
        
        # Format and normalise weights
        if len(self.weights.shape) == 3:
            weights_in = self.weights.reshape([Nims, Nstars, Nwavels, 1, 1])
            weights_in /= numpy.expand_dims(weights_in.sum(2), axis=2) 
        elif len(self.weights.shape) == 2:
            weights_in = self.weights.reshape([1, Nstars, Nwavels, 1, 1])
            weights_in /= numpy.expand_dims(weights_in.sum(2), axis=2) 
        elif self.weights.shape[0] == self.Nwavels:
            weights_in = self.weights.reshape([1, 1, Nwavels, 1, 1])
            weights_in /= numpy.expand_dims(weights_in.sum(2), axis=2) 
        else:
            weights_in = self.weights
        
        # Format Fluxes
        if len(self.fluxes) == 1:
            fluxes = self.fluxes
        else:
            fluxes = self.fluxes.reshape([1, Nstars, 1, 1, 1])
        
        
        # Apply weights and fluxus
        psfs *= weights_in
        psfs *= fluxes
        return psfs
        
        
