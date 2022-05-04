import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from jax import vmap
from equinox import Module
from equinox import static_field

"""
This script stores all classes which are defined by equinox.Module()
"""


########################
### Layer Base Class ###
########################

class Layer(Module):
    """
    Base Layer class
    Can optionally pass in 'function' in order to facilitate parameterised planes
    Can optionally pass in 'static' to automatically freeze in the parameters
    """
    size_in: int = static_field()
    size_out: int = static_field()
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out


###################################
### Optical System Base Classes ###
###################################
    
class OpticalSystem(Module):
    """
    Base class defining some optical system
    
    Dev: Automatically stores intermediate wavefronts for examination
    
    layers must be a list of Layer objects
    Each layer object can either be some optical operation, 
    transform, or NN type layer
    """
    layers: list # Can this be set to a static field?    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, wavel, offset):
        """
        offset is now passed through all layers so that the shift can be 
        done in layers other than the intitial (which people might want?)
        """
        # Inialise value and objects to store data
        x, pixelscale = None, None
        
        # Inialise values and iterate 
        for i in range(len(self.layers)):
            x, pixelscale = self.layers[i](x, wavel, offset, pixelscale)
        return x
    
class DevOpticalSystem(Module):
    """
    Base class defining some optical system
    
    Dev: Automatically stores intermediate wavefronts for examination
    
    layers must be a list of Layer objects
    Each layer object can either be some optical operation, 
    transform, or NN type layer
    """
    layers: list # Can this be set to a static field?    
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, wavel, offset):
        
        # Inialise value and objects to store data
        functions, intermed_wavefronts, intermed_pixelscales = [], [], []
        x, pixelscale = None, None
        
        # Inialise values and iterate 
        for i in range(len(self.layers)):
            x, pixelscale = self.layers[i](x, wavel, offset, pixelscale)
            # Store Values in list
            intermed_wavefronts.append(x)
            intermed_pixelscales.append(pixelscale)
            
        return x, intermed_wavefronts, intermed_pixelscales


########################
### Scene Base Class ###
########################

class Scene(Module):
    """
    A Class to store and apply properties external to the optical system
    Ie: stellar positions and specturms
    
    positions: (Nstars, 2) array
    wavels: (Nwavels) array
    weights: (Nwavel)/(Nwavels, Nstars) array
    
    Take in layers in order to re-intialise the model every call?
    
    If the scaling array is created at initialisation, then do we lose the
    the ability to keep them normalised?
    
    General image output shape: (Nimages, Nstars, Nwavels, Npix, Npix)
    
    Currently doesnt allow temporal variation in specturm becuase im lazy
    Currently doesnt allow temporal variation in flux becuase im lazy
    
    ToDo: Add getter methods for acessing weights and fluxes attributes that
    use np.squeeze to remove empy axes
    
    QUESTION: Should dithered positions be a static field to allow grads to 
    prop through to the undelying position and shifts
    """
    optical_system: Module # Not static becuase we want to take grads through this?
    detector_layers: list
    wavels: ndarray = static_field()
    positions: ndarray
    fluxes: ndarray
    weights: ndarray
    dithers: ndarray
    
    # Determined from inputs
    Nstars: int = static_field()
    Nwavels: int = static_field()
    Nims: int = static_field()
    
    # dithered_positions: jax.numpy.ndarray = equinox.static_field() # Trying out Q
    
    def __init__(self, optical_system, wavels, positions, fluxes, 
                       weights=None, detector_layers=[], dithers=np.zeros([1, 2])):
                 
        self.optical_system = optical_system
        self.detector_layers = detector_layers
        self.wavels = wavels
        self.positions = positions
        self.fluxes = fluxes
        self.dithers = dithers
        
        # Determined from inputs
        self.Nstars = len(positions)
        self.Nwavels = len(wavels)
        self.Nims = len(dithers)
        
        # Function to do an outer sum becuase Jax hasnt ported the functionality
        # Turns out I might actually be learning something!
        # outer = vmap(vmap(lambda a, b: a + b, in_axes=(0, None)), in_axes=(None, 0))
        # self.dithered_positions = outer(dithers, positions) 
        # Output shape: (Nstars, Ndithers, 2)
        
        # Format weights
        if weights is None:
            # Each star has the same uniform spectrum
            self.weights = np.ones([1, 1, self.Nwavels, 1, 1])
        elif len(weights.shape) == 1:
            # Each star has the same non-uniform spectrum
            self.weights = np.expand_dims(weights, axis=(-1, 0, 1, 3))
        else:
            # Each star has a different non-uniform spectrum
            self.weights = np.expand_dims(weights, axis=(-2, -1, 0))
            
    def _update_positions(self):
        self.positions += 1e2
            
    def _dither_positions(self):
        outer = vmap(vmap(lambda a, b: a + b, in_axes=(0, None)), in_axes=(None, 0))
        dithered_positions = outer(self.dithers, self.positions) 
        return dithered_positions
        
        
    def _apply_detector_layers(self, image):
        for i in range(len(self.detector_layers)):
            image = self.detector_layers[i](image)
        return image
            
        
    def __call__(self):
        """
        shift parameter is optional so that Scene can be called independently
        of MultiImageScene
        """
        # Maps the wavelength and position calcualtions across multiple dimesions
        # We want to vmap in three dims
        # Optical system inputs: (wavel, position)
        
        # First over wavels
        wavel_vmap = vmap(self.optical_system, in_axes=(0, None))
        # wavel_vmap = pmap(self.optical_system, in_axes=(0, None))
        
        # Then over the positions for each image
        # Mapping over the stars in a single image first (zeroth axis)
        position_vmap = vmap(wavel_vmap, in_axes=(None, 0))
        # position_vmap = pmap(wavel_vmap, in_axes=(None, 0))
        
        # Then map over each image
        # Mapping over each image (first axis)
        image_vmap = vmap(position_vmap, in_axes=(None, 1))
        # image_vmap = pmap(position_vmap, in_axes=(None, 1))

        # Generate PSFs
        dithered_positions = self._dither_positions()
        psfs = image_vmap(self.wavels, dithered_positions)
        

        # Normalise Weights, and format weights/fluxes
        weights_norm = np.expand_dims(self.weights.sum(2), axis=2) # Normliase along wavels
        weights_in = self.weights / weights_norm  # Expand dimension back out
        fluxes_in = np.expand_dims(self.fluxes, axis=(0, -1, -2, -3)) # Expand to correct dims
        
        # Apply weights and fluxes
        psfs *= weights_in
        psfs *= fluxes_in
        
        # Sum into images
        psf_images = psfs.sum([1, 2])
        
        # Vmap operation over each image
        detector_vmap = vmap(self._apply_detector_layers, in_axes=0)
        # detector_vmap = pmap(self._apply_detector_layers, in_axes=0)
        images = detector_vmap(psf_images)
        
        return images