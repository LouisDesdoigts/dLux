import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from jax import vmap
from equinox import Module
from equinox import static_field

__all__ = ['Layer', 'OpticalSystem', 'Scene']

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
    
class OpticalSystem(Module):
    """ OpticalSystem class, Equinox Modle
    
    A class to store the list of layer objects that specifically define an 
    optical system. 
    
    Parameters
    ----------
    layers: list, 
        A list of Layer objects 
    
    Notes:
     - layers could theoretically also be a dictionary if needed for optax
     - test if everything still works if this is set as static
    """
    layers: list
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, wavelength, offset):
        """ Propagation Function
        
        This function propagates the wavefront by iterating though the layers
        list and calling the __call__() function 
        
        Parameters
        ----------
        wavelength: float
            Units: meters
            Wavelength of light being propagated through the opitcal
            system
        offset: jax.numpy.ndarray
            Units: radians
            shape: (2,)
            The (x, y) angular offset of the source object from the optical 
            axis in radians
        
        Returns
        -------
        wavefront: jax.numpy.ndarray
            The output wavefront after being 'propaged' through all the layer
            objects
        """
        
        # Inialise values
        wavefront, pixelscale = None, None
        
        # Inialise values and iterate 
        for i in range(len(self.layers)):
            wavefront, pixelscale = self.layers[i](wavefront, wavelength, offset, pixelscale)
        return wavefront
    
    def debug_prop(self, wavel, offset):
        """ Debugging/Heler Propagation Function
        
        Helper function that is identical to the __call__() function except it
        stores the intermediary wavefront and pixelscale values and returns 
        them. 
        
        Parameters
        ----------
        wavel: float
            Units: meters
            Wavelength of light being propagated through the opitcal
            system
        offset: jax.numpy.ndarray
            Units: radians
            shape: (2,)
            The (x, y) angular offset of the source object from the optical 
            axis in radians
            
        Intermediate
        ------------
        pixelscale: float
            Units: meters/pixel
            The pixelscae of each array between each layer operation
        
        Returns
        -------
        wavefront: jax.numpy.ndarray
            The output wavefront after being 'propaged' through all the layer
            objects
        intermed_wavefronts: list
            a list storing the state of the wavefront between each layer 
            operation.
            This can be used to check that your wavefront is being transformed
            properly between each layer and that Layers are working as expected
        intermed_pixelscales: list
            a list storing the pixelscale of the wavefront between each layer 
            operation. 
            This can be used to check for inconsistencies and look for layers
            where interpolation is needed
        """
        
        # Inialise value and objects to store data
        functions, intermed_wavefronts, intermed_pixelscales = [], [], []
        wavefront, pixelscale = None, None
        
        # Inialise values and iterate 
        for i in range(len(self.layers)):
            wavefront, pixelscale = self.layers[i](wavefront, wavel, offset, pixelscale)
            
            # Store Values in list
            intermed_wavefronts.append(wavefront)
            intermed_pixelscales.append(pixelscale)
            
        return wavefront, intermed_wavefronts, intermed_pixelscales

    

class Scene(Module):
    """ Scene class, Equinox Modle
    
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
            
    def _dither_positions(self):
        # Function to do an outer sum becuase Jax hasnt ported the functionality
        # Turns out I might actually be learning something!
        # outer = vmap(vmap(lambda a, b: a + b, in_axes=(0, None)), in_axes=(None, 0))
        # self.dithered_positions = outer(dithers, positions) 
        # Output shape: (Nstars, Ndithers, 2)
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
        
        # Then over the positions for each image
        # Mapping over the stars in a single image first (zeroth axis)
        position_vmap = vmap(wavel_vmap, in_axes=(None, 0))
        
        # Then map over each image
        # Mapping over each image (first axis)
        image_vmap = vmap(position_vmap, in_axes=(None, 1))

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
        images = detector_vmap(psf_images)
        
        return images