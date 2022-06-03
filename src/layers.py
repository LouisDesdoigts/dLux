import jax
import jax.numpy as np
import equinox as eqx
from jax.scipy.ndimage import map_coordinates
# from .zernike import zernike_basis

# __all__ = [
    
#     # Optics Layers
#     'CreateWavefront', 'TiltedWavefront', 'CircularAperture', 'Wavefront2PSF', 
#     'NormaliseWavefront', 'AddPhase', 'ApplyOPD', 'ApplyZernike', 'ThinLens', 
    
#     # Instrument Layers
#    'ApplyPixelResponse',
    
#     # Generic Layers
#    'Pad', 'Crop', 'MultiplyArray', 'AddScalar', 'AddArray', 
#    'MultiplyScalar', 'Interpolator', 'InvertXY', 'InvertX', 'InvertY'

# ]


"""
Layer __call__ functions Template:

    def __call__(self, params_dict):
    
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        offset = WF.offset
        pixelscale = WF.pixelscale
        planetype = WF.planetype

        # Do things


        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.offset,     WF, offset)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, planetype)
        params_dict["Wavefront"] = WF
        return params_dict

"""



###################################################
############## Optical Layers #####################
###################################################

class CreateWavefront(eqx.Module):
    """ 
    Initialises an on-axis input wavefront

    Parameters
    ----------
    pixelscale: float, equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
        
    wavefront_size: float
        Units: meters
        Width of the array representing the wavefront in physical units
        
    """
    npix:           int = eqx.static_field()
    wavefront_size: float
    pixelscale:     float
    
    def __init__(self, npix, wavefront_size):
        self.npix = int(npix)
        self.wavefront_size = np.array(wavefront_size).astype(float)
        self.pixelscale = np.array(wavefront_size/npix).astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        # wavefront = np.ones([self.npix, self.npix], dtype=complex)
        wavefront = 1 + 1j*np.ones([self.npix, self.npix])
        pixelscale = self.pixelscale
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront,  
                         is_leaf=lambda x: x is None)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale, 
                         is_leaf=lambda x: x is None)
        params_dict["Wavefront"] = WF
        return params_dict
    
class TiltWavefront(eqx.Module):
    """ 
    Tilts an input wavefront

    Parameters
    ----------
    pixelscale: float, equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
        
    wavefront_size: float
        Units: meters
        Width of the array representing the wavefront in physical units
        
    """
    shift: float
    
    def __init__(self, shift=0.):
        self.shift = np.array(shift).astype(float)
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        xangle, yangle = WF.offset
        
        # Calc and apply tilt phasor
        xcoords, ycoords = WF.get_xycoords()
        tiltphasor = np.exp(-2.0j * np.pi * (xcoords*xangle + ycoords*yangle) / wavel)
        wavefront_out = wavefront * tiltphasor
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class CircularAperture(eqx.Module):
    """
    Multiplies the input wavefront by a pre calculated circular binary 
    (float) mask that fills the size of the array
    
    Note there is a known bug where gradients become nan if phase operations
    are applied after this layer
    """
    npix:  int = eqx.static_field()
    array: np.ndarray
    
    def __init__(self, npix, rmin=0., rmax=1., eps=1e-8):
        self.npix = int(npix)
        self.array = self.create_aperture(self.npix, rmin=rmin, rmax=rmax) + eps
    
    def create_aperture(self, npix, rmin=0., rmax=1.):        
        c = (npix - 1) / 2.
        xs = (np.arange(npix) - c) / c
        XX, YY = np.meshgrid(xs, xs)
        RR = np.sqrt(XX ** 2 + YY ** 2)
        aperture = np.logical_and(RR <= rmax, RR > rmin).astype(float)
        return aperture.astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Multiple by pre-calcauted circular aperture
        wavefront_out = np.multiply(WF.wavefront, self.array)
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class NormaliseWavefront(eqx.Module):
    """ 
    Normalises the input wavefront
    """
    def __init__(self):
        pass
                
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront

        # Normalise input wavefront
        norm_factor = np.sqrt(np.sum(np.abs(wavefront)**2))
        wavefront_out = wavefront/norm_factor

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class ApplyBasisOPD(eqx.Module):
    """
    NEW DOCTRING:
        TO DO
        
    OLD DOCSTRING:
    Adds an array of phase values to the input wavefront calculated from the OPD
     
    Parameters
    ----------
    nterms: int, equinox.static_field
        The number of zernike terms to apply, ignoring the first two radial
        terms: Piston, Tip, Tilt
        
    basis: jax.numpy.ndarray, equinox.static_field
        Arrays holding the pre-calculated basis terms
        
    coeffs: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term
    """
    npix: int = eqx.static_field()
    basis: np.ndarray
    coeffs: np.ndarray
    
    def __init__(self, basis, coeffs=None):
        self.basis = np.array(basis)
        self.npix = self.basis.shape[-1]
        self.coeffs = np.zeros(len(self.basis)) if coeffs is None \
                 else np.array(coeffs).astype(float)

    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel

        # Get basis phase
        opd = self.get_opd(self.basis, self.coeffs)
        phase = self.opd_to_phase(opd, wavel)
        
        # Add phase to wavefront
        phase_out = np.angle(wavefront) + phase
        
        # Recombine into wavefront
        wavefront_out = np.abs(wavefront) * np.exp(1j*phase_out)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
    def get_opd(self, basis, coeffs):
        return np.dot(basis.T, coeffs)
    
    def get_total_opd(self):
        return self.get_opd(self.basis, self.coeffs)
    
class AddPhase(eqx.Module):
    """ 
    
    Takes in an array of phase values and adds them to the phase term of the 
    input wavefront. ie wavelength independent
    
    This would represent a geometric phase optic like the TinyTol Pupil
    
    Parameters
    ----------
    array: jax.numpy.ndarray, equinox.static_field
        Units: radians
        Array of phase values to be applied to the input wavefront
    """
    npix: int = eqx.static_field()
    phase_array: np.ndarray
    
    def __init__(self, phase_array):
        self.phase_array = np.array(phase_array)
        self.npix = phase_array.shape[0]
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront

        # Add phase to input wavefront
        phase = np.angle(wavefront) + self.phase_array
        wavefront_out = np.abs(wavefront) * np.exp(1j*phase)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class ApplyOPD(eqx.Module):
    """ 
    
    Takes in an array representing the Optical Path Difference (OPD) and 
    applies the corresponding phase difference to the input wavefront. 

    This would represent an etched reflective optic, or phase plate
    
    Parameters
    ----------
    array: jax.numpy.ndarray, equinox.static_field
        Units: radians
        Array of OPD values to be applied to the input wavefront
    """
    npix: int = eqx.static_field()
    opd_array: np.ndarray
    
    def __init__(self, opd_array):
        self.opd_array = np.array(opd_array)
        self.npix = opd_array.shape[0]
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel

        # Apply opd phase to input wavefront
        phase_array = self.opd_to_phase(self.opd_array, wavel)
        phase = np.angle(wavefront) + phase_array
        wavefront_out = np.abs(wavefront) * np.exp(1j*phase)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
class InterpReIm(eqx.Module):
    """
    Note this has strange behvaiour with hessian calcuations (gives nans)
    """
    npix_out: int = eqx.static_field()
    pixelscale_out: float
    
    def __init__(self, npix_out, pixelscale_out):
        self.npix_out = int(npix_out)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)

    def __call__(self, params_dict):
        """
        NOTE: Poppy pads all arrays by 2 pixels before interpolating to reduce 
        edge effects - We will not do that here, chosing instead to have
        all layers as minimal as possible, and have guidelines around best 
        practice to get the best results
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        pixelscale = WF.pixelscale
        
        # Get coords arrays
        npix_in = wavefront.shape[0]
        ratio = self.pixelscale_out/pixelscale
        shift = (npix_in - ratio*self.npix_out)/2
        xs = ratio*(np.arange(self.npix_out)) + shift
        YY, XX = np.meshgrid(xs, xs)
        coords = np.array([XX, YY])
        
        # Interp real and imag
        re = map_coordinates(wavefront.real, coords, order=1, mode='nearest')
        im = map_coordinates(wavefront.imag, coords, order=1, mode='nearest')
        wavefront_out = re + 1j*im
        
        # Enforce conservation of energy:
        pixscale_ratio = pixelscale / self.pixelscale_out
        wavefront_out *= 1. / pixscale_ratio
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class InterpAmPh(eqx.Module):
    """
    Note this has strange behvaiour with hessian calcuations (gives nans)
    """
    npix_out: int = eqx.static_field()
    pixelscale_out: float
    
    def __init__(self, npix_out, pixelscale_out):
        self.npix_out = int(npix_out)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)

    def __call__(self, params_dict):
        """
        NOTE: Poppy pads all arrays by 2 pixels before interpolating to reduce 
        edge effects - We will not do that here, chosing instead to have
        all layers as minimal as possible, and have guidelines around best 
        practice to get the best results
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        pixelscale = WF.pixelscale
        
        # Get coords arrays
        npix_in = wavefront.shape[0]
        ratio = self.pixelscale_out/pixelscale
        shift = (npix_in - ratio*self.npix_out)/2
        xs = ratio*(np.arange(self.npix_out)) + shift
        YY, XX = np.meshgrid(xs, xs)
        coords = np.array([XX, YY])

        # Interp mag and phase
        mag   = map_coordinates(np.abs(wavefront),   coords, order=1)
        phase = map_coordinates(np.angle(wavefront), coords, order=1)
        wavefront_out = mag * np.exp(1j*phase)

        # Enforce conservation of energy:
        pixscale_ratio = pixelscale / self.pixelscale_out
        wavefront_out *= 1. / pixscale_ratio
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        params_dict["Wavefront"] = WF
        return params_dict




######################################################
############## Instrumental Layers ###################
######################################################
    
    
class ApplyPixelResponse(eqx.Module):
    """
    
    """
    pixel_response: np.ndarray
    
    def __init__(self, pixel_response):
        """
        
        """
        self.pixel_response = np.array(pixel_response)
        
    def __call__(self, image):
        """
        
        """
        image *= self.pixel_response
        return image
    
class ApplyJitter(eqx.Module):
    """
    Convolves the output image with a gaussian kernal
    """
    kernel_size: int
    sigma: float
    
    def __init__(self, sigma, kernel_size=25):
        self.kernel_size = int(kernel_size)
        self.sigma = np.array(sigma).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Generate distribution
        x = np.linspace(-10, 10, self.kernel_size)
        window = jax.scipy.stats.norm.pdf(x,          scale=self.sigma) * \
                 jax.scipy.stats.norm.pdf(x[:, None], scale=self.sigma)
        
        # Normalise
        window /= np.sum(window)
        
        # Convolve with image
        image_out = jax.scipy.signal.convolve(image, window, mode='same')
        return image_out
    
class ApplySaturation(eqx.Module):
    """
    Reduces any values above self.saturation to self.saturation
    """
    saturation: float
    
    def __init__(self, saturation):
        self.saturation = np.array(saturation).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Apply saturation
        image_out = np.minimum(image, self.saturation)
        return image_out
    
    
    
    
    

######################################################
############## Generic Array Ops #####################
######################################################

class Pad(eqx.Module):
    """
    
    """
    npix_in:  int = eqx.static_field()
    npix_out: int = eqx.static_field()
    
    def __init__(self, npix_in, npix_out):
        self.npix_in =  int(npix_in)
        self.npix_out = int(npix_out)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront

        # Pad wavefront
        padded = np.zeros([self.npix_out, self.npix_out], dtype=wavefront.dtype)
        cen = self.npix_out//2
        s = self.npix_in//2
        wavefront_out = padded.at[cen-s:cen+s, cen-s:cen+s].set(wavefront)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict  
    
class Crop(eqx.Module):
    """
    
    """
    npix_in:  int = eqx.static_field()
    npix_out: int = eqx.static_field()
    
    def __init__(self, npix_in, npix_out):
        self.npix_in =  int(npix_in)
        self.npix_out = int(npix_out)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront

        # Crop wavefront
        cen = self.npix_in//2
        s = self.npix_out//2
        wavefront_out = wavefront[cen-s:cen+s, cen-s:cen+s]

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict  

class MultiplyScalar(eqx.Module):
    """
    Multiplies the input wavefront by a scalar
    """
    value: float
    
    def __init__(self, value):
        self.value = np.array(value).astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Multiply by value
        wavefront_out = WF.wavefront * self.value
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict

class MultiplyArray(eqx.Module):
    """
    Multiplies the input wavefront by an array
    """
    npix: int = eqx.static_field()
    array: np.ndarray
    
    def __init__(self, array):
        self.array = np.array(array)
        self.npix = array.shape[0]
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Multiply by array
        wavefront_out = WF.wavefront * self.array
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict

class AddScalar(eqx.Module):
    """
    Add a scalar tp the input wavefront
    """
    value: float
    
    def __init__(self, value):
        self.value = np.array(value).astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Multiply by pvalue
        wavefront_out = WF.wavefront + self.value
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict

class AddArray(eqx.Module):
    """
    Adds an array of values to the input wavefront
    """
    npix: int = eqx.static_field()
    array: np.ndarray
    
    def __init__(self, array):
        self.array = np.array(array)
        self.npix = array.shape[0]
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Multiply by pvalue
        wavefront_out = WF.wavefront + self.array
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class InvertXY(eqx.Module):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    def __init__(self):
        pass
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        
        # Invert Axis
        wavefront_out = wavefront[::-1, ::-1]
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class InvertX(eqx.Module):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    def __init__(self):
        pass
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        
        # Invert Axis
        wavefront_out = wavefront[:, ::-1]
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class InvertY(eqx.Module):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    def __init__(self):
        pass
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        
        # Invert Axis
        wavefront_out = wavefront[::-1]
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    

    
# class PadToWavel(eqx.Module):
#     """ 
#     To Do
#     Implement this 
#     Possibly force fixed wavelength and pre-calc array sizes
#     Possibly pre-calculate array sizes and store than in osys object?
#     """
#     pass