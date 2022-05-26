import jax.numpy as np
import equinox as eqx
from .jaxinterp2d import interp2d
from .zernike import zernike_basis

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
        wavefront = np.ones([self.npix, self.npix], dtype=complex)
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
        xcoords, ycoords = WF.get_xycoords(shift=self.shift)
        tiltphasor = np.exp(-2.0j * np.pi * (xcoords*xangle + ycoords*yangle) / wavel)
        wavefront_out = wavefront * tiltphasor
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
class CircularAperture(eqx.Module):
    """
    Multiplies the input wavefront by a pre calculated circular binary mask
    that fills the size of the array
    __call__() is a mirror of MultiplyArray(Layer)
    """
    npix:  int = eqx.static_field()
    array: np.ndarray
    
    def __init__(self, npix):
        self.npix = int(npix)
        self.array = self.create_aperture(self.npix)
        
    def create_aperture(self, npix):
        xs = np.arange(-npix//2, npix//2)
        XX, YY = np.meshgrid(xs, xs)
        RR = np.hypot(XX, YY)
        aperture = RR < npix//2
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
    
class ApplyZernike(eqx.Module):
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
        Arrays holding the pre-calculated zernike basis terms
        
    coefficients: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term
    """
    npix: int = eqx.static_field()
    names: list = eqx.static_field()
    basis: np.ndarray = eqx.static_field()
    coefficients: np.ndarray
    
    def __init__(self, npix, coefficients, indexes=None):
        self.npix = int(npix)
        self.coefficients = np.array(coefficients)
        
        # Load basis
        indexes = np.arange(len(coefficients)) if indexes is None else indexes
        
        # Check indexes
        if np.max(indexes) >= 22:
            raise ValueError("Zernike indexes above 22 not currently supported")
            
        # Get full basis
        full_basis = np.array(np.nan_to_num(
                zernike_basis(nterms=np.max(indexes)+1, npix=int(npix))))
        
        # Get basis
        self.basis = np.array([full_basis[indx] for indx in indexes])
        
        # Names - Helper
        all_names = ['Piston', 'Tilt X', 'Tilt Y',
                     'Focus', 'Astigmatism 45', 'Astigmatism 0',
                     'Coma Y', 'Coma X',
                     'Trefoil Y', 'Trefoil X',
                     'Spherical', '2nd Astig 0', '2nd Astig 45',
                     'Tetrafoil 0', 'Tetrafoil 22.5',
                     '2nd coma X', '2nd coma Y', '3rd Astig X', '3rd Astig Y',
                     'Pentafoil X', 'Pentafoil Y', '5th order spherical']
        
        # Load in names
        self.names = [all_names[indx] for indx in indexes]

    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel

        # Get zernike phase
        zernike_opd = self.get_opd(self.basis, self.coefficients)
        zernike_phase = self.opd_to_phase(zernike_opd, wavel)
        
        # Add phase to wavefront
        phase_out = np.angle(wavefront) + zernike_phase
        
        # Recombine into wavefront
        wavefront_out = np.abs(wavefront) * np.exp(1j*phase_out)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
    def opd_to_phase(self, opd, wavel):
        """
        
        """
        return 2*np.pi*opd/wavel
    
    def get_opd(self, basis, coefficients):
        """
        
        """
        return np.dot(basis.T, coefficients)
    
class ThinLens(eqx.Module):
    """
    Applies the thin-lens formula
    To Do:
    Check if the center of r_coords is on the corner or center of a pixel
    
    Parameters
    ----------
    pixelscale: float equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
    r_coords: jax.numpy.ndarray equinox.static_field
        Pre-calcualted array defining the physical radial distance from the
        array center
    f: float equinox.static_field
        Units: meters
        Focal length
    """
    f: float
    
    def __init__(self, f):
        self.f = np.array(f).astype(float)
        
    def __call__(self, params_dict):
        """
        x/y_coords: spatial coordinate system (m)
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel

        # Apply Thin Lens Equation
        k = 2*np.pi / wavel # Wavenumber
        x_coords, y_coords = WF.get_xycoords()
        r_coords = np.hypot(x_coords, y_coords)
        wavefront_out = wavefront * np.exp(-0.5j * k * r_coords**2 * 1/self.f)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        params_dict["Wavefront"] = WF
        return params_dict
    
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
    
class Interpolator(eqx.Module):
    """
    
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
        wavel = WF.wavel
        pixelscale = WF.pixelscale
        
        # Resample
        wavefront_out = self.interpolate(
                                wavefront, 
                                pixelscale,
                                self.npix_out,
                                self.pixelscale_out)
        
        # Enforce conservation of energy:
        pixscale_ratio = pixelscale / self.pixelscale_out
        wavefront_out *= 1. / pixscale_ratio
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        params_dict["Wavefront"] = WF
        return params_dict

    def interpolate(self, wf_in, pixscale_in, npix_out, pixscale_out):
        
        npix_in = wf_in.shape[0]
        x_in =  self._make_axis(npix_in,  pixscale_in)
        y_in =  self._make_axis(npix_in,  pixscale_in)
        x_out = self._make_axis(npix_out, pixscale_out)
        y_out = self._make_axis(npix_out, pixscale_out)

        # New Method
        shape_out = (self.npix_out, self.npix_out)
        XX_out, YY_out = np.meshgrid(x_out, y_out)
        
        # # Real and imag
        # real = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, wavefront.real).reshape(shape_out)
        # imag = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, wavefront.imag).reshape(shape_out)
        # new_wf = real + 1j * imag
        
        # Mag and Phase
        mag =   interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in,   np.abs(wf_in)).reshape(shape_out)
        phase = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, np.angle(wf_in)).reshape(shape_out)
        wf_out = mag * np.exp(1j*phase)
        
        return wf_out

    def _make_axis(self, npix, step):
        """ Helper function to make coordinate axis for interpolation """
        return step * np.arange(-npix // 2, npix // 2, dtype=float)
    
    
    
    
class PadToWavel(eqx.Module):
    """ 
    To Do
    Implement this as an aleternative to interpolate
     -> How to do this with static array sizes since size out depends on wavel?
     -> Probably not possible
    
    Possibly pre-calculate array sizes and store than in osys object?
    """
    pass




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
        self.npix = array.shape
        
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
    
