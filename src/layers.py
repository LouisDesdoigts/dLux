import jax.numpy as np
import equinox as eqx

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
        ampl = np.ones([self.npix, self.npix])
        phase = np.zeros([self.npix, self.npix])
        pixelscale = self.pixelscale
        
        # # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.amplitude,  WF, ampl,  
                         is_leaf=lambda x: x is None)
        
        WF = eqx.tree_at(lambda WF: WF.phase,  WF, phase,  
                         is_leaf=lambda x: x is None)
        
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale, 
                         is_leaf=lambda x: x is None)
        params_dict["Wavefront"] = WF
        return params_dict
    
class TiltWavefront(eqx.Module):
    """ 
    Applies a paraxial tilt by adding a phase slope

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
    
    def __init__(self):
        pass
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Calc phase tilt 
        xangle, yangle = WF.offset
        xcoords, ycoords = WF.get_xycoords()
        k = 2*np.pi/WF.wavel
        phase = -k * (xcoords*xangle + ycoords*yangle)
        WF = WF.add_phase(phase)
        
        # # Update Wavefront Object
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
        WF = WF.multiply_ampl(self.array)

        # Update Wavefront Object
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
        WF = WF.normalise()
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
        WF = WF.add_opd(self.get_total_opd())
        params_dict["Wavefront"] = WF
        return params_dict
    
    def get_total_opd(self):
        return np.dot(self.basis.T, self.coeffs)
    
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
        WF = WF.add_phase(phase_array)
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
        WF = WF.add_opd(self.opd_array)
        params_dict["Wavefront"] = WF
        return params_dict