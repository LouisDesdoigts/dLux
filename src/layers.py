import jax
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
        WF = WF.add_phase(self.phase_array)
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
    
class ApplyAperture(eqx.Module):
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
    aperture: np.ndarray
    
    def __init__(self, aperture):
        self.aperture = np.array(aperture)
        self.npix = aperture.shape[0]
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        WF = WF.multiply_ampl(self.aperture)
        params_dict["Wavefront"] = WF
        return params_dict
    
    
class ApplyBasisCLIMB(eqx.Module):
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
        
    coefficients: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term
    """
    npix: int = eqx.static_field()
    basis: np.ndarray
    coeffs: np.ndarray
    ideal_wavel: float
    
    def __init__(self, basis, coeffs, ideal_wavel):
        self.npix = int(basis.shape[-1])
        self.basis = np.array(basis).astype(float)
        self.coeffs = np.array(coeffs).astype(float)
        self.ideal_wavel = np.array(ideal_wavel).astype(float)

    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavel = WF.wavel

        # Get basis phase
        latent = self.get_opd(self.basis, self.coeffs)
        binary_phase = np.pi*self.CLIMB(latent)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavel)
        WF = WF.add_opd(opd)

        params_dict["Wavefront"] = WF
        return params_dict
    
    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
    def phase_to_opd(self, phase, wavel):
        return phase*wavel/(2*np.pi)
    
    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)
    
    def get_total_opd(self):
        return self.get_opd(self.basis, self.coeffs)
    
    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coeffs)
        binary_phase = np.pi*self.CLIMB(latent)
        return binary_phase
    
    def lsq_params(self, img):
        xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),np.linspace(0,1,img.shape[1]))
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
        return matrix, xx, yy, A

    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix,img.ravel())

    def area(self, img, epsilon = 1e-15):

        a,b,c = self.lsq(img)
        a, b, c = np.where(a==0,epsilon,a), np.where(b==0,epsilon,b), np.where(c==0,epsilon,c)
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

        dummy = np.array(wf.split(ppsz))
        dummy = np.array(dummy.split(ppsz, axis = 2))
        subarray = dummy[:,:,0,0]

        flat = dummy.reshape(-1, 3, 3)
        vmap_mask = jax.vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin