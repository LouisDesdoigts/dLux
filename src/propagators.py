import jax.numpy as np
import equinox as eqx

class MFT(eqx.Module):
    """
    Matches poppy but assumes square
    """
    npix_out: int = eqx.static_field()
    focal_length:   float
    pixelscale_out: float
    
    def __init__(self, npix_out, focal_length, pixelscale_out):
        self.npix_out = int(npix_out)
        self.focal_length =   np.array(focal_length).astype(float)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        pixelscale = WF.pixelscale


        # Calculate NlamD parameter (Do on Wavefront class??)
        npup = wavefront.shape[0] 
        npix = self.npix_out
        
        wf_size_in = pixelscale * npup # Wavefront size 'd'        
        det_size = self.pixelscale_out * npix # detector size
        
        wavel_scale = det_size * wf_size_in / self.focal_length
        nlamD = wavel_scale / wavel

        # Calulate Arrays
        dX = 1.0 / float(npup)
        dU = nlamD / npix
        
        Xs = (np.arange(npup, dtype=float) - float(npup) / 2.0) * dX
        Us = (np.arange(npix, dtype=float) - float(npix) / 2.0) * dU
        XU = np.outer(Xs, Us)
        expXU = np.exp(-2.0 * np.pi * 1j * XU)

        # Note: Can casue overflow issues on 32-bit
        norm_coeff = np.sqrt((nlamD**2) / (npup**2 * npix**2)) 

        # Perform MFT
        t1 = np.dot(expXU.T, wavefront)
        t2 = np.dot(t1, expXU)
        wavefront_out = norm_coeff * t2
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict
    
class OffsetMFT(eqx.Module):
    """
    Matches poppy but assumes square
    """
    npix_out: int = eqx.static_field()
    focal_length:   float
    pixelscale_out: float
    
    def __init__(self, npix_out, focal_length, pixelscale_out):
        self.npix_out =       np.array(npix_out).astype(int)
        self.focal_length =   np.array(focal_length).astype(float)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)
        
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        offset = WF.offset
        pixelscale = WF.pixelscale
        
        
        # Calculate NlamD (radius from optical axis measured in fringes)
        npup = wavefront.shape[0] 
        npix = self.npix_out
        wf_size_in = pixelscale * npup # Wavefront size, 'd'
        det_size = self.pixelscale_out * npix # detector size
        wavel_scale =  wf_size_in * npix * self.pixelscale_out / self.focal_length
        # wavel_scale =  wf_size_in * npix * pixel_rad
        nlamD = wavel_scale / wavel
        
        # Calulate Arrays
        dX = 1.0 / float(npup)
        dU = nlamD / float(npix)
        
        offsetX, offsetY = offset * self.focal_length / self.pixelscale_out
        dY = 1.0 / float(npup)
        dV = nlamD / float(npix)
        
        Xs = (np.arange(npup, dtype=float) - float(npup) / 2.0 + offsetX) * dX
        Ys = (np.arange(npup, dtype=float) - float(npup) / 2.0 + offsetY) * dY
        Us = (np.arange(npix, dtype=float) - float(npix) / 2.0 + offsetX) * dU
        Vs = (np.arange(npix, dtype=float) - float(npix) / 2.0 + offsetY) * dV
        
        XU = np.outer(Xs, Us)
        YV = np.outer(Ys, Vs)

        expYV = np.exp(-2.0 * np.pi * 1j * YV).T
        expXU = np.exp(-2.0 * np.pi * 1j * XU)
        t1 = np.dot(expYV, wavefront)
        t2 = np.dot(t1, expXU)

        norm_coeff = np.sqrt((nlamD**2) / (npup**2 * npix**2))         
        wavefront_out = norm_coeff * t2
        
        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict
    
class FFT(eqx.Module):
    """
    
    """
    focal_length: float
    
    def __init__(self, focal_length):
        self.focal_length = np.array(focal_length).astype(float)
        
    def __call__(self, params_dict):
        """
        Performs normalisation matching poppy
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        pixelscale = WF.pixelscale

        # Calculate Wavefront & Pixelscale
        npix_in = wavefront.shape[0]
        norm = npix_in
        wavefront_out = norm * np.fft.fftshift(np.fft.ifft2(wavefront))
        pixelscale_out = wavel * self.focal_length / (pixelscale * npix_in)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict
    
class IFFT(eqx.Module):
    """
    
    """
    focal_length: float
    
    def __init__(self, focal_length):
        self.focal_length = np.array(focal_length).astype(float)
        
    def __call__(self, params_dict):
        """
        Performs normalisation matching poppy
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        pixelscale = WF.pixelscale

        # Calculate Wavefront & Pixelscale
        npix_in = wavefront.shape[0]
        norm = 1./npix_in
        wavefront_out = norm * np.fft.fft2(np.fft.ifftshift(wavefront))
        pixelscale_out = wavel * self.focal_length / (pixelscale * npix_in)

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Pupil")
        params_dict["Wavefront"] = WF
        return params_dict
    
class FresnelProp(eqx.Module):
    """
    Layer for Fresnel propagation
    
    Note this algorithm is completely not intensity conservative and will
    give different answers for each wavelength too
    
    Note this probably gives wrong answers for tilted input wavefronts becuase
    I believe its based on the paraxial wave equation and tilted wavefronts
    are not paraxial
    
    -> Do something clever with MFTs in the final TF with the offset term to 
    get propper off axis behaviour?
    """
    prop_dist:    float
    
    def __init__(self, focal_length, prop_dist):
        """
        Initialisation
        pixelscale must be in m/pixel, ie aperture/npix
        
        Aperture pixelscale is the pixelscale at the ORIGINAL lens -> for multiple
        non-conjuagte lens systems is it slightly more complex (EFL, EFZ?)
        
        Do we need 'effective focal length' and 'effective final distance' in order to 
        track pixelscales correctly? Ie in an RC telescope after M2 the EFL is changed
        and the effective propagated distance is different too
          -> What about free space propagation?
        """
        self.prop_dist = np.array(prop_dist).astype(float)

    def __call__(self, params_dict):
        """
        Propagates Fresnel
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.wavefront
        wavel = WF.wavel
        pixelscale = WF.pixelscale

        # Calc Pixelscale
        npix_in = wavefront.shape[0]
        z_prop = self.prop_dist
        pixelscale_out = wavel * z_prop / (npix_in * pixelscale)
        
        # Wave number & Coordinates
        k = 2*np.pi / wavel
        x_coords, y_coords = WF.get_xycoords()

        # First Phase Operation
        rho1 = np.exp(1.0j * k * (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront = rho1 * wavefront
        
        # Assume z > 0 for now
        wavefront = np.fft.ifftshift(wavefront)
        wavefront = np.fft.fft2(wavefront)
        wavefront = np.fft.fftshift(wavefront)
        wavefront *= pixelscale ** 2
        
        # Second Phase Operation
        rho2 = np.exp(1.0j * k * z_prop) / (1.0j * wavel * z_prop) * np.exp(1.0j * k * 
                                (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront_out = rho2 * wavefront

        # Update Wavefront Object
        WF = eqx.tree_at(lambda WF: WF.wavefront,  WF, wavefront_out)
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, None)
        params_dict["Wavefront"] = WF
        return params_dict