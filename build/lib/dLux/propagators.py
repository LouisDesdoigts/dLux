from .base import Layer
import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field

class FresnelProp(Layer):
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
    focal_length: float = static_field
    XX: ndarray = static_field
    YY: ndarray = static_field
    z: float
    
    def __init__(self, size, focal_length, z):
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
        self.size_in = size
        self.size_out = size
        self.focal_length = focal_length
        self.z = z

        # Check if this matches PSF centering
        xs = np.arange(0, size) - size//2 
        self.XX, self.YY = np.meshgrid(xs, xs)
        
    def _get_pixelscale(self, focal_length, pixelscale, z, wavel):
        """
        We calcualte these values outside of the function to help keep track of 
        wavefronts passing through multiple non-conjugate planes.
        
        Assumes a linear scaliing between initial and final pixel scale along
        z axis
        """
        # Calculate pixel scale
        pixelscale_conj = wavel * focal_length / (pixelscale * self.size_in)
        pixelscale_out = (pixelscale_conj - pixelscale) * (z / focal_length) + pixelscale
        return pixelscale_out
        
    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        Propagates Fresnel
        """
        
        z_prop = self.z
        
        # Wave number
        k = 2*np.pi / wavel
        
        # Coordinates & Pixelscale
        x_coords = self.XX * pixelscale
        y_coords = self.YY * pixelscale
        pixelscale_out = self._get_pixelscale(self.focal_length, pixelscale, self.z, wavel)
        
        # Units: pixels * m / pixel = m 'simulation size'
        s = self.size_in * pixelscale 
            
        # First Phase Operation
        rho1 = np.exp(1.0j * k * (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront *= rho1
        
        # Assume z > 0 for now
        wavefront = np.fft.ifftshift(wavefront)
        wavefront = np.fft.fft2(wavefront)
        wavefront = np.fft.fftshift(wavefront)
        wavefront *= pixelscale ** 2
        
        # Second Phase Operation
        rho2 = np.exp(1.0j * k * z_prop) / (1.0j * wavel * z_prop) * np.exp(1.0j * k * 
                                (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront *= rho2
        
        pixelscale_out_popppy = wavel * self.z / s
        # print(pixelscale_out)
        # print(pixelscale)
        
        return wavefront, pixelscale_out_popppy
    
class MFT(Layer):
    """
    Matches poppy but assumes square
    """
    focal_length: float = static_field()
    pixelscale_out: float = static_field()
    oversample: int = static_field()
    
    def __init__(self, size_in, size_out, oversample, focal_length, pixelscale_out):
        self.size_in = size_in
        self.size_out = size_out
        self.oversample = oversample
        self.focal_length = focal_length
        self.pixelscale_out = pixelscale_out
        
    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        Should we add offset here too?
        I have removed it but we have the code needed in the old notebooks
        
        Potentially use different parameters based on what inputs are given?
        
        Add shift parameter?
        """
        # Calculate NlamD parameter
        npup, npix = self.size_in, self.size_out
        wf_size_in = pixelscale * npup # Wavefront size
        aperture = wf_size_in / self.oversample # Aperture size
        det_size = self.pixelscale_out * npix # detector size
        wavel_scale = det_size * aperture / self.focal_length
        nlamD = wavel_scale / wavel
        
        # Calulate Arrays
        dX = 1.0 / float(npup)
        dU = nlamD / float(npix)
        
        offset = 0 # Keeping this here for potential later use
        Xs = (np.arange(npup, dtype=float) - float(npup) / 2.0 + offset) * dX
        Us = (np.arange(npix, dtype=float) - float(npix) / 2.0 + offset) * dU
        XU = np.outer(Xs, Us)
        expXU = np.exp(-2.0 * np.pi * 1j * XU)

        # Note: Can casue overflow issues on 32-bit
        norm_coeff = np.sqrt((nlamD**2) / (npup**2 * npix**2)) 

        # Perform MFT
        t1 = np.dot(expXU.T, wavefront)
        t2 = np.dot(t1, expXU)
        wavefront_out = norm_coeff * t2
        return wavefront_out, self.pixelscale_out

class FFT(Layer):
    focal_length: float = static_field()
    
    def __init__(self, size, focal_length):
        self.size_in = size
        self.size_out = size
        self.focal_length = focal_length
        
    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        Performs normalisation matching poppy
        """
        # Calculate Wavefront
        norm = wavefront.shape[0]
        wavefront_out = norm * np.fft.fftshift( np.fft.ifft2(wavefront) )
        
        # Calculate pixel scale
        pixelscale_out = wavel * focal_length / (pixelscale * self.size_in)
        return wavefront_out, pixelscale_out
    
class IFFT(Layer):
    focal_length: float = static_field()
    
    def __init__(self, size, focal_length):
        self.size_in = size
        self.size_out = size
        self.focal_length = focal_length
        
    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        Performs normalisation matching poppy
        """
        # Calculate Wavefront
        norm = 1./wavefront.shape[0]
        wavefront_out = norm * np.fft.fft2( np.fft.ifftshift(wavefront) )
        
        # Calculate pixel scale
        pixelscale_out = 2 * wavel * focal_length / (pixelscale * self.size_in)
        return wavefront_out, pixelscale_out