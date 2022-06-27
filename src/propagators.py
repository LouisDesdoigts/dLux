import jax.numpy as np
import equinox as eqx

class MFT(eqx.Module):
    """
    Matches poppy but assumes square
    """
    npix_out: int = eqx.static_field()
    tilt_wf: bool
    inverse: bool
    focal_length:   float
    pixelscale_out: float
    
    def __init__(self, npix_out, focal_length, pixelscale_out, tilt_wf=False, inverse=False):
        self.npix_out = int(npix_out)
        self.tilt_wf = tilt_wf
        self.inverse = inverse
        self.focal_length =   np.array(focal_length).astype(float)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Convert 
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavel
        pixelscale = WF.pixelscale
        offset = WF.offset if self.tilt_wf else np.array([0., 0.])
        
        # Calculate NlamD parameter (Do in __init__?)
        npup = wavefront.shape[0] 
        npix = self.npix_out
        
        wf_size_in = pixelscale * npup # Wavefront size 'd'        
        det_size = self.pixelscale_out * npix # detector size
        
        # wavel_scale =  wf_size_in * npix * pixel_rad
        wavel_scale = det_size * wf_size_in / self.focal_length
        num_fringe = wavel_scale / wavel
        
        # Calulate values
        offsetX, offsetY = offset * self.focal_length / self.pixelscale_out
        dX = 1.0 / float(npup)
        dY = 1.0 / float(npup)
        dU = num_fringe / float(npix)
        dV = num_fringe / float(npix)
        
        # Generate arrays
        Xs = (np.arange(npup, dtype=float) - float(npup)/2 + offsetX + 0.5) * dX
        Ys = (np.arange(npup, dtype=float) - float(npup)/2 + offsetY + 0.5) * dY
        Us = (np.arange(npix, dtype=float) - float(npix)/2 + offsetX + 0.5) * dU
        Vs = (np.arange(npix, dtype=float) - float(npix)/2 + offsetY + 0.5) * dV
        XU = np.outer(Xs, Us)
        YV = np.outer(Ys, Vs)

        # Propagate wavefront
        sign = -1 if self.inverse else +1
        expXU = np.exp(sign * -2.0 * np.pi * 1j * XU)
        expYV = np.exp(sign * -2.0 * np.pi * 1j * YV).T
        t1 = np.dot(expYV, wavefront)
        wavefront = np.dot(t1, expXU)

        # Normalise wavefront
        # norm_coeff = np.sqrt((num_fringe**2) / (npup**2 * npix**2))
        norm_coeff = np.exp(np.log(num_fringe) - (np.log(npup) + np.log(npix)))
        wavefront_out = wavefront * norm_coeff
        
        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict

class FFT(eqx.Module):
    """
    Note: FFT's natively center the zero frequency on a pixel center, basically
    fuck FFTs MFT 4 lyf
    """
    focal_length: float
    inverse: bool
    
    def __init__(self, focal_length, inverse=False):
        self.focal_length = np.array(focal_length).astype(float)
        self.inverse = inverse
        
    def __call__(self, params_dict):
        """
        Performs normalisation matching poppy
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavel
        pixelscale = WF.pixelscale

        # Calculate Wavefront & Pixelscale
        npix_in = wavefront.shape[0]
        if not self.inverse: # Forwards
            wavefront_out = npix_in * np.fft.fftshift(np.fft.ifft2(wavefront))
        else: # Inverse
            wavefront_out = 1./npix_in * np.fft.fft2(np.fft.ifftshift(wavefront))
        
        # Calculate Pixelscale
        pixelscale_out = wavel * self.focal_length / (pixelscale * npix_in)

        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, "Focal")
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
    npix_out: int
    focal_length: float
    focal_shift: float
    pixelscale_out: float
    
    def __init__(self, npix_out, focal_length, focal_shift, pixelscale_out):
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
        self.npix_out = int(npix_out)
        self.focal_length =   np.array(focal_length).astype(float)
        self.focal_shift = np.array(focal_shift).astype(float)
        self.pixelscale_out = np.array(pixelscale_out).astype(float)
        
    def __call__(self, params_dict):
        """
        Propagates Fresnel
        
        Note: Normalisation calcuated in the Far-Field
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavel
        pixelscale = WF.pixelscale

        # Coords
        x_coords, y_coords = WF.get_xycoords()
        r_coords = np.hypot(x_coords, y_coords)

        # Apply Thin Lens Equation
        k = 2*np.pi / wavel # Wavenumber
        wavefront *= np.exp(-0.5j * k * r_coords**2 * 1/self.focal_length)
        
        # Calc prop parameters
        npix = wavefront.shape[0]
        wf_size = npix * pixelscale
        det_size = self.npix_out * self.pixelscale_out
        z_prop = self.focal_length + self.focal_shift
        focal_ratio = self.focal_length / z_prop
        num_fringe_out = focal_ratio * wf_size * det_size / (wavel * self.focal_length)

        # First Phase Operation
        rho1 = np.exp(1.0j * k * (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront *= rho1
        wavefront = self.mft(wavefront, num_fringe_out, self.npix_out)
        wavefront *= pixelscale ** 2

        # Coords
        # NOTE: This needs to be able to match the cenetering convtion
        # of the input wavefront
        xs = np.arange(self.npix_out) - self.npix_out//2
        YY, XX = np.meshgrid(xs, xs)
        x_coords, y_coords = self.pixelscale_out * np.array([XX, YY])

        # Second Phase Operation
        rho2 = np.exp(1.0j * k * z_prop) / (1.0j * wavel * z_prop) * np.exp(1.0j * k * 
                                (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront_out = rho2 * wavefront

        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixelscale, WF, self.pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.planetype,  WF, None)
        params_dict["Wavefront"] = WF
        return params_dict
    
    def mft(self, wavefront, num_fringe, npix):
        """
        Minimal on-axis MFT function
        """

        # npix = wavefront.shape[0]
        npup = wavefront.shape[0]

        # Calulate Arrays
        dX = 1.0 / float(npup)
        dU = num_fringe / npix

        Xs = (np.arange(npup, dtype=float) - float(npup) / 2.0) * dX
        Us = (np.arange(npix, dtype=float) - float(npix) / 2.0) * dU
        XU = np.outer(Xs, Us)
        expXU = np.exp(-2.0 * np.pi * 1j * XU)

        # Note: Can casue overflow issues on 32-bit
        # norm_coeff = np.sqrt((num_fringe**2) / (npup**2 * npix**2))
        norm_coeff = np.exp(np.log(num_fringe) - (np.log(npup) + np.log(npix)))

        # Perform MFT
        t1 = np.dot(expXU.T, wavefront)
        t2 = np.dot(t1, expXU)
        wavefront_out = norm_coeff * t2

        return wavefront_out


class GaussianPropagator(Propagator):


    # Constants
    INDEX_GENERATOR = numpy.array([1, 2])


    def planar_to_planar(self: FresnelWavefront, distance: float) -> None:
        """
        Modifies the state of the wavefront by propagating a planar 
        wavefront to a planar wavefront. 

        Parameters
        ----------
        distance : float
            The distance of the propagation in metres.
        """
        wavefront = self.amplitude * \
            numpy.exp(1j * self.phase)

        new_wavefront = numpy.fft.ifft2(
            self.transfer_function(distance) * \
            numpy.fft.fft2(wavefront))

        amplitude = numpy.abs(new_wavefront)
        phase = self.calculate_phase(new_wavefront)
        return self.update_phasor(amplitude, phase)        


    def waist_to_spherical(self: FresnelWavefront, 
            distance: float) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        the waist of the gaussian beam to a spherical wavefront. 

        Parameters
        ----------
        distance : float 
            The distance of the propagation in metres.
        """
        coefficient = 1 / 1j / self.wavel / distance

        wavefront = self.amplitude * numpy.exp(1j * self.phase)

        # TODO: Check that these are the correct algorithms to 
        # use and ask if we are going to actually require the 
        # reverse direction
        fourier_transform = jax.lax.cond(numpy.sign(distance) > 0, 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                numpy.fft.fft2(wavefront), 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                numpy.fft.ifft2(wavefront),
            wavefront, distance)

        new_wavefront = coefficient * fourier_transform
        phase = self.calculate_phase(new_wavefront)
        amplitude = jax.numpy.abs(new_wavefront)
        self.update_phasor(amplitude, phase)

        # TODO: Confirm that I need to update the pixel scale for 
        # this transformation.


    def spherical_to_waist(self: FresnelWavefront, 
            distance: float) -> None:
        """
        Modifies the state of the wavefront by propagating it from 
        a spherical wavefront to the waist of the Gaussian beam. 

        Parameters
        ----------
        distance : float
            The distance of propagation in metres
        """
        coefficient = 1 / 1j / self.wavel / distance * \
            quadratic_phase_factor(distance)

        wavefront = self.amplitude * numpy.exp(1j * self.phase)

        # TODO: Check that these are the correct algorithms to 
        # use and ask if we are going to actually require the 
        # reverse direction
        fourier_transform = jax.lax.cond(numpy.sign(distance) > 0, 
            lambda wavefront: numpy.fft.fft2(wavefront), 
            lambda wavefront: numpy.fft.ifft2(wavefront),
            wavefront)

        new_wavefront = coefficient * fourier_transform
        phase = self.calculate_phase(new_wavefront)
        amplitude = jax.numpy.abs(new_wavefront)
        self.update_phasor(amplitude, phase)

        # TODO: Confirm that I need to update the pixel scale for 
        # this transformation.


    def outside_to_outside(self: FresnelWavefront, distance: float) -> None:
        """
        Propagation from outside the Rayleigh range to another 
        position outside the Rayleigh range. 

        Parameters
        ----------
        distance : float
            The distance to propagate in metres.
        """
        self.waist_to_spherical(self.position + distance - \
            self.location_of_waist()) * self.waist_to_spherical(
            self.location_of_waist() - self.position)


    def outside_to_inside(self: FresnelWavefront, distance: float) -> None:
        """
        Propagation from outside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        distance : float
            The distance to propagate in metres.
        """
        self.planar_to_planar(self.position + distance - \
            self.location_of_waist()) * self.spherical_to_waist(
            self.location_of_waist() - self.position)


    def inside_to_inside(self: FresnelWavefront, distance: float) -> None:
        """
        Propagation from inside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        distance : float
            The distance to propagate in metres.
        """
        # Consider removing after checking the Jaxpr for speed. 
        # This is just an alias for another function. 
        self.planar_to_planar(distance)


    def inside_to_outside(self: FresnelWavefront, distance: float) -> None:
        """
        Propagation from inside the Rayleigh range to outside the 
        rayleigh range. 

        Parameters
        ----------
        distance : float
            The distance to propagate in metres.
        """
        self.waist_to_spherical(self.position + distance - \
            self.location_of_waist()) * planar_to_planar(
            self.location_of_waist() - self.position)


    def propagate(self: FresnelWavefront, distance: float) -> None:
        """
        Propagates the wavefront approximated by a Gaussian beam 
        the amount specified by distance. Note that distance can 
        be negative.

        Parameters 
        ----------
        distance : float
            The distance to propagate in metres.
        """
        # This works by considering the current position and distnace 
        # as a boolean array. The INDEX_GENERATOR converts this to 
        # and index according to the following table.
        #
        # sum((0, 0) * (1, 2)) == 0
        # sum((1, 0) * (1, 2)) == 1
        # sum((0, 1) * (1, 2)) == 2
        # sum((1, 1) * (1, 2)) == 3
        #
        # TODO: Test if a simple lookup is faster. 
        decision_vector = self.is_inside([0., distance])
        decision_index = numpy.sum(self.INDEX_GENERATOR * descision_vector)
 
        # Enters the correct function differentiably depending on 
        # the descision.
        jax.lax.switch(decision_index, 
            [self.inside_to_inside, self.inside_to_outside,
            self.outside_to_inside, self.outside_to_outside],
            distance) 
