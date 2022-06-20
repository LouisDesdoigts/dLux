def prop_p2p(wavefront, wavel, wf_pixelscale, zi, zf):
    """
    wf_size: Physical size of total array, ie including padding (m)
    wf_pixelscale: Pixelscale of wavefront array, (m/pix)
    zi: Current propagated distance
    zf: Final Propaged distance
    """
    npix = wavefront.shape[0]
    xs = np.arange(0, npix) - npix//2
    XX, YY = np.meshgrid(xs, xs)
    x_coords = XX * self.pixelscale # (m)
    y_coords = YY * self.pixelscale # (m)
    
    rhosqr = np.fft.fftshift((x_coords / (wf_pixelscale ** 2 * npix)) ** 2 + 
                             (y_coords / (wf_pixelscale ** 2 * npix)) ** 2)
    
    dz = zf - zi
    exp_t = np.exp(-1.0j * np.pi * wavel * dz * rhosqr)
    
    wavefront = np.fft.fft2(wavefront) # Unnormalised, potentially use FFT class normalisation
    wavefront *= rhosqr
    wavefront = np.fft.ifft2(wavefront) # Unnormalised, potentially use IFFT class normalisation
    
    # pixelscale remains unchaged in the plane to plane propagation regieme
    return wavefront, wf_pixelscale

def get_quad_phase(wavefront, pixelscale, z, shift):
    """
    Quadratic phase term for spherical wavefront
    """
    npix = wavefront.shape[0]
    xs = np.arange(0, npix) - npix//2
    XX, YY = np.meshgrid(xs, xs)
    x_coords = XX * self.pixelscale # (m)
    y_coords = YY * self.pixelscale # (m)
    opd = (x_coords ** 2 + y_coords ** 2)  / (2.0 *z)
    return opd

def prop_w2s(wavefront, wavel, wf_pixelscale, zi, zf):
    """
    Waist to spherical
    """
    dz = zf-zi
    wavefront *= get_quad_phase(wavefront, wf_pixelscale, dz)
    
    # SIGN CONVENTION: forward optical propagations want a positive sign in the complex exponential, which
    # numpy implements as an "inverse" FFT
    if dz > 0:
        np.fft.ifft2(wavefront)
    else:
        np.fft.fft2(wavefront)
        
    npix = wavefront.shape[0]

    pixelscale_out = wavel * np.abs(dz) / (npix * wf_pixelscale)
    return wavefront, pixelscale_out


def prop_s2w(wavefront, wavel, wf_pixelscale, zi, zf):
    """
    
    """
    dz = zf-zi

    # SIGN CONVENTION: forward optical propagations want a positive sign in the complex exponential, which
    # numpy implements as an "inverse" FFT
    if dz > 0:
        np.fft.ifft2(wavefront)
    else:
        np.fft.fft2(wavefront)

    # update to new pixel scale before applying curvature
    pixelscale_out = wavel * np.abs(dz) / (npix * wf_pixelscale)
    wavefront *= get_quad_phase(wavefront, pixelscale_out, dz)
    return wavefront, pixelscale_out


def planar_range(self, z):
    """
    Returns True if the input range z is within the Rayleigh range of the waist.

    Parameters
    -----------
    z : float
        distance from the beam waist

    """
    return np.abs(self.z_w0 - z) < self.z_r

def r_c(self, z=None):
    """
    The gaussian beam radius of curvature as a function of distance z

    Parameters
    -------------
    z : float, optional
        Distance along the optical axis.
        If not specified, the wavefront's current z coordinate will
        be used, returning the beam radius of curvature at the current position.

    Returns
    -------
    Astropy.units.Quantity of dimension length

    """
    dz = (z - self.z_w0)  # z relative to waist
    if dz == 0:
        return np.inf * u.m
    return dz * (1 + (self.z_r / dz) ** 2)

def spot_radius(self, z=None):
    """
    radius of a propagating gaussian wavefront, at a distance z

    Parameters
    -------------
    z : float, optional
        Distance along the optical axis.
        If not specified, the wavefront's current z coordinate will
        be used, returning the beam radius at the current position.

    Returns
    -------
    Astropy.units.Quantity of dimension length
    """
    return self.w_0 * np.sqrt(1.0 + ((z - self.z_w0) / self.z_r) ** 2)


@property
def z_r(self):
    """
    Rayleigh distance for the gaussian beam, based on
    current beam waist and wavelength.

    I.e. the distance along the propagation direction from the
    beam waist at which the area of the cross section has doubled.
    The depth of focus is conventionally twice this distance.
    """

    return np.pi * self.w_0 ** 2 / self.wavelength

@property
def divergence(self):
    """
    Half-angle divergence of the gaussian beam

    I.e.  the angle between the optical axis and the beam radius (at a large distance  from the waist) in radians.
    """
    return self.wavelength / (np.pi * self.w_0)


    self.w_0 = beam_radius.to(self.units)  # convert to base units.
    """Beam waist radius at initial plane"""
    self.z = 0 * units
    """Current wavefront coordinate along the optical axis"""
    self.z_w0 = 0 * units
    """Coordinate along the optical axis of the latest beam waist"""
    self.waists_w0 = [self.w_0.to(u.m).value]
    """List of beam waist radii, in series as encountered during the course of an optical propagation."""
    self.waists_z = [self.z_w0.to(u.m).value]
    """List of beam waist distances along the optical axis, in series as encountered
    during the course of an optical propagation."""
    self.spherical = False
    """Is this wavefront spherical or planar?"""
    self.k = np.pi * 2.0 / self.wavelength
    """ Wavenumber"""
    self.rayleigh_factor = rayleigh_factor
    """Threshold for considering a wave spherical, in units of Rayleigh distance"""

    self.focal_length = np.inf * u.m
    """Focal length of the current beam, or infinity if not a focused beam"""
