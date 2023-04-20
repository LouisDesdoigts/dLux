import jax.numpy as np
import dLux
from jax import Array


__all__ = ["simple_optical_system", "toliman"]


def simple_optical_system(aperture_diameter         : Array,
                          wavefront_npixels         : int,
                          detector_npixels          : int,
                          detector_pixel_size       : Array,
                          angular                   : bool  = True,
                          focal_length              : Array = None,
                          secondary_mirror_diameter : Array = None,
                          nzernike                  : int   = None,
                          zernike_coefficients      : Array = None,
                          extra_layers              : list  = None,
                          return_layers             : bool  = False):
    """
    Constucts a simple Fourier optical system.

    Parameters
    ----------
    aperture_diameter : Array, meters
        The diameter of the optical system aperture.
    wavefront_npixels : int
        The number of pixel used to represent the wavefront.
    detector_npixels : int
        The number of pixel of the detector
    detector_pixel_size : Array, arcseconds/pixel or meters/pixel
        The size of the detector pixels. Taken in units of arcseconds per pixel
        if anuglar == True, else units are taken in meters per pixel.
    angular : bool = True
        Whether to use angular (radians) or cartesian (meters) units.
    focal_length : Array = None
        The focal length of the optical system. This paramter is only used if
        angular == False.
    secondary_mirror_diameter : Array = None
        The diameter of the secondary mirror obscuration.
    nzernike : int = None
        The number of zernike terms to use. Ignore piston tip tilt.
    zernike_coefficients : Array = None
        The values of the zernike coefficients. Only used if nzerike == None.
    extra_layers : list = None
        The extra layers to add to the optical system.
    return_layers : bool = False
        Should the function return the layers, or an Optics class.

    Returns
    -------
    optics : OpticalSystem
        The optical system with the optical layers loaded.
    """
    # Inputs checks
    aperture_diameter = np.asarray(aperture_diameter, dtype=float)
    assert aperture_diameter.ndim == 0, ("aperture_diameter must be scalar.")

    assert isinstance(wavefront_npixels, int), \
    ("wavefront_npixels must be an integer.")

    assert isinstance(detector_npixels, int), \
    ("detector_npixels must be an integer.")
    detector_pixel_size = np.asarray(detector_pixel_size, dtype=float)

    assert detector_pixel_size.ndim == 0, \
    ("detector_pixel_size must be scalar.")

    assert isinstance(angular, bool), ("angular must be a boolean.")

    if not angular:
        focal_length = np.asarray(focal_length, dtype=float)
        assert focal_length.ndim == 0, ("focal_length must be scalar.")

    if secondary_mirror_diameter is not None:
        secondary_mirror_diameter = np.asarray(secondary_mirror_diameter, \
                                               dtype=float)
        assert secondary_mirror_diameter.ndim == 0, \
        ("focal_length must be scalar.")

    if nzernike is not None:
        assert isinstance(nzernike, int), ("nzernike must an integer.")

    if zernike_coefficients is not None:
        zernike_coefficients = np.asarray(zernike_coefficients, dtype=float)
        assert zernike_coefficients.ndim == 1, \
        ("zernike_coefficients must a one dimensional array.")
        assert len(zernike_coefficients) == nzernike, \
        ("The lenght of zernike_coefficients must be equal to nzerike.")

    if extra_layers is not None:
        assert isinstance(extra_layers, list), ("extra_layers must be a list.")
        for layer in extra_layers:
            assert isinstance(layer, dLux.optics.OpticalLayer), \
            ("Each item in extra_layers must be a dLux OpticalLayer.")

    assert isinstance(return_layers, bool), "return_layers must be a boolean."

    # Create wavefront
    layers = [dLux.optics.CreateWavefront(wavefront_npixels,
                                            aperture_diameter)]

    # Construct aperture
    if secondary_mirror_diameter is not None:
        secondary_ratio = secondary_mirror_diameter/aperture_diameter
    else:
        secondary_ratio = 0.
    
    layers += [dLux.ApertureFactory(wavefront_npixels, 
        secondary_ratio=secondary_ratio)]

    # Aberrations
    if nzernike is not None:
        zernikes = np.arange(3, nzernike + 3)
        if zernike_coefficients is not None:
            coeffs = zernike_coefficients
        else:
            coeffs = np.zeros(nzernike)
        
        layers += [dLux.AberrationFactory(wavefront_npixels, zernikes=zernikes,
            coefficients=coeffs)]


    # Extra Layers
    if extra_layers is not None:
        for layer in extra_layers:
            layers.append(layer)

    # Normalised wavefront
    layers += [dLux.NormaliseWavefront()]

    # Propagator
    if angular:
        layers += [dLux.AngularMFT(detector_npixels,
                        dLux.utils.arcseconds_to_radians(detector_pixel_size))]
    else:
        layers += [dLux.CartesianMFT(detector_npixels, detector_pixel_size,
                                   focal_length)]

    # Return optics or layers
    return layers if return_layers else dLux.core.Optics(layers)


def toliman(wavefront_npixels         : int,
            detector_npixels          : int,
            aperture_diameter         : Array = 0.13,
            secondary_mirror_diameter : Array = 0.032,
            detector_pixel_size       : Array = 0.375,
            angular                   : bool  = True,
            focal_length              : Array = None,
            nzernike                  : int   = None,
            zernike_coefficients      : Array = None,
            extra_layers              : list  = None,
            return_layers             : bool  = False):
    """
    Gets a simple Toliman optical system by calling the simple_optical_system
    function with pre-loaded values.

    Parameters
    ----------
    wavefront_npixels : int
        The number of pixel used to represent the wavefront.
    detector_npixels : int
        The number of pixel of the detector
    aperture_diameter : Array, meters = 0.13
        The diameter of the optical system aperture.
    secondary_mirror_diameter : Array = 0.032
        The diameter of the secondary mirror obscuration.
    detector_pixel_size : Array, arcseconds/pixel or meters/pixel = 0.375
        The size of the detector pixels. Taken in units of arcseconds per pixel
        if anuglar == True, else units are taken in meters per pixel. The
        physical detector pixels are 2.4um
    angular : bool = True
        Whether to use angular (radians) or cartesian (meters) units.
    focal_length : Array = None
        The focal length of the optical system. This paramter is only used if
        angular == False. The EFL is 1.32m.
    nzernike : int = None
        The number of zernike terms to use. Ignore piston tip tilt.
    zernike_coefficients : Array = None
        The values of the zernike coefficients. Only used if nzerike == None.
    extra_layers : list = None
        The extra layers to add to the optical system.
    return_layers : bool = False
        Should the function return the layers, or an Optics class.

    Returns
    -------
    optics : OpticalSystem
        The optical system with the optical layers loaded.
    """
    return simple_optical_system(
                        aperture_diameter,
                        wavefront_npixels,
                        detector_npixels,
                        detector_pixel_size=detector_pixel_size,
                        angular=angular,
                        focal_length=focal_length,
                        secondary_mirror_diameter=secondary_mirror_diameter,
                        nzernike=nzernike,
                        zernike_coefficients=zernike_coefficients,
                        extra_layers=extra_layers,
                        return_layers=return_layers)