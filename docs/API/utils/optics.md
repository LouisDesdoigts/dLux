# Optics Utility Functions

This module contains a number of common equations used in optics, such as converting between Optical Path Difference (OPD) and phase, and a few functions used to calculate sampling rates in focal planes.

---

## OPD to Phase

Converts an input OPD and wavelength (metres) to phase (radians).

??? info "OPD to Phase API"
    ::: dLux.utils.optics.opd_to_phase

---

## Phase to OPD

Converts an input phase (radians) and wavelength (metres) to OPD (metres).

??? info "Phase to OPD API"
    ::: dLux.utils.optics.phase_to_opd

---

## Get Fringe Size

Calculates the angular size of the diffraction fringes (radians) in a focal plane based on the wavelength and aperture (metres).

??? info "Get Fringe Size API"
    ::: dLux.utils.optics.get_fringe_size

---

## Get Pixels Per Fringe

Calculates the number of pixels per diffraction fringe in a focal plane based on the wavelength and aperture in metres. A Nyquist-sampled system will have 2 pixels per fringe.

??? info "Get Pixels Per Fringe API"
    ::: dLux.utils.optics.get_pixels_per_fringe

---

## Get Pixel Scale

Calculates the required pixel scale (radians/pixel or metres/pixel) in a focal plane based on the wavelength and aperture (metres) in order to sample the diffraction fringes by some sampling rate. A sampling rate of 2 will give a Nyquist-sampled system. If a focal length is provided, the output will be in metres per pixel, otherwise it will be in radians per pixel.

??? info "Get Pixel Scale API"
    ::: dLux.utils.optics.get_pixel_scale

---

## Get Airy Pixel Scale

Calculates the required pixel scale (radians/pixel or metres/pixel) in a focal plane based on the wavelength and aperture (metres) in order to sample the diffraction fringes by some sampling rate, based on the slightly larger diffraction fringes given by an Airy disk PSF. A sampling rate of 2 will give a Nyquist-sampled system. If a focal length is provided the output will be in metres per pixel, otherwise it will be in radians per pixel.

??? info "Get Airy Pixel Scale API"
    ::: dLux.utils.optics.get_airy_pixel_scale