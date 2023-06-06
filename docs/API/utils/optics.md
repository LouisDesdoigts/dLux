# Optics Utility Functions

This module contains a number of common equations used in optics, such as converting between OPDs and phases, and a few functions used to calculate sampling rates in focal planes.

---

## OPD to Phase

Converts an input OPD and wavelength in metres to converts it to phase in radians.

??? info "OPD to Phase API"
    ::: dLux.utils.optics.opd_to_phase

---

## Phase to OPD

Converts an input phase in radians and wavelength in metres to converts it to OPD in metres.

??? info "Phase to OPD API"
    ::: dLux.utils.optics.phase_to_opd

---

## Get Fringe Size

Calculates the angular size in radians of the diffraction fringes in a focal plane based on the wavelength and aperture in metres.

??? info "Get Fringe Size API"
    ::: dLux.utils.optics.get_fringe_size

---

## Get Pixels Per Fringe

Calculates the number of pixels per diffraction fringe in a focal plane based on the wavelength and aperture in metres. A Nyquist sampled system will have 2 pixels per fringe.

??? info "Get Pixels Per Fringe API"
    ::: dLux.utils.optics.get_pixels_per_fringe

---

## Get Pixel Scale

Calculates the pixel scale in either radians or metres per pixel in a focal plane based on the wavelength and aperture in metres, in order to sample the diffraction fringes by some smapling rate. A sampling rate of 2 will give a Nyquist sampled system. If a focal length is provided the output will be in metres per pixel, otherwise it will be in radians per pixel.

??? info "Get Pixel Scale API"
    ::: dLux.utils.optics.get_pixel_scale

---

## Get Airy Pixel Scale

Calculates the pixel scale in either radians or metres per pixel in a focal plane based on the wavelength and aperture in metres, in order to sample the diffraction fringes by some smapling rate, based on the slightly larger diffraction fringes given by an airy disk PSF. A sampling rate of 2 will give a Nyquist sampled system. If a focal length is provided the output will be in metres per pixel, otherwise it will be in radians per pixel.

??? info "Get Airy Pixel Scale API"
    ::: dLux.utils.optics.get_airy_pixel_scale