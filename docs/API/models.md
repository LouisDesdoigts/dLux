# Models Utility Functions

This module contains a single function used to generate a simple model of an optial system.

## Simple Optical System

This function generates a simple optical system with a single pupil and single focal plane. It requires the diameter of the aperture, the number of pixels representing the wavefront, the number of detector pixels, and the size of the detector pixels. Units can be either angular or cartesian by setting the `angular` flag. Aberrations can also be added, along with any other extra layers to be appended to the last layer of the pupil plane.

??? info "Simple Optical System API"
    ::: dLux.utils.models.simple_optical_system
