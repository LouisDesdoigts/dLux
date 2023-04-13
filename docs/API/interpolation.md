# Interpolation Utility Functions

This module contains some basic interpolation functions that can be used on arrays and wavefronts.

## Interpolate

Paraxially interpolates an array by some sampling ratio with optional shifts.

??? info "Interpolate API"
    ::: dLux.utils.interpolation.interpolate

## Scale Array

Scales an array to some number of output pixels.

??? info "Scale Array API"
    ::: dLux.utils.interpolation.scale_array

## Interpolate Field

Paraxially interpolates a wavefront by some sampling ratio with optional shifts. Inputs of the wavefront arrays can be either amplitude and phase or real and imaginary.

??? info "Interpolate Field API"
    ::: dLux.utils.interpolation.interpolate_field

## Rotate

Rotates an array via interpolation by some angle in radians.

??? info "Rotate API"
    ::: dLux.utils.interpolation.rotate

## Fourier Rotate

Rotates an array via Fourier methods by some angle in radians.

??? info "Fourier Rotate API"
    ::: dLux.utils.interpolation.fourier_rotate

## Generate Coordiantes

Just a helper function used to generate the coordaintes required by the low-level interpolation functions.

??? info "Generate Coordiantes API"
    ::: dLux.utils.interpolation.generate_coordinates