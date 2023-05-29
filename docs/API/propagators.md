# Propagators: propagators.py

This module contains the classes that define the behaviour of PropagatorLayers in dLux.

These classes do not implement the propagation functionality themselves, but instead store the parameters of the propagation and call the inbuild methods of the Wavefront class, so its API is essentially a mirror of those methods.

There are four public classes:

- `MFT`
- `FFT`
- `ShiftedMFT`
- `FarFieldFresnel`

### `MFT(npixels, pixel_scale, focal_length=None, inverse=False)`

Performs a Matrix Fourier Transform (MFT) on the wavefront, propagating from Pupil to Focal planes. If the focal_length is None, the pixel_scale is assumed to be in angular units (radians), otherwise it is assumed to be in cartesian units (meters).

??? info "MFT API"
    ::: dLux.propagators.MFT

### `FFT(pad, focal_length=None, inverse=False)`

Performs a Fast Fourier Transform (FFT) on the wavefront, propagating from Pupil to Focal planes. If the focal_length is None, the output units will angular, otherwise cartesian.

??? info "FFT API"
    ::: dLux.propagators.FFT

### `shiftedMFT(npixels, pixel_scale, shift, focal_length=None, pixel=True, inverse=False)`

Performs a Matrix Fourier Transform (MFT) on the wavefront, propagating from Pupil to Focal planes. If the focal_length is None, the pixel_scale is assumed to be in angular units (radians), otherwise it is assumed to be in cartesian units (meters). The shift parameter is used to shift the center of the output plane by 'shift', which is treated in units of pixels by default, otherwise it is treated in the units of the pixel_scale.

??? info "Shifted MFT API"
    ::: dLux.propagators.ShiftedMFT

### `FarFieldFresnel(npixels, pixel_scale, focal_length, focal_shift, shift, pixel=True)`

Performs a Fresnel propagation on the wavefront, propagating from Pupil to Focal planes. The focal_shift parameter represents the distance from the focal plane at which the PSF is modelled. The shift parameter is used to shift the center of the output plane by 'shift', which is treated in units of pixels by default, otherwise it is treated in the units of the pixel_scale.

??? info "Far Field Fresnel API"
    ::: dLux.propagators.FarFieldFresnel