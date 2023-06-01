# Wavefronts: `wavefronts.py`

This module contains the classes that define the behaviour of wavefronts in ∂Lux.

There are two public classes: `Wavefront` and `FresnelWavefront`.

All `Wavefront` objects have the following attributes:

- `wavelength`
- `amplitude`
- `phase`
- `pixel_scale`
- `plane`
- `units`

The `wavelength`, `amplitude` and `phase` attributes are the respective wavelength, amplitude and phase of the wavefront. The `pixel_scale` is the physical size of the pixels representing the wavefront. The `plane` is the current plane of the wavefront, which can be '_Pupil_', '_Focal_' or '_Intermediate_'. The `units` attribute is the current units of the wavefront, which can be '_Cartesian_' or '_Angular_'.

It is important to note here that `plane` and `units` do not actually determine the behaviour of the wavefront, but rather are used to track the current unit type. That is, if you set the units parameter from '_Cartesian_' to '_Angular_', the wavefront will not be converted to angular units, but rather the units will be _treated_ as angular in propagations, which can lead to incorrect results.

??? info "Wavefront API"
    :::dLux.wavefronts.Wavefront

---

# Operators

The wavefront classes contain a series of methods that allow for the manipulation and propagation of the wavefront. It also has a series of operators that allow for the manipulation of the wavefront via standard arithmetic operators, i.e.:

```python
import jax.numpy as np
import dLux as dl

npixels = 16
diameter = 1 # metres
wavelength = 1e-6 # metres

wf = dl.Wavefront(npixels, diameter, wavelength)

# Multiply to modify the amplitude
wf *= 0.5 # Halves the amplitude

# Multiply by complex array to transform both amplitude and phase
wf *= np.ones((npixels, npixels)) * np.exp(1j * np.zeros((npixels, npixels)))

# Multiply or Add by OpticalLayer to apply it
aperture = dl.ApertureFactory(16)
wf *= aperture
wf += aperture

# Add to modify the phase through OPD units
wf += 1 # Adds 1m of OPD across full wavefront
```

---

# Property Methods

The `Wavefront` classes also have a series of property methods for ease of use:

`diameter` Returns the current wavefront diameter calculated with the pixel scale
and number of pixels.

`npixels` Returns the side length of the arrays currently representing the
wavefront. Taken from the last axis of the amplitude array.

`real` Returns the real component of the `Wavefront`.

`imaginary` Returns the imaginary component of the `Wavefront`.

`phasor` Returns the electric field phasor described by the `Wavefront` in complex form.

`psf` Calculates the Point Spread Function (PSF), i.e. the squared modulus
of the complex wavefront.

`coordinates` Returns the physical positions of the wavefront pixels in meters.

`wavenumber` Returns the wavenumber of the wavefront ($2\pi/\lambda$).

`fringe_size` Returns the size of the fringes in angular units, i.e $\lambda/D$.

---

# General Methods

On top of these, the wavefront classes implement a number of methods that allow for the
manipulation of the image:

`add_opd(opd)` Applies the input array as an OPD to the wavefront.

`add_phase(phase)` Applies input array to the phase of the wavefront.

`tilt(angles)` Tilts the wavefront by the angles in $(x, y)$, by modifying the phase arrays.

`normalise()` Normalises the total wavefront power to unity.

`flip(axis)` Flips the amplitude and phase of the wavefront along the specified axes.

`scale_to(npixels, pixel_scale)` Performs a paraxial interpolation on the wavefront, determined by the `pixel_scale` and `npixels` parameters.

`rotate(angle)` Performs a paraxial rotation on the wavefront, determined by the `angle` parameter, using interpolation.

`pad_to(npixels)` Paraxially zero-pads the `Wavefront` to the size determined by `npixels`. Note this only supports padding arrays of even dimension to even dimension, and odd dimension to to odd dimension, e.g. $2 \rightarrow 4$ or $3 \rightarrow 5$.

`crop_to(npixels)` Paraxially crops the `Wavefront` to the size determined by `npixels`. Note this only supports cropping arrays of even dimension to even dimension, and odd dimension to to odd dimension, e.g. $4 \rightarrow 2$ or $5 \rightarrow 3$.

---

# Propagator Methods

`Wavefront` objects also have methods used to propagate them between planes. These
methods are:

`FFT(pad, focal_length=None)` Performs a Fast Fourier Transform (FFT) on the wavefront, propagating from Pupil to Focal plane. If the `focal_length` is None, the output units will angular, otherwise cartesian.

`IFFT(pad, focal_length=None)` Performs an Inverse Fast Fourier Transform (IFFT) on the wavefront, propagating from Focal to Pupil plane.

`MFT(npixels, pixel_scale, focal_length=None)` Performs a Matrix Fourier Transform (MFT) on the wavefront, propagating from Pupil to Focal plane. If the `focal_length` is None, the `pixel_scale` is assumed to be in angular units (radians), otherwise it is assumed to be in cartesian units (metres).

`IMFT(npixels, pixel_scale, focal_length=None)` Performs an Inverse Matrix Fourier Transform (IMFT) on the wavefront, propagating from Focal to Pupil plane.

`shifted_MFT(npixels, pixel_scale, shift, focal_length=None, pixel=True)` Performs a Matrix Fourier Transform (MFT) on the wavefront, propagating from Pupil to Focal plane. If the `focal_length` is None, the `pixel_scale` is assumed to be in angular units (radians), otherwise it is assumed to be in cartesian units (meters). The `shift` parameter is used to shift the center of the output plane, which is treated in units of pixels by default, otherwise it is treated in the units of the `pixel_scale`.

`shifted_IMFT(npixels, pixel_scale, shift, focal_length=None, pixel=True)` Performs an Inverse Matrix Fourier Transform (IMFT) on the wavefront, propagating from Focal to Pupil plane. The `shift` parameter is used to shift the center of the output plane, which is treated in units of pixels by default, otherwise it is treated in the units of the `pixel_scale`.

---

# Fresnel Wavefront

The `FresnelWavefront` class is a subclass of the `Wavefront`, implementing the methods required to perform a FarFieldFresnel propagation (i.e., close to the focal plane). It implements two more methods that are essential extensions of the MFT methods. This class _only_ implements cartesian propagation and only propagates from the Pupil plane to the Focal plane.

??? info "Fresnel Wavefront API"
    :::dLux.wavefronts.FresnelWavefront

`fresnel_prop(npixels, pixel_scale, focal_length, focal_shift)` Performs a Fresnel propagation on the wavefront, propagating from Pupil to Focal plane. The `focal_shift` parameter represents the distance from the focal plane at which the PSF is modelled.

`shifted_fresnel_prop(npixels pixel_scale, shift, focal_length, focal_shift, pixel=True)` Performs a Fresnel propagation on the wavefront, propagating from Pupil to Focal plane. The `focal_shift` parameter represents the distance from the focal plane at which the PSF is modelled. The shift parameter is used to shift the center of the output plane, which is treated in units of pixels by default, otherwise it is treated in the units of the `pixel_scale`.