# Working with OpticalSystem Objects

This tutorial is designed to give an overview of the main class in dLux - The `OpticalSystem` class. 


```python
# Basic imports
import jax.numpy as np

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
```

## Overview

There are three `OpticalSystem`s implemented in dLux:

1. `LayeredOpticalSystem`
2. `AngularOpticalSystem`
3. `CartesianOpticalSystem`

All are constructed similarly, and share the the following attributes:

- `wf_npixels`
- `diameter`
- `layers`

The `wf_npixls` parameter defines the number of pixels used to initialise the wavefront, `diameter` defines the diameter of the wavefront in meters, and `layers` is a list of `OpticalLayer` objects that define the transformations to that wavefront.

The `AngularOpticalSystem` and `CartesianOpticalSystem` are both subclasses of the `LayeredOpticalSystem` class, extending it to include three extra attributes:

- `psf_npixels`
- `psf_pixel_scale`
- `oversample`

These attributes define the size of the PSF, the pixel scale of the PSF, and the oversampling factor used when calculating the PSF. The difference between the two is that the `AngularOpticalSystem` has `psf_pixel_scale` in units of arcseconds, while the `CartesianOpticalSystem` has `psf_pixel_scale` in units of microns. Note that an oversample of 2 will result in an output psf with shape `(2 * psf_npixels, 2 * psf_npixels)`, with the idea that the PSF will be downsampled later to the correct size and pixel scale. 

Beyond this, the `CartesianOpticalSystem` has an extra attribute `focal_length`, with units of meters.

Now lets create a minimal `AnguarOpticalSystem` to demonstrate how to use these classes.


```python
# Define our wavefront properties
wf_npix = 512  # Number of pixels in the wavefront
diameter = 1.0  # Diameter of the wavefront, meters

# Construct a simple circular aperture
coords = dlu.pixel_coords(wf_npix, diameter)
aperture = dlu.circle(coords, 0.5 * diameter)

# Define our detector properties
psf_npix = 64  # Number of pixels in the PSF
psf_pixel_scale = 50e-3  # 50 mili-arcseconds
oversample = 3  # Oversampling factor for the PSF

# Define the optical layers
layers = [('aperture', dl.layers.Optic(aperture, normalise=True))]

# Construct the optics object
optics = dl.AngularOpticalSystem(
    wf_npix, diameter, layers, psf_npix, psf_pixel_scale, oversample
)

# Let examine the optics object! The dLux framework has in-built
# pretty-printing, so we can just print the object to see what it contains.
print(optics)
```

    AngularOpticalSystem(
      wf_npixels=512,
      diameter=1.0,
      layers={
        'aperture':
        Optic(opd=None, phase=None, transmission=f32[512,512], normalise=True)
      },
      psf_npixels=64,
      oversample=3,
      psf_pixel_scale=0.05
    )



## Methods

All three of these object are quite similar, and share the same three primary methods:

1. `.propagate_mono`
2. `.propagate`
3. `.model`

Lets look at them one-by-one.

## `propagate_mono`

`propagate_mono` has the following signature: `optics.propagate_mono(wavelength, offset=np.zeros(2), return_wf=False)`

- `wavelength` is the wavelength of the light to propagate, in meters
- `offset` is the offset of the source from the center of optical system, in radians
- `return_wf` is a boolean flag that determines whether the wavefront object should be returned, as opposed to the psf array.

Note that the `propagate_mono` method should generally not be used, as its functionality is superceeded by the `propagate` method, but lets look at how it works anyway.


```python
# 1 micron wavelength
wavelength = 1e-6 

# 5-pixel offset in the x-direction
shift = np.array([5 * psf_pixel_scale, 0])
offset = dlu.arcsec2rad(shift)

# Propagate a psf
psf = optics.propagate_mono(wavelength, offset)

# Plot the results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture Transmission")
plt.imshow(optics.transmission)
plt.colorbar(label="Transmission")

plt.subplot(1, 2, 2)
plt.title("Sqrt PSF")
plt.imshow(psf**0.5)
plt.colorbar(label="Sqrt Intensity")

plt.tight_layout()
plt.show()
```


    
![png](optical_systems_files/optical_systems_6_0.png)
    


We can also return the `Wavefront` object too, allowing us to look at the amplitude, phase, and any other properties.


```python
# Get the Wavefront
wf = optics.propagate_mono(wavelength, offset, return_wf=True)

# First we examine the wavefront object
print(wf)
```

    Wavefront(
      wavelength=f32[],
      pixel_scale=f32[],
      amplitude=f32[192,192],
      phase=f32[192,192],
      plane='Focal',
      units='Angular'
    )



```python
# Get the amplitude and phase
amplitude = wf.amplitude
phase = wf.phase

# Get the fov for plotting
fov = dlu.rad2arcsec(wf.diameter)
extent = [-fov/2, fov/2, -fov/2, fov/2]

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Amplitude')
plt.imshow(amplitude, extent=extent)
plt.colorbar()
plt.xlabel('x (arcseconds)')
plt.ylabel('y (arcseconds)')

plt.subplot(1, 2, 2)
plt.title('Phase')
plt.imshow(phase, extent=extent)
plt.colorbar()
plt.xlabel('x (arcseconds)')
plt.ylabel('y (arcseconds)')

plt.tight_layout()
plt.show()
```


    
![png](optical_systems_files/optical_systems_9_0.png)
    


## `propagate`

`propagate` is the core propagation function of optical systems. It has the following signature: `optics.propagate(wavelengths, offsets=np.zeros(2), weights=None, return_wf=False, return_psf=False)`


- `wavelengths` is an array of wavelengths to propagate, in meters
- `offset` is the offset of the source from the center of optical system, in radians
- `weights` is an array of weights to apply to each wavelength. If `None`, then all wavelengths are weighted equally.
- `return_wf` is a boolean flag that determines whether the `Wavefront` object should be returned, as opposed to the psf array.
- `return_psf` is a boolean flag that determines whether the `PSF` object should be returned, as opposed to the psf array.

Lets see how to ues it.


```python
# Wavelengths array - Note we can also pass in a single float value!
wavelengths = 1e-6 * np.linspace(0.9, 1.1, 10)

# Weights array - Note these are relative weights, the input
# is automatically normalised
weights = np.linspace(0.5, 1.5, len(wavelengths))

# 5-pixel offset in the x-direction
shift = np.array([5 * psf_pixel_scale, 0])
offset = dlu.arcsec2rad(shift)

# Propagate a psf
psf = optics.propagate(wavelengths, offset, weights)

# Plot the results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aperture Transmission")
plt.imshow(optics.transmission)
plt.colorbar(label="Transmission")

plt.subplot(1, 2, 2)
plt.title("Sqrt PSF")
plt.imshow(psf**0.5)
plt.colorbar(label="Sqrt Intensity")

plt.tight_layout()
plt.show()
```


    
![png](optical_systems_files/optical_systems_11_0.png)
    


Now lets see how the amplitudes and phases look.


```python
# Get the Wavefront
wf = optics.propagate(wavelengths, offset, weights, return_wf=True)

# First we examine the wavefront object
print(wf)
```

    Wavefront(
      wavelength=f32[10],
      pixel_scale=f32[10],
      amplitude=f32[10,192,192],
      phase=f32[10,192,192],
      plane='Focal',
      units='Angular'
    )


Interesting, as we can see the returned `Wavefront` object in _vectorised_ down its first axis. This is one of the benfits of working within the `Equinox`/`Zodiax` framework, as we can vectorise our objects _directly_ meaning we dont need to updack values into arrays to be vectorised. 


```python
# Get the mean amplitude and phase
amplitude = wf.amplitude.mean(0)
phase = wf.phase.mean(0)

# Get the fov for plotting
fov = dlu.rad2arcsec(wf.diameter.mean(0))
extent = [-fov/2, fov/2, -fov/2, fov/2]

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Mean Amplitude')
plt.imshow(amplitude, extent=extent)
plt.colorbar()
plt.xlabel('x (arcseconds)')
plt.ylabel('y (arcseconds)')

plt.subplot(1, 2, 2)
plt.title('Mean Phase')
plt.imshow(phase, extent=extent)
plt.colorbar()
plt.xlabel('x (arcseconds)')
plt.ylabel('y (arcseconds)')

plt.tight_layout()
plt.show()
```


    
![png](optical_systems_files/optical_systems_15_0.png)
    


We can also return the `PSF` object too, allowing us to keep track of the pixel scale and perform operations like downsampling. Lets have a look at that now


```python
# Get the PSF object
PSF = optics.propagate(wavelengths, offset, weights, return_psf=True)

# Downsample the PSF to the 'true' pixel scale
true_PSF = PSF.downsample(oversample)

# Lets examine it, and plot it
print(true_PSF)

# Plot
plt.figure(figsize=(5, 4))
plt.title('True PSF')
plt.imshow(true_PSF.data**0.5)
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.show()
```

    PSF(data=f32[64,64], pixel_scale=f32[])



    
![png](optical_systems_files/optical_systems_17_1.png)
    


## `model`

`model` is the other core function of optical systems. It is designed to be a simple interface between optical systems and `Source` objects. It has the following signature: `optics.model(source, return_wf=False, return_psf=False)`


- `source` is any dLux `Source` object
- `return_wf` is a boolean flag that determines whether the `Wavefront` object should be returned, as opposed to the psf array.
- `return_psf` is a boolean flag that determines whether the `PSF` object should be returned, as opposed to the psf array.

Lets see how to ues it, although we will not look at the `return_wf` and `return_psf` flags as they behave identically to the above example.


```python
# Create a simple point-source object
source = dl.PointSource(wavelengths=wavelengths, weights=weights)

# Propagate it through the optics
psf = optics.model(source)

# Plot
plt.figure(figsize=(5, 4))
plt.title('Sqrt PSF')
plt.imshow(psf**0.5)
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.show()
```


    
![png](optical_systems_files/optical_systems_19_0.png)
    


## Summary

Thats all there is to it! These objects are designed to be simple to use, and to be as flexible as possible.


```python

```
