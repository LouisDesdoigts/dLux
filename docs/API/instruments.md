# Instruments: instruments.py

This module contains the classes that define the behaviour of instruments in dLux. Instruments classes hold the various different types of dLux classes and handles the coherent modelling of these different classes.

There is one public class:

- `Instrument`

The Instrument class has three main methods:

1. `model()` Models the sources through the instrument optics and detector.
2. `observe()` Calls the `observe` method of the stored observation class.
3. `normalise()` Returns a new instrument with normalised source objects. Note this method is automatically called during the `.model()` method to ensure quantities are conserved during optimisation loops.

??? info "Instrument API"
    ::: dLux.instruments.Instrument

---

# Examples

Lets see how we can construct a minimal instrument object with all of its attributes populated.

```python
import jax.numpy as np
import dLux as dl

# Define the optical parameters
wf_npixels = 256
diameter = 1 # meters
psf_npixels = 128
psf_pixel_scale = 0.1 # arcseconds
psf_oversample = 4

# Use ApertureFactory class to make a simple circular aperture
aperture = dl.ApertureFactory(wf_npixels)

# Construct the optics class
optics = dl.AngularOptics(wf_npixels, diameter, aperture, 
    psf_npixels, psf_pixel_scale, psf_oversample)

# Construct Source
wavelengths = np.linspace(1e-6, 1.2e-6, 5) # meters
source = dl.PointSource(wavelengths)

# Construct Detector
detector = dl.LayeredDetector([dl.ApplyJitter(20)])

# Construct Observation
observation = dl.Dither(np.array([[0, 0], [1e-6, 1e-6]]))

# Construct the instrument and observe
instrument = dl.Instrument(optics, source, detector, observation)
psfs = instrument.observe()
```

??? abstract "Plotting Code"
    ```python
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("$\sqrt{PSF_1}$")
    plt.imshow(psfs[0]**0.5)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("$\sqrt{PSF_2}$")
    plt.imshow(psfs[1]**0.5)
    plt.colorbar()
    plt.savefig('assets/instrument.png')
    ```

![instrument](../assets/instrument.png)

!!! tip "Model" 
    We can also use the `.model()` method to model the instrument without applying the observation class!