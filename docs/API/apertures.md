# Aperture Layers: `apertures.py`

This module contains the classes that define the behaviour of aperture layers in ∂Lux.

These classes provide a simple set of classes used to perform basic transformations of wavefronts.

There are nine public classes:

- `CircularAperture`
- `RectangularAperture`
- `RegPolyAperture`
- `IrregPolyAperture`
- `AberratedAperture`
- `UniformSpider`
- `CompoundAperture`
- `MultiAperture`

Plus also one public method, `ApertureFactory`, that allows for the easy construction of various simple apertures.

# API

??? info "Circular Aperture API"
    :::dLux.apertures.CircularAperture

??? info "Rectangular Aperture API"
    :::dLux.apertures.RectangularAperture

??? info "Regular Polygonal Aperture API"
    :::dLux.apertures.RegPolyAperture

??? info "Irregular Polygonal Aperture API"
    :::dLux.apertures.IrregPolyAperture

??? info "Aberrated Aperture API"
    :::dLux.apertures.AberratedAperture

??? info "Compound Aperture API"
    :::dLux.apertures.CompoundAperture

??? info "Multi Aperture API"
    :::dLux.apertures.MultiAperture

??? info "Aperture Factory API"
    :::dLux.apertures.ApertureFactory

These classes are broken into three categories: _Dynamic_ Apertures, _Aberrated_ Apertures, and _Composite_ Apertures.

## Dynamic Apertures

The Dynamic apertures are defined by physical units (such as radius) and are generated at run-time on the input coordinates. These apertures all have a set of input transformation parameters that can be used to modify the shape of the aperture. These parameters are:

- `centre` The $(x, y)$ coordinates of the centre of the aperture.
- `shear` The $(x, y)$ linear shear of the aperture.
- `compression` The $(x, y)$ compression of the aperture.
- `rotation` The clockwise rotation of the aperture.
- `occulting` Is the aperture occulting (`True`) or transmissive (`False`).
- `softening` The approximate pixel width of the soft boundary applied to the
    aperture. Hard edges can be achieved by setting the softening to zero.

## Aberrated Apertures

The Aberrated apertures are defined by a set of Zernike coefficients. These apertures and coefficients are generated at run-time on the input coordinates. These classes contain a Dynamic aperture, plus a dynamic Zernike basis and coefficients.

## Composite Apertures

The Composite apertures are defined by a set of sub-apertures that are combined. These apertures are generated at run-time on the input coordinates. These classes contain a list of Dynamic apertures that are combined. `CompoundApertures` are apertures that are combined via a multiplication (i.e., combining a primary mirror and secondary mirror with spiders), whereas `MultiApertures` are apertures that are combined via an addition (i.e., combining different holes of an aperture mask).

Most users will not need to use these dynamic features, so all of these classes have a `.make_static(npixels, diameter)` method that returns a static version of the aperture which is faster to use. These static versions are implemented as `Optic` classes, and can be used in the same way as any other `Optic` class.

The primary interface for these classes is the `.transmission(npixels, diameter)` method that returns the transmission of the aperture on a set of coordinates defined by the number of pixels and diameter.

---

# Examples

Let's look at how to apply some simple transformations to circular apertures.

```python
import dLux as dl

apertures = [
    dl.CircularAperture(1.),
    dl.CircularAperture(1., centre=[.5, .5]),
    dl.CircularAperture(1., shear=[.05, .05]),
    dl.CircularAperture(1., compression=[1.05, .95]),
    dl.CircularAperture(1., softening=20),
    dl.CircularAperture(1., occulting=True)
]
```

??? abstract "Plotting code"
    ```python
    import matplotlib.pyplot as plt

    plt.figure(figsize=(30, 4))
    for i in range(len(apertures)):
        plt.subplot(1, 6, i+1)
        plt.imshow(apertures[i].transmission(256, 2))
    plt.tight_layout()
    plt.savefig("assets/basic_apertures.png")
    ```
![basic_apertures](../assets/basic_apertures.png)

We can esaily add aberrations to these, and make them static:

```python
import dLux as dl
import jax.numpy as np
import jax.random as jr

# Construct a Hexagonal Aperture
hex = dl.RegPolyAperture(6, 1.)

# Turn it into an aberrated aperture
zernikes = np.arange(1, 7)
coefficients = jr.normal(jr.PRNGKey(0), (6,))
aberrated_hex = dl.AberratedAperture(hex, zernikes, coefficients)

# Promote it to static
static_hex = aberrated_hex.make_static(256, 2)
```

??? abstract "Plotting code"
    ```python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Transmission")
    plt.imshow(static_hex.transmission)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("OPD")
    plt.imshow(static_hex.transmission * static_hex.opd)
    plt.colorbar()
    plt.savefig('assets/aberrated_apertures.png')
    ```
![aberrated_apertures](../assets/aberrated_apertures.png)

We can also use the `ApertureFactory` class to construct a simple aperture:

```python
import dLux as dl
import jax.numpy as np
import jax.random as jr

# Construct Zernikes
radial_orders = [2, 3]
coefficients = jr.normal(jr.PRNGKey(0), (7,))

# Construct aperture
aperture = dl.ApertureFactory(
    npixels         = 512,
    secondary_ratio = 0.1, 
    nstruts         = 4, 
    strut_ratio     = 0.01, 
    radial_orders   = radial_orders, 
    coefficients    = coefficients)
```

??? abstract "Plotting code"
    ```python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Transmission")
    plt.imshow(aperture.transmission)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    opd = aperture.opd.at[aperture.transmission == 0].set(np.nan)
    plt.title("OPD")
    plt.imshow(opd)
    plt.colorbar()
    plt.savefig('assets/aperture_factory.png')
    ```
![aperture_factory](../assets/aperture_factory.png)