# An Overview of the Apertures

## Introduction

`dLux` implements a number of aperture components for telescopes. Because `dLux` is powered by autodiff, the shape of the aperture can be learned. Theoretically, you could learn the value of every pixel in the aperture. Learning by pixel would be computationally expensive and the model could chase noise making the results meaningless. Instead the apertures that we have implemented are **parametrised**. In general the apertures can be, *translated*, *sheared, compressed androtated*, all in a differentiable manner by softening the hard bounary using a sigmoid.

There are four different aperture types: Dynamic, Static, Aberrated and Composite. The dynamic apertures are the most flexible and can be used to learn the shape of the aperture. The static apertures pre-compute the aperture and use the fixed array representation. The aberrated apertures are used to learn the shape of the aperture and the basis functions for phase retrieval. The compound apertures are used to combine multiple apertures into a single aperture. There are also spider apertures that are used to model secondary mirror supports.

## Dynamic Apertures

The dynamic apertures are form the basis for the rest of the apertures and contains 7 classes: `CircularAperture`, `AnnularAperture`, `HexagonalAperture`, `RectangularAperture`, `SquareAperture`, `RegularPolygonAperture`, `IrregularPolygonAperture`. Each of these classes has a seriers of common parameters: `translation`, `rotation`, `shear`, `compression`, `softening` and `occulting`. The `translation` and `rotation` parameters are used to move and rotate the aperture. The `shear` and `compression` parameters are used to change the shape of the aperture. The `softening` parameter is used to soften the hard boundary of the aperture, and `occulting` controls if the aperture is transmissive or occulting.

Each of these classes then has a different parameterisation of the aperture itself, for example the `CircularAperture` has a `radius` parameter, the `AnnularAperture` has `inner_radius` and `outer_radius` parameters.

??? info "Circular Aperture API"
    :::dLux.apertures.CircularAperture

??? info "Annular Aperture API"
    :::dLux.apertures.AnnularAperture

??? info "Hexagonal Aperture API"
    :::dLux.apertures.HexagonalAperture

??? info "Rectangular Aperture API"
    :::dLux.apertures.RectangularAperture

??? info "Square Aperture API"
    :::dLux.apertures.SquareAperture

??? info "Regular Polygonal Aperture API"
    :::dLux.apertures.RegularPolygonalAperture

??? info "Irregular Polygonal Aperture API"
    :::dLux.apertures.IrregularPolygonalAperture

## Static Apertures

The inbuild flexibility of the `dLux.apertures` module is very powerful but unlikely to be needed in most cases. For this reason Dynamic apertures can be turned into static apertures where the array of tranmission values are calculated once, and then kept fixed to avoid re-calculation of the aperture every evaluation.

??? info "Static Aperture API"
    :::dLux.apertures.StaticAperture

## Aberrated Apertures

Both dynamic and static apertures can have aberations applied to them using the `AberratedAperture` class. This class takes an aperture as an argument and then applies a set of basis vectors to the aperture. The basis vectors are derived from the *Zernike* polynomials and calculated to be orthonormal on all regular-polygon apertures. The underlying aberrations are generated in the aberations module.

??? info "Aberrated Aperture API"
    :::dLux.apertures.AberratedAperture

??? info "Static Aberrated Aperture API"
    :::dLux.apertures.StaticAberratedAperture

## Composite Apertures

The Composite apertures are designed to take in a series of dynamic aperture and combine them to create arbirary aperture shapes. There are two types of composite aperture, Compound and Multi apertures. The CompoundAperture is used to combine apertures that are overlapping, and the MultiAperture is used to combine apertures that are not overlapping. For example if we wanted to create a HST-like aperture we would combine an annular aperture with a spiders class. If we wanted to create an aperture mask, we would combine a series of circular apertures in a MultiAperture class.

??? info "Compound Aperture API"
    :::dLux.apertures.CompoundAperture

??? info "Multi Aperture API"
    :::dLux.apertures.MultiAperture

## Spiders

The spiders class are just a specific parametrisation of rectangular apertures for simplicity.

??? info "Uniform Spider API"
    :::dLux.apertures.UniformSpider

# Usage and Examples

Now let's write some code. We can create a basic circular aperture at the centre of the coordinate system with a 1m radius like so:

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
        plt.imshow(apertures[i].get_aperture(256, 2))
    plt.tight_layout()
    plt.savefig("assets/apertures.png")
    ```

![apertures](../assets/apertures.png)

## Aperture Factory

Most users will not need to use the dynamic apertures, so the `ApertureFactory` class is designed to provide a simple interface to generate the most common apertures. It is able to construct hard-edged circular or regular poygonal apertures. Secondary mirrors obscurations with the same aperture shape can be constructed, along with uniformly spaced struts. Aberrations can also be applied to the aperture. The ratio of the primary aperture opening to the array size is determined by the aperture_ratio parameter, with secondary mirror obscurations and struts being scaled relative to the aperture diameter.

??? info "Aperture Factory API"
    :::dLux.apertures.ApertureFactory

Lets look at an example of how to construct a simple circular aperture with a secondary mirror obscurtion held by 4 struts and some low-order aberrations. For this example lets take a 2m diameter aperutre, with a 20cm secondary mirror held by 3 struts with a width of 2cm. In this example the secondary mirror is 10% of the primary aperture diameter and the struts are 1% of the primary aperture diameter, giving us values of 0.1 and 0.01 for the secondary_ratio and strut_ratio parameters. Let calcualte this for a 512x512 array with the aperture spanning the full array.

```python
import dLux as dl
from jax import numpy as np, random as jr

# Construct Zernikes
zernikes = np.arange(4, 11)
coefficients = jr.normal(jr.PRNGKey(0), (zernikes.shape[0],))

# Construct aperture
aperture = dl.ApertureFactory(
    npixels         = 512,
    secondary_ratio = 0.1, 
    nstruts         = 4, 
    strut_ratio     = 0.01, 
    zernikes        = zernikes, 
    coefficients    = coefficients,
    name            = 'Aperture')

print(aperture)
```

```python
> StaticAberratedAperture(
>   name='Aperture',
>   aperture=f32[512,512],
>   coefficients=f32[7],
>   basis=f32[7,512,512]
> )
```

As we can see the resulting aperture class has three parameters, `aperture`, `basis` and `coefficients`. Lets have a look at the resulting aperture and aberrations.

??? abstract "Plotting code"
    ```python
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(aperture.aperture)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(aperture.opd)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("assets/aperture_factory.png")
    ```

![aperture_factory](../assets/aperture_factory.png)
