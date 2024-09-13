# Working with dLux Objects

This tutorial is designed to give users a quick overview of how to work with dLux objects. Build using `Equinox` and `Zodiax`, dLux objects intuitive and simple to work with, so let's have a look at how to get started.


```python
# Basic imports
import jax.numpy as np
import jax.random as jr

# dLux imports
import dLux as dl
import dLux.utils as dlu
```

## An Optical System

First we set up a dLux object to work with, in this case a simple optical system. We will not cover the details of how to build an optical system here, as it is covered elsewhere in the tutorials.


```python
# Define our wavefront properties
wf_npix = 512  # Number of pixels in the wavefront
diameter = 1.0  # Diameter of the wavefront, meters

# Construct a simple circular aperture
coords = dlu.pixel_coords(wf_npix, diameter)
aperture = dlu.circle(coords, 0.5 * diameter)

# Zernike aberrations
indices = np.array([2, 3, 7, 8, 9, 10])
basis = 1e-9 * dlu.zernike_basis(indices, coords, diameter=diameter)
coefficients = 50 * jr.normal(jr.PRNGKey(0), indices.shape)

# Define our detector properties
psf_npix = 64  # Number of pixels in the PSF
psf_pixel_scale = 50e-3  # 50 milli-arcseconds
oversample = 3  # Oversampling factor for the PSF

# Define the optical layers
# Note here we can pass in a tuple of (key, layer) pairs to be able to
# access the layer from the optics object with the key!
layers = [
    (
        "aperture",
        dl.layers.BasisOptic(
            transmission=aperture,
            basis=basis,
            coefficients=coefficients,
            normalise=True,
        ),
    ),
    dl.layers.Tilt(np.zeros(2)),
    
]

# Construct the optics object
optics = dl.AngularOpticalSystem(
    wf_npix, diameter, layers, psf_npix, psf_pixel_scale, oversample
)


# Let's examine the optics object! The dLux framework has in-built
# pretty-printing, so we can just print the object to see what it contains.
print(optics)
```

    AngularOpticalSystem(
      wf_npixels=512,
      diameter=1.0,
      layers={
        'aperture':
        BasisOptic(
          basis=f32[6,512,512],
          coefficients=f32[6],
          as_phase=False,
          transmission=f32[512,512],
          normalise=True
        ),
        'Tilt':
        Tilt(angles=f32[2])
      },
      psf_npixels=64,
      oversample=3,
      psf_pixel_scale=0.05
    )


## Paths

So now that we have our optical system set up and we can see the layout, let's have a look at how to work with them. Being built in `Zodiax`, dLux gains access to a 'path-based' interface, greatly simplifying how we work with these objects. 

A path in `Zodiax` works very similarly to a path in a file system. It is a way of navigating through the object, and accessing the data we want via strings, joined with a dot ('.'). Here are some example paths for our optical system:

- `'diameter'`
- `'layers.Tilt.angles'`
- `'layers.aperture.transmission'`

dLux also makes extensive use of the `__getattr__` methods, which allows for the raising of low-level attributes to the top level object. Primarily this means we can skip the `'layers'` part of these paths, so the above paths become:

- `'diameter'`
- `'Tilt.angles'`
- `'aperture.transmission'`

Now to access these values at these paths, we can either use the `.get(path)` method, or just access the via the regular attribute accessors. Let's have a look at this in practice:


```python
# Using the regular accessors
print("Diameter: ", optics.diameter)
print("Angles: ", optics.Tilt.angles)
print("Transmission", optics.aperture.transmission)
```

    Diameter:  1.0
    Angles:  [0. 0.]
    Transmission [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]



```python
# Using the .get method
print("Diameter: ", optics.get('diameter'))
print("Angles: ", optics.get('Tilt.angles'))
print("Transmission", optics.get('aperture.transmission'))
```

    Diameter:  1.0
    Angles:  [0. 0.]
    Transmission [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


So, if we can access the attributes with regular accessors, what is the point of the `Zodiax` `.get` method? Well the `.get` method lets us access _multiple_ attributes at once by passing in a _list_ of paths. Let's have a look at this in practice:


```python
print(optics.get(['diameter', 'Tilt.angles', 'aperture.transmission']))
```

    [1.0, Array([0., 0.], dtype=float32), Array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)]


These paths can actually be simplified _further_, since dLux also raises attributes from the _values_ of the layers dictionary, so the above paths become:

- `'diameter'`
- `'transmission'`
- `'angles'`

Let's look at this in practice.

**NOTE**

While this level parameter raising can greatly simplify the paths we work with, we need to cognisant that each path is _unique_. For example if we have two layers that have the same values `as_phase`, then using `'as_phase'` as our path will only return _one_ of these values. To distinguish between these two we would need to reference the layer by its dictionary key, ie `'layer1.as_phase'` or `'layer2.as_phase'`.


```python
print(optics.get(['diameter', 'angles', 'transmission']))

```

    [1.0, Array([0., 0.], dtype=float32), Array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)]


## Zodiax Methods

`Zodiax` gives us access to a series of class methods that are designed to mirror the `jax.Array` syntax, ie `.at`, `.set`, `.multiply` etc. The syntax a slightly different in that we need to specify a path to the data we want to work with, but the functionality is the same. Here are the main `Zodiax` methods:
some of the zodiax methods:

- `.get(paths)`
- `.set(paths, values)`
- `.add(paths, values)`
- `.multiply(paths, values)`
- `.divide(paths, values)`
- `.min(paths, values)`
- `.max(paths, values)`

Let's use the `.add` method to see how we can modify our optical system.


```python
paths = ['diameter', 'angles']
new_optics = optics.add(paths, [1, 0.5])
print(new_optics.get(paths))
```

    [2.0, Array([0.5, 0.5], dtype=float32)]


## Nesting

Zodiax goes further here, and allows to 'nest' paths, such that we can operate on _mulitple_ values in the same operation. Let's look at some examples of this in practice:


```python
# Operate on multiple values simultaneously
paths = ['diameter', 'angles']
new_optics = optics.multiply(paths, 0)
print(new_optics.get(paths))
```

    [0.0, Array([0., 0.], dtype=float32)]


We can also nest _within_ our path itself, let's see how:


```python
# Set nested values simulatenously
# Note that 'paths' here has two entries, so we need to supply a 
# list of values of the same length
paths = [['diameter', 'angles'], 'transmission']
values = [1, 2]
new_optics = optics.set(paths, values)
print(new_optics.get(paths))
```

    [1, 1, 2]


Excellent! Now some keen readers may have noticed that by using the `.set` method, the `transmission` values have changed from an array to a float! This is becuase there is no _robust_ way to do runtime type and shape checking, plus sometimes we may want to change the type anyway. Do be cognisant when setting values that you are setting then to valid types for the object.

**Summary**

So that is a quick overview of how to work with dLux objects. We have seen how to access the data via paths, and how to modify the data using the `Zodiax` methods. We have also seen how to nest paths, and how to nest paths within paths. Hopefully this has given you a good overview of how to work with dLux objects, and you can now go and work with the objects with ease!

[comment]: <> (TODO: Add jit to this tutorial?)

[comment]: <> (TODO: Add object vectorsation?)

[comment]: <> (TODO: Learn how to spell "vectorsation"?)


