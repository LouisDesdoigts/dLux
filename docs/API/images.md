# Images: images.py

This module contains the classes that define the behaviour of Images in dLux.

There is only a Single public class:

- `Image`

This class only has two attributes:

- `image`
- `pixel_scale`

The image attribute is a 2D array of the image. The pixel_scale is the pixel scale of the image, which as units depending on the type of Optics used.

??? info "Image API"
    ::: dLux.images.Image

It has a number of methods that allow for the manipulation of the image. It implements as series of operators that allow for the manipulation of the image via standard arithmetic operators. ie:

```python
import jax.numpy as np
import dLux as dl

image = dl.Image(np.ones((256, 256)), 0.1)
image *= 2
image += 1
image /= 2
```

On top of these, it implements a number of methods that allow for the manipulation of the image, these are:

### `Downsample`

Downsamples the image by a factor of n. This is done by summing the image pixels in n x n blocks.

### `Convolve`

Convolves the image with another image. This is done using the `jax.scipy.signal.convolve` function.

### `Rotate`

Rotates the image by a given angle. This is done using interpolation methods.
