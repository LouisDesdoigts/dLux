# Images: `images.py`

This module contains the classes that define the behaviour of images in ∂Lux.

There is only a single public class:

- `Image`

This class has two attributes:

- `image`
- `pixel_scale`

The `image` attribute is a 2D array of the image. The `pixel_scale` is the pixel scale of the image, the units of which depends on the `Optics` used.

??? info "Image API"
    ::: dLux.images.Image

The `Image` class has a number of methods that allow for image manipulation. They are implemented as standard arithmetic operators, i.e.:

```python
import jax.numpy as np
import dLux as dl

image = dl.Image(np.ones((256, 256)), 0.1)
image *= 2
image /= 2
image += 1
image -= 1
```

Here `*` and `/` scale the pixel values by the given factor, and `+` and `-` add or subtract the given value from all pixel values.

On top of these operators, the `Image` class also implements a number of methods that allow for the manipulation of the image.
For more details on these, see the `Image API` above. These methods are:

### `Downsample`

Downsamples the image by a factor of $n$. This is done by summing the image pixels in $n \times n$ blocks.

### `Convolve`

Convolves the image with another. This is done using the `jax.scipy.signal.convolve` function.

### `Rotate(angle)`

Rotates the image by a given angle. This is done using interpolation methods.
