# Detector Layers

???+ info "ApplyPixelResponse"
    ::: dLux.layers.detector_layers.ApplyPixelResponse

???+ info "ApplyJitter"
    ::: dLux.layers.detector_layers.ApplyJitter

???+ info "ApplySaturation"
    ::: dLux.layers.detector_layers.ApplySaturation

???+ info "AddConstant"
    ::: dLux.layers.detector_layers.AddConstant

???+ info "Downsample"
    ::: dLux.layers.detector_layers.Downsample

<!-- # Detector Layers: `detector_layers.py`

This module contains the classes that define the behaviour of detector layers in ∂Lux.

There are six public classes:

- `ApplyPixelResponse`
- `ApplyJitter`
- `ApplySaturation`
- `AddConstant`
- `IntegerDownsample`
- `RotateDetector`

These classes operate on `Image` classes. They have one main method: `.__call__(image)`, which takes in a ∂Lux `Image` class and applies the detector layer to it.

These classes are relatively simple, so let's quickly move through them.

### `ApplyPixelResponse`

Applies a pixel response array to the input image via a multiplication.

??? info "ApplyPixelResponse API"
    :::dLux.detector_layers.ApplyPixelResponse

### `ApplyJitter`

Convolves the image with a Gaussian kernel parameterised by the standard deviation (`sigma`).

??? info "ApplyJitter API"
    :::dLux.detector_layers.ApplyJitter

### `ApplySaturation`

Applies a simple saturation model to the input image, by clipping any values above `saturation`.

??? info "ApplySaturation API"
    :::dLux.detector_layers.ApplySaturation

### `AddConstant`

Add a constant to the output image. This is typically used to model the mean value of the detector noise.

??? info "AddConstant API"
    :::dLux.detector_layers.AddConstant

### `IntegerDownsample`

Downsamples an input image by an integer number of pixels via a summation. The number of pixels in the input image must be integer divisible by `kernel_size`.

??? info "IntegerDownsample API"
    :::dLux.detector_layers.IntegerDownsample

### `RotateDetector`

Applies a rotation to the image using interpolation methods.

??? info "RotateDetector API"
    :::dLux.detector_layers.RotateDetector -->
