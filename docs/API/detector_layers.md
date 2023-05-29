# Detector Layers: detector_layers.py

This module contains the classes that define the behaviour of DetectorLayers in dLux.

There are six public classes:

- `ApplyPixelResponse`
- `ApplyJitter`
- `ApplySaturation`
- `AddConstant`
- `IntegerDownsample`
- `Rotate`

These classes operate on `Image` classes. They have one main method, `.__call__(image)` that takes in a dLux `Image` class and applies the detector layer to it.

These classes are relatively simple so lets quick fire through them.

### `ApplyPixelResponse`

Applies a pixel response array to the the input image, via a multiplication.

??? info "ApplyPixelResponse API"
    :::dLux.detector_layers.ApplyPixelResponse

### `ApplyJitter`

Convolves the image with a gaussian kernel parameterised by the standard deviation (sigma).

??? info "ApplyJitter API"
    :::dLux.detector_layers.ApplyJitter

### `ApplySaturation`

Applies a simple saturation model to the input image, by clipping any values above saturation, to saturation.

??? info "ApplySaturation API"
    :::dLux.detector_layers.ApplySaturation

### `AddConstant`

Add a constant to the output image. This is typically used to model the mean value of the detector noise.

??? info "AddConstant API"
    :::dLux.detector_layers.AddConstant

### `IntegerDownsample`

Downsamples an input image by an integer number of pixels via a sum. The number of pixels in the input image must by integer divisible by the kernel_size.

??? info "IntegerDownsample API"
    :::dLux.detector_layers.IntegerDownsample

### `Rotate`

Applies a rotation to the image using interpolation methods.

??? info "Rotate API"
    :::dLux.detector_layers.Rotate