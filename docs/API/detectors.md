# Detectors

The `detectors.py` script contains the general `DetectorLayer` classes. The main class is `DetectorLayer` which is the base class for all other detector layers. Unless you are creating a new optical layer, you will not need to use this class directly. If you do, please refer to the "Creating a Layer" tutorial.

---

## Apply Pixel Response

This layer takes in an array of per-pixel response values and multiplies the psf by these values. Input arrays must be the same size as the psf.

??? info "Apply Pixel Response API"
    :::dLux.detectors.ApplyPixelResponse

---

## Apply Jitter

This layer takes in an jitter value in pixel units and applies a convolves the psf with a 2d gaussian of that size.

??? info "Apply Jitter API"
    :::dLux.detectors.ApplyJitter

---

## Apply Saturation

This layer takes in an saturation value and applies a simply threshold to the psf, it does not model any charge bleeding effects.

??? info "Apply Saturation API"
    :::dLux.detectors.ApplySaturation

---

## Add Constant

This layer takes in an constant value and adds it to the psf, representing the mean of the background noise.

??? info "Add Constant API"
    :::dLux.detectors.AddConstant

---

## Integer Downsample

This layer takes in an integer downsampling factor and downsamples the psf by that factor.

??? info "Integer Downsample API"
    :::dLux.detectors.IntegerDownsample

---

## Rotate

This layer takes in an angle in radians and rotates the psf by that angle using interpolation.

??? info "Rotate API"
    :::dLux.detectors.Rotate