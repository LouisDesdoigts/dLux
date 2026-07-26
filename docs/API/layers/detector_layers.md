# Detector Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_detector_layers_BaseDetectorLayer["BaseDetectorLayer"]
    class dLux_layers_detector_layers_DetectorLayer["DetectorLayer"]
    class dLux_layers_detector_layers_ApplyPixelResponse["ApplyPixelResponse"]
    class dLux_layers_detector_layers_ApplyJitter["ApplyJitter"]
    class dLux_layers_detector_layers_ApplySaturation["ApplySaturation"]
    class dLux_layers_detector_layers_AddConstant["AddConstant"]
    class dLux_layers_optical_layers_BaseLayer["BaseLayer"]
    dLux_layers_optical_layers_BaseLayer <|-- dLux_layers_detector_layers_BaseDetectorLayer
    click dLux_layers_detector_layers_BaseDetectorLayer href "#dLux.layers.detector_layers.BaseDetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_layers_BaseDetectorLayer <|-- dLux_layers_detector_layers_DetectorLayer
    click dLux_layers_detector_layers_DetectorLayer href "#dLux.layers.detector_layers.DetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplyPixelResponse
    click dLux_layers_detector_layers_ApplyPixelResponse href "#dLux.layers.detector_layers.ApplyPixelResponse" "Attributes: pixel_response"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplyJitter
    click dLux_layers_detector_layers_ApplyJitter href "#dLux.layers.detector_layers.ApplyJitter" "Attributes: sigma, kernel_size, oversample · Methods: kernel()"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplySaturation
    click dLux_layers_detector_layers_ApplySaturation href "#dLux.layers.detector_layers.ApplySaturation" "Attributes: threshold"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_AddConstant
    click dLux_layers_detector_layers_AddConstant href "#dLux.layers.detector_layers.AddConstant" "Attributes: value"
```

???+ info "BaseDetectorLayer"
    ::: dLux.layers.detector_layers.BaseDetectorLayer

???+ info "DetectorLayer"
    ::: dLux.layers.detector_layers.DetectorLayer

???+ info "ApplyPixelResponse"
    ::: dLux.layers.detector_layers.ApplyPixelResponse

???+ info "ApplyJitter"
    ::: dLux.layers.detector_layers.ApplyJitter

???+ info "ApplySaturation"
    ::: dLux.layers.detector_layers.ApplySaturation

???+ info "AddConstant"
    ::: dLux.layers.detector_layers.AddConstant
