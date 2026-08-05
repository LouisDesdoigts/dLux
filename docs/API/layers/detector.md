# Detector

## Inheritance

```mermaid
classDiagram
    class dLux_layers_detector_BaseDetectorLayer["BaseDetectorLayer"]
    class dLux_layers_detector_DetectorLayer["DetectorLayer"]
    class dLux_layers_detector_ApplyPixelResponse["ApplyPixelResponse"]
    class dLux_layers_detector_ApplyJitter["ApplyJitter"]
    class dLux_layers_detector_ApplySaturation["ApplySaturation"]
    class dLux_layers_detector_AddConstant["AddConstant"]
    class dLux_layers_optical_BaseLayer["BaseLayer"]
    dLux_layers_optical_BaseLayer <|-- dLux_layers_detector_BaseDetectorLayer
    click dLux_layers_detector_BaseDetectorLayer href "#dLux.layers.detector.BaseDetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_BaseDetectorLayer <|-- dLux_layers_detector_DetectorLayer
    click dLux_layers_detector_DetectorLayer href "#dLux.layers.detector.DetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplyPixelResponse
    click dLux_layers_detector_ApplyPixelResponse href "#dLux.layers.detector.ApplyPixelResponse" "Attributes: pixel_response"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplyJitter
    click dLux_layers_detector_ApplyJitter href "#dLux.layers.detector.ApplyJitter" "Attributes: sigma, kernel_size, oversample · Properties: kernel"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplySaturation
    click dLux_layers_detector_ApplySaturation href "#dLux.layers.detector.ApplySaturation" "Attributes: threshold"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_AddConstant
    click dLux_layers_detector_AddConstant href "#dLux.layers.detector.AddConstant" "Attributes: value"
```

???+ info "BaseDetectorLayer"
    ::: dLux.layers.detector.BaseDetectorLayer

???+ info "DetectorLayer"
    ::: dLux.layers.detector.DetectorLayer

???+ info "ApplyPixelResponse"
    ::: dLux.layers.detector.ApplyPixelResponse

???+ info "ApplyJitter"
    ::: dLux.layers.detector.ApplyJitter

???+ info "ApplySaturation"
    ::: dLux.layers.detector.ApplySaturation

???+ info "AddConstant"
    ::: dLux.layers.detector.AddConstant
