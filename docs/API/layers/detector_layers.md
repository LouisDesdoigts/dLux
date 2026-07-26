# Detector Layers

## Inheritance

```mermaid
classDiagram
    BaseDetectorLayer <|-- DetectorLayer
    BaseLayer <|-- BaseDetectorLayer
    DetectorLayer <|-- AddConstant
    DetectorLayer <|-- ApplyJitter
    DetectorLayer <|-- ApplyPixelResponse
    DetectorLayer <|-- ApplySaturation
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
