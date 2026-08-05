# Unified Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_unified_layers_UnifiedLayer["UnifiedLayer"]
    class dLux_layers_unified_layers_Resize["Resize"]
    class dLux_layers_unified_layers_Downsample["Downsample"]
    class dLux_layers_unified_layers_Flip["Flip"]
    class dLux_layers_unified_layers_Interpolate["Interpolate"]
    class dLux_layers_unified_layers_Normalise["Normalise"]
    class dLux_layers_unified_layers_Lambda["Lambda"]
    class dLux_layers_detector_layers_DetectorLayer["DetectorLayer"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_unified_layers_UnifiedLayer
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_unified_layers_UnifiedLayer
    click dLux_layers_unified_layers_UnifiedLayer href "#dLux.layers.unified_layers.UnifiedLayer" "No direct public attributes or methods"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Resize
    click dLux_layers_unified_layers_Resize href "#dLux.layers.unified_layers.Resize" "Attributes: npixels"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Downsample
    click dLux_layers_unified_layers_Downsample href "#dLux.layers.unified_layers.Downsample" "Attributes: n"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Flip
    click dLux_layers_unified_layers_Flip href "#dLux.layers.unified_layers.Flip" "Attributes: axes"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Interpolate
    click dLux_layers_unified_layers_Interpolate href "#dLux.layers.unified_layers.Interpolate" "Attributes: transformation, method, complex, fill"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Normalise
    click dLux_layers_unified_layers_Normalise href "#dLux.layers.unified_layers.Normalise" "Attributes: mode, value"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Lambda
    click dLux_layers_unified_layers_Lambda href "#dLux.layers.unified_layers.Lambda" "No direct public attributes or methods"
```

???+ info "UnifiedLayer"
    ::: dLux.layers.unified_layers.UnifiedLayer

???+ info "Resize"
    ::: dLux.layers.unified_layers.Resize

???+ info "Downsample"
    ::: dLux.layers.unified_layers.Downsample

???+ info "Flip"
    ::: dLux.layers.unified_layers.Flip

???+ info "Interpolate"
    ::: dLux.layers.unified_layers.Interpolate

???+ info "Normalise"
    ::: dLux.layers.unified_layers.Normalise

???+ info "Lambda"
    ::: dLux.layers.unified_layers.Lambda
