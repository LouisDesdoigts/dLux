# Optical Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_optical_layers_BaseLayer["BaseLayer"]
    class dLux_layers_optical_layers_BaseOpticalLayer["BaseOpticalLayer"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    class dLux_layers_optical_layers_TransmissiveLayer["TransmissiveLayer"]
    class dLux_layers_optical_layers_AberratedLayer["AberratedLayer"]
    class dLux_layers_optical_layers_Optic["Optic"]
    class dLux_layers_optical_layers_Tilt["Tilt"]
    class dLux_parametric_parametrics_ParametricHolder["ParametricHolder"]
    dLux_parametric_parametrics_ParametricHolder <|-- dLux_layers_optical_layers_BaseLayer
    click dLux_layers_optical_layers_BaseLayer href "#dLux.layers.optical_layers.BaseLayer" "Methods: apply()"
    dLux_layers_optical_layers_BaseLayer <|-- dLux_layers_optical_layers_BaseOpticalLayer
    click dLux_layers_optical_layers_BaseOpticalLayer href "#dLux.layers.optical_layers.BaseOpticalLayer" "No direct public attributes or methods"
    dLux_layers_optical_layers_BaseOpticalLayer <|-- dLux_layers_optical_layers_OpticalLayer
    click dLux_layers_optical_layers_OpticalLayer href "#dLux.layers.optical_layers.OpticalLayer" "Methods: context()"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_TransmissiveLayer
    click dLux_layers_optical_layers_TransmissiveLayer href "#dLux.layers.optical_layers.TransmissiveLayer" "Attributes: transmission, normalise"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_AberratedLayer
    click dLux_layers_optical_layers_AberratedLayer href "#dLux.layers.optical_layers.AberratedLayer" "Attributes: opd, phase"
    dLux_layers_optical_layers_TransmissiveLayer <|-- dLux_layers_optical_layers_Optic
    dLux_layers_optical_layers_AberratedLayer <|-- dLux_layers_optical_layers_Optic
    click dLux_layers_optical_layers_Optic href "#dLux.layers.optical_layers.Optic" "Attributes: transmission, opd, phase, normalise · Methods: phasor()"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_Tilt
    click dLux_layers_optical_layers_Tilt href "#dLux.layers.optical_layers.Tilt" "Attributes: angles, unit"
```

???+ info "BaseLayer"
    ::: dLux.layers.optical_layers.BaseLayer

???+ info "BaseOpticalLayer"
    ::: dLux.layers.optical_layers.BaseOpticalLayer

???+ info "OpticalLayer"
    ::: dLux.layers.optical_layers.OpticalLayer

???+ info "TransmissiveLayer"
    ::: dLux.layers.optical_layers.TransmissiveLayer

???+ info "AberratedLayer"
    ::: dLux.layers.optical_layers.AberratedLayer

???+ info "Optic"
    ::: dLux.layers.optical_layers.Optic

???+ info "Tilt"
    ::: dLux.layers.optical_layers.Tilt
