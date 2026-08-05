# Dynamic Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_dynamic_layers_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_dynamic_layers_DynamicTransmissiveLayer["DynamicTransmissiveLayer"]
    class dLux_layers_dynamic_layers_DynamicAberratedLayer["DynamicAberratedLayer"]
    class dLux_layers_dynamic_layers_DynamicOptic["DynamicOptic"]
    class dLux_layers_optical_layers_AberratedLayer["AberratedLayer"]
    class dLux_layers_optical_layers_BaseOpticalLayer["BaseOpticalLayer"]
    class dLux_layers_optical_layers_Optic["Optic"]
    class dLux_layers_optical_layers_TransmissiveLayer["TransmissiveLayer"]
    dLux_layers_optical_layers_BaseOpticalLayer <|-- dLux_layers_dynamic_layers_BaseDynamicLayer
    click dLux_layers_dynamic_layers_BaseDynamicLayer href "#dLux.layers.dynamic_layers.BaseDynamicLayer" "Attributes: coordinates, transformation · Methods: context()"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicTransmissiveLayer
    dLux_layers_optical_layers_TransmissiveLayer <|-- dLux_layers_dynamic_layers_DynamicTransmissiveLayer
    click dLux_layers_dynamic_layers_DynamicTransmissiveLayer href "#dLux.layers.dynamic_layers.DynamicTransmissiveLayer" "Attributes: coordinates, transformation, transmission, normalise"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicAberratedLayer
    dLux_layers_optical_layers_AberratedLayer <|-- dLux_layers_dynamic_layers_DynamicAberratedLayer
    click dLux_layers_dynamic_layers_DynamicAberratedLayer href "#dLux.layers.dynamic_layers.DynamicAberratedLayer" "Attributes: coordinates, transformation, opd, phase"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicOptic
    dLux_layers_optical_layers_Optic <|-- dLux_layers_dynamic_layers_DynamicOptic
    click dLux_layers_dynamic_layers_DynamicOptic href "#dLux.layers.dynamic_layers.DynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise"
```

???+ info "BaseDynamicLayer"
    ::: dLux.layers.dynamic_layers.BaseDynamicLayer

???+ info "DynamicTransmissiveLayer"
    ::: dLux.layers.dynamic_layers.DynamicTransmissiveLayer

???+ info "DynamicAberratedLayer"
    ::: dLux.layers.dynamic_layers.DynamicAberratedLayer

???+ info "DynamicOptic"
    ::: dLux.layers.dynamic_layers.DynamicOptic
