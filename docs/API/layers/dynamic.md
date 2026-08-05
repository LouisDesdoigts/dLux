# Dynamic

## Inheritance

```mermaid
classDiagram
    class dLux_layers_dynamic_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_dynamic_DynamicTransmissiveLayer["DynamicTransmissiveLayer"]
    class dLux_layers_dynamic_DynamicAberratedLayer["DynamicAberratedLayer"]
    class dLux_layers_dynamic_DynamicOptic["DynamicOptic"]
    class dLux_layers_optical_AberratedLayer["AberratedLayer"]
    class dLux_layers_optical_BaseOpticalLayer["BaseOpticalLayer"]
    class dLux_layers_optical_Optic["Optic"]
    class dLux_layers_optical_TransmissiveLayer["TransmissiveLayer"]
    dLux_layers_optical_BaseOpticalLayer <|-- dLux_layers_dynamic_BaseDynamicLayer
    click dLux_layers_dynamic_BaseDynamicLayer href "#dLux.layers.dynamic.BaseDynamicLayer" "Attributes: coordinates, transformation · Methods: context()"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicTransmissiveLayer
    dLux_layers_optical_TransmissiveLayer <|-- dLux_layers_dynamic_DynamicTransmissiveLayer
    click dLux_layers_dynamic_DynamicTransmissiveLayer href "#dLux.layers.dynamic.DynamicTransmissiveLayer" "Attributes: coordinates, transformation, transmission, normalise"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicAberratedLayer
    dLux_layers_optical_AberratedLayer <|-- dLux_layers_dynamic_DynamicAberratedLayer
    click dLux_layers_dynamic_DynamicAberratedLayer href "#dLux.layers.dynamic.DynamicAberratedLayer" "Attributes: coordinates, transformation, opd, phase"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicOptic
    dLux_layers_optical_Optic <|-- dLux_layers_dynamic_DynamicOptic
    click dLux_layers_dynamic_DynamicOptic href "#dLux.layers.dynamic.DynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise"
```

???+ info "BaseDynamicLayer"
    ::: dLux.layers.dynamic.BaseDynamicLayer

???+ info "DynamicTransmissiveLayer"
    ::: dLux.layers.dynamic.DynamicTransmissiveLayer

???+ info "DynamicAberratedLayer"
    ::: dLux.layers.dynamic.DynamicAberratedLayer

???+ info "DynamicOptic"
    ::: dLux.layers.dynamic.DynamicOptic
