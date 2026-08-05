# Polarised Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_polarised_layers_PolarisationLayer["PolarisationLayer"]
    class dLux_layers_polarised_layers_PolarisingOptic["PolarisingOptic"]
    class dLux_layers_polarised_layers_UniformPolarisingOptic["UniformPolarisingOptic"]
    class dLux_layers_polarised_layers_LinearPolariser["LinearPolariser"]
    class dLux_layers_polarised_layers_Retarder["Retarder"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    class dLux_layers_polarised_layers_BasePolarisingOptic["BasePolarisingOptic"]
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_polarised_layers_PolarisationLayer
    click dLux_layers_polarised_layers_PolarisationLayer href "#dLux.layers.polarised_layers.PolarisationLayer" "Attributes: polarisation"
    dLux_layers_polarised_layers_BasePolarisingOptic <|-- dLux_layers_polarised_layers_PolarisingOptic
    click dLux_layers_polarised_layers_PolarisingOptic href "#dLux.layers.polarised_layers.PolarisingOptic" "Attributes: jones"
    dLux_layers_polarised_layers_PolarisingOptic <|-- dLux_layers_polarised_layers_UniformPolarisingOptic
    click dLux_layers_polarised_layers_UniformPolarisingOptic href "#dLux.layers.polarised_layers.UniformPolarisingOptic" "Attributes: jones, orientation"
    dLux_layers_polarised_layers_BasePolarisingOptic <|-- dLux_layers_polarised_layers_LinearPolariser
    click dLux_layers_polarised_layers_LinearPolariser href "#dLux.layers.polarised_layers.LinearPolariser" "Attributes: angle · Properties: jones"
    dLux_layers_polarised_layers_BasePolarisingOptic <|-- dLux_layers_polarised_layers_Retarder
    click dLux_layers_polarised_layers_Retarder href "#dLux.layers.polarised_layers.Retarder" "Attributes: retardance, angle · Properties: jones"
```

???+ info "PolarisationLayer"
    ::: dLux.layers.polarised_layers.PolarisationLayer

???+ info "PolarisingOptic"
    ::: dLux.layers.polarised_layers.PolarisingOptic

???+ info "UniformPolarisingOptic"
    ::: dLux.layers.polarised_layers.UniformPolarisingOptic

???+ info "LinearPolariser"
    ::: dLux.layers.polarised_layers.LinearPolariser

???+ info "Retarder"
    ::: dLux.layers.polarised_layers.Retarder
