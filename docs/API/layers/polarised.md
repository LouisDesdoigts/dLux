# Polarised

## Inheritance

```mermaid
classDiagram
    class dLux_layers_polarised_PolarisationLayer["PolarisationLayer"]
    class dLux_layers_polarised_PolarisingOptic["PolarisingOptic"]
    class dLux_layers_polarised_UniformPolarisingOptic["UniformPolarisingOptic"]
    class dLux_layers_polarised_LinearPolariser["LinearPolariser"]
    class dLux_layers_polarised_Retarder["Retarder"]
    class dLux_layers_optical_OpticalLayer["OpticalLayer"]
    class dLux_layers_polarised_BasePolarisingOptic["BasePolarisingOptic"]
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_polarised_PolarisationLayer
    click dLux_layers_polarised_PolarisationLayer href "#dLux.layers.polarised.PolarisationLayer" "Attributes: polarisation"
    dLux_layers_polarised_BasePolarisingOptic <|-- dLux_layers_polarised_PolarisingOptic
    click dLux_layers_polarised_PolarisingOptic href "#dLux.layers.polarised.PolarisingOptic" "Attributes: jones"
    dLux_layers_polarised_PolarisingOptic <|-- dLux_layers_polarised_UniformPolarisingOptic
    click dLux_layers_polarised_UniformPolarisingOptic href "#dLux.layers.polarised.UniformPolarisingOptic" "Attributes: jones, orientation"
    dLux_layers_polarised_BasePolarisingOptic <|-- dLux_layers_polarised_LinearPolariser
    click dLux_layers_polarised_LinearPolariser href "#dLux.layers.polarised.LinearPolariser" "Attributes: angle · Properties: jones"
    dLux_layers_polarised_BasePolarisingOptic <|-- dLux_layers_polarised_Retarder
    click dLux_layers_polarised_Retarder href "#dLux.layers.polarised.Retarder" "Attributes: retardance, angle · Properties: jones"
```

???+ info "PolarisationLayer"
    ::: dLux.layers.polarised.PolarisationLayer

???+ info "PolarisingOptic"
    ::: dLux.layers.polarised.PolarisingOptic

???+ info "UniformPolarisingOptic"
    ::: dLux.layers.polarised.UniformPolarisingOptic

???+ info "LinearPolariser"
    ::: dLux.layers.polarised.LinearPolariser

???+ info "Retarder"
    ::: dLux.layers.polarised.Retarder
