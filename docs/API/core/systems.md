# Systems

## Inheritance

```mermaid
classDiagram
    class dLux_systems_LayeredSystem["LayeredSystem"]
    class dLux_systems_OpticalSystem["OpticalSystem"]
    class dLux_systems_DetectorSystem["DetectorSystem"]
    class dLux_layers_optical_layers_BaseOpticalLayer["BaseOpticalLayer"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_systems_LayeredSystem
    click dLux_systems_LayeredSystem href "#dLux.systems.LayeredSystem" "Attributes: layers · Methods: apply(), debug(), insert_layer(), remove_layer()"
    dLux_systems_LayeredSystem <|-- dLux_systems_OpticalSystem
    dLux_layers_optical_layers_BaseOpticalLayer <|-- dLux_systems_OpticalSystem
    click dLux_systems_OpticalSystem href "#dLux.systems.OpticalSystem" "Attributes: layers, spec · Methods: initialise_wavefront(), propagate_mono(), propagate(), model(), debug_propagate_mono()"
    dLux_systems_LayeredSystem <|-- dLux_systems_DetectorSystem
    click dLux_systems_DetectorSystem href "#dLux.systems.DetectorSystem" "Attributes: layers · Methods: model()"
```

???+ info "LayeredSystem"
    ::: dLux.systems.LayeredSystem

???+ info "OpticalSystem"
    ::: dLux.systems.OpticalSystem

???+ info "DetectorSystem"
    ::: dLux.systems.DetectorSystem

???+ info "Detector"
    ::: dLux.systems.DetectorSystem
