# Systems

## Inheritance

```mermaid
classDiagram
    class dLux_systems_LayeredSystem["LayeredSystem"]
    class dLux_systems_OpticalSystem["OpticalSystem"]
    class dLux_systems_DetectorSystem["DetectorSystem"]
    class dLux_base_Base["Base"]
    class dLux_layers_optical_BaseOpticalLayer["BaseOpticalLayer"]
    dLux_base_Base <|-- dLux_systems_LayeredSystem
    click dLux_systems_LayeredSystem href "#dLux.systems.LayeredSystem" "Attributes: layers · Methods: debug(), insert_layer(), remove_layer()"
    dLux_systems_LayeredSystem <|-- dLux_systems_OpticalSystem
    dLux_layers_optical_BaseOpticalLayer <|-- dLux_systems_OpticalSystem
    click dLux_systems_OpticalSystem href "#dLux.systems.OpticalSystem" "Attributes: layers, grid · Methods: apply_mono(), apply(), initialise_wavefront(), propagate_mono(), propagate(), model(), debug_propagate_mono()"
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
