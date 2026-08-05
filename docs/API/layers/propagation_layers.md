# Propagation Layers

## Inheritance

```mermaid
classDiagram
    class dLux_layers_propagation_layers_ABCDElement["ABCDElement"]
    class dLux_layers_propagation_layers_ABCDFreeSpace["ABCDFreeSpace"]
    class dLux_layers_propagation_layers_ABCDLens["ABCDLens"]
    class dLux_layers_propagation_layers_ABCDMirror["ABCDMirror"]
    class dLux_layers_propagation_layers_ABCDFraunhofer["ABCDFraunhofer"]
    class dLux_layers_propagation_layers_Propagator["Propagator"]
    class dLux_layers_propagation_layers_FocalPropagator["FocalPropagator"]
    class dLux_layers_propagation_layers_ABCDPropagator["ABCDPropagator"]
    class dLux_layers_propagation_layers_FreeSpace["FreeSpace"]
    class dLux_layers_propagation_layers_Fraunhofer["Fraunhofer"]
    class dLux_layers_propagation_layers_Fresnel["Fresnel"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_layers_propagation_layers_ABCDElement
    click dLux_layers_propagation_layers_ABCDElement href "#dLux.layers.propagation_layers.ABCDElement" "No direct public attributes or methods"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDFreeSpace
    click dLux_layers_propagation_layers_ABCDFreeSpace href "#dLux.layers.propagation_layers.ABCDFreeSpace" "Attributes: distance · Properties: abcd"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDLens
    click dLux_layers_propagation_layers_ABCDLens href "#dLux.layers.propagation_layers.ABCDLens" "Attributes: focal_length · Properties: abcd"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDMirror
    click dLux_layers_propagation_layers_ABCDMirror href "#dLux.layers.propagation_layers.ABCDMirror" "Attributes: radius · Properties: abcd"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDFraunhofer
    click dLux_layers_propagation_layers_ABCDFraunhofer href "#dLux.layers.propagation_layers.ABCDFraunhofer" "Attributes: focal_length · Properties: abcd"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_propagation_layers_Propagator
    click dLux_layers_propagation_layers_Propagator href "#dLux.layers.propagation_layers.Propagator" "Attributes: spec · Methods: validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_FocalPropagator
    click dLux_layers_propagation_layers_FocalPropagator href "#dLux.layers.propagation_layers.FocalPropagator" "Attributes: spec, focal_length · Methods: validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_ABCDPropagator
    click dLux_layers_propagation_layers_ABCDPropagator href "#dLux.layers.propagation_layers.ABCDPropagator" "Attributes: spec, ABCDs, method · Properties: abcd · Methods: validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_FreeSpace
    click dLux_layers_propagation_layers_FreeSpace href "#dLux.layers.propagation_layers.FreeSpace" "Attributes: spec, distance, crop"
    dLux_layers_propagation_layers_FocalPropagator <|-- dLux_layers_propagation_layers_Fraunhofer
    click dLux_layers_propagation_layers_Fraunhofer href "#dLux.layers.propagation_layers.Fraunhofer" "Attributes: spec, focal_length, method"
    dLux_layers_propagation_layers_FocalPropagator <|-- dLux_layers_propagation_layers_Fresnel
    click dLux_layers_propagation_layers_Fresnel href "#dLux.layers.propagation_layers.Fresnel" "Attributes: spec, focal_length, defocus, method"
```

???+ info "ABCDElement"
    ::: dLux.layers.propagation_layers.ABCDElement

???+ info "ABCDFreeSpace"
    ::: dLux.layers.propagation_layers.ABCDFreeSpace

???+ info "ABCDLens"
    ::: dLux.layers.propagation_layers.ABCDLens

???+ info "ABCDMirror"
    ::: dLux.layers.propagation_layers.ABCDMirror

???+ info "ABCDFraunhofer"
    ::: dLux.layers.propagation_layers.ABCDFraunhofer

???+ info "Propagator"
    ::: dLux.layers.propagation_layers.Propagator

???+ info "FocalPropagator"
    ::: dLux.layers.propagation_layers.FocalPropagator

???+ info "ABCDPropagator"
    ::: dLux.layers.propagation_layers.ABCDPropagator

???+ info "FreeSpace"
    ::: dLux.layers.propagation_layers.FreeSpace

???+ info "Fraunhofer"
    ::: dLux.layers.propagation_layers.Fraunhofer

???+ info "Fresnel"
    ::: dLux.layers.propagation_layers.Fresnel
