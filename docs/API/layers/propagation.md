# Propagation

## Inheritance

```mermaid
classDiagram
    class dLux_layers_propagation_ABCDElement["ABCDElement"]
    class dLux_layers_propagation_ABCDFreeSpace["ABCDFreeSpace"]
    class dLux_layers_propagation_ABCDLens["ABCDLens"]
    class dLux_layers_propagation_ABCDMirror["ABCDMirror"]
    class dLux_layers_propagation_ABCDFraunhofer["ABCDFraunhofer"]
    class dLux_layers_propagation_Propagator["Propagator"]
    class dLux_layers_propagation_FocalPropagator["FocalPropagator"]
    class dLux_layers_propagation_ABCDPropagator["ABCDPropagator"]
    class dLux_layers_propagation_FreeSpace["FreeSpace"]
    class dLux_layers_propagation_Fraunhofer["Fraunhofer"]
    class dLux_layers_propagation_Fresnel["Fresnel"]
    class dLux_layers_optical_OpticalLayer["OpticalLayer"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_layers_propagation_ABCDElement
    click dLux_layers_propagation_ABCDElement href "#dLux.layers.propagation.ABCDElement" "No direct public attributes or methods"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDFreeSpace
    click dLux_layers_propagation_ABCDFreeSpace href "#dLux.layers.propagation.ABCDFreeSpace" "Attributes: distance · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDLens
    click dLux_layers_propagation_ABCDLens href "#dLux.layers.propagation.ABCDLens" "Attributes: focal_length · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDMirror
    click dLux_layers_propagation_ABCDMirror href "#dLux.layers.propagation.ABCDMirror" "Attributes: radius · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDFraunhofer
    click dLux_layers_propagation_ABCDFraunhofer href "#dLux.layers.propagation.ABCDFraunhofer" "Attributes: focal_length · Properties: abcd"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_propagation_Propagator
    click dLux_layers_propagation_Propagator href "#dLux.layers.propagation.Propagator" "Attributes: spec · Methods: apply(), validate()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_FocalPropagator
    click dLux_layers_propagation_FocalPropagator href "#dLux.layers.propagation.FocalPropagator" "Attributes: spec, focal_length, inverse · Methods: validate()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_ABCDPropagator
    click dLux_layers_propagation_ABCDPropagator href "#dLux.layers.propagation.ABCDPropagator" "Attributes: spec, ABCDs, method · Properties: abcd · Methods: validate()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_FreeSpace
    click dLux_layers_propagation_FreeSpace href "#dLux.layers.propagation.FreeSpace" "Attributes: spec, distance, crop"
    dLux_layers_propagation_FocalPropagator <|-- dLux_layers_propagation_Fraunhofer
    click dLux_layers_propagation_Fraunhofer href "#dLux.layers.propagation.Fraunhofer" "Attributes: spec, focal_length, inverse, method"
    dLux_layers_propagation_FocalPropagator <|-- dLux_layers_propagation_Fresnel
    click dLux_layers_propagation_Fresnel href "#dLux.layers.propagation.Fresnel" "Attributes: spec, focal_length, inverse, defocus, method"
```

???+ info "ABCDElement"
    ::: dLux.layers.propagation.ABCDElement

???+ info "ABCDFreeSpace"
    ::: dLux.layers.propagation.ABCDFreeSpace

???+ info "ABCDLens"
    ::: dLux.layers.propagation.ABCDLens

???+ info "ABCDMirror"
    ::: dLux.layers.propagation.ABCDMirror

???+ info "ABCDFraunhofer"
    ::: dLux.layers.propagation.ABCDFraunhofer

???+ info "Propagator"
    ::: dLux.layers.propagation.Propagator

???+ info "FocalPropagator"
    ::: dLux.layers.propagation.FocalPropagator

???+ info "ABCDPropagator"
    ::: dLux.layers.propagation.ABCDPropagator

???+ info "FreeSpace"
    ::: dLux.layers.propagation.FreeSpace

???+ info "Fraunhofer"
    ::: dLux.layers.propagation.Fraunhofer

???+ info "Fresnel"
    ::: dLux.layers.propagation.Fresnel
