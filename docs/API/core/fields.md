# Fields

## Inheritance

```mermaid
classDiagram
    Base <|-- BaseField
    BaseField <|-- ContinuousField
    BaseField <|-- DiscreteField
    ContinuousField <|-- PSF
    ContinuousField <|-- Wavefront
    DiscreteField <|-- Image
    Wavefront <|-- PolarisedWavefront
```

???+ info "BaseField"
    ::: dLux.fields.BaseField

???+ info "ContinuousField"
    ::: dLux.fields.ContinuousField

???+ info "DiscreteField"
    ::: dLux.fields.DiscreteField

???+ info "Wavefront"
    ::: dLux.fields.Wavefront

???+ info "PolarisedWavefront"
    ::: dLux.fields.PolarisedWavefront

???+ info "PSF"
    ::: dLux.fields.PSF

???+ info "Image"
    ::: dLux.fields.Image
