# Sources

## Inheritance

```mermaid
classDiagram
    BaseSource <|-- BinarySource
    BaseSource <|-- Source
    ParametricHolder <|-- BaseSource
    ParametricHolder <|-- Spectrum
    Spectrum <|-- BinarySource
    Spectrum <|-- Source
```

???+ info "BaseSource"
    ::: dLux.sources.BaseSource

???+ info "Spectrum"
    ::: dLux.sources.Spectrum

???+ info "Source"
    ::: dLux.sources.Source

???+ info "BinarySource"
    ::: dLux.sources.BinarySource
