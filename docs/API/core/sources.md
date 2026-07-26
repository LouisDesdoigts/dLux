# Sources

## Inheritance

```mermaid
classDiagram
    class dLux_sources_BaseSource["BaseSource"]
    class dLux_sources_Spectrum["Spectrum"]
    class dLux_sources_Source["Source"]
    class dLux_sources_BinarySource["BinarySource"]
    class dLux_parametric_parametrics_ParametricHolder["ParametricHolder"]
    dLux_parametric_parametrics_ParametricHolder <|-- dLux_sources_BaseSource
    click dLux_sources_BaseSource href "#dLux.sources.BaseSource" "Attributes: flux, distribution, units · Methods: source_params(), distribution_params(), model()"
    dLux_parametric_parametrics_ParametricHolder <|-- dLux_sources_Spectrum
    click dLux_sources_Spectrum href "#dLux.sources.Spectrum" "Attributes: wavelengths, weights, units · Methods: spectrum_params(), model()"
    dLux_sources_BaseSource <|-- dLux_sources_Source
    dLux_sources_Spectrum <|-- dLux_sources_Source
    click dLux_sources_Source href "#dLux.sources.Source" "Attributes: wavelengths, weights, position, flux, distribution, units · Methods: params()"
    dLux_sources_BaseSource <|-- dLux_sources_BinarySource
    dLux_sources_Spectrum <|-- dLux_sources_BinarySource
    click dLux_sources_BinarySource href "#dLux.sources.BinarySource" "Attributes: wavelengths, weights, centre, separation, position_angle, contrast, flux, distribution, units · Methods: params()"
```

???+ info "BaseSource"
    ::: dLux.sources.BaseSource

???+ info "Spectrum"
    ::: dLux.sources.Spectrum

???+ info "Source"
    ::: dLux.sources.Source

???+ info "BinarySource"
    ::: dLux.sources.BinarySource
