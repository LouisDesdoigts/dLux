# Builders

## Inheritance

```mermaid
classDiagram
    class dLux_builders_GridBuilder["GridBuilder"]
    class dLux_builders_OPDDef["OPDDef"]
    class dLux_builders_ApertureData["ApertureData"]
    class dLux_builders_Norm["Norm"]
    class dLux_builders_ZernikeDef["ZernikeDef"]
    class dLux_builders_ApertureBuilder["ApertureBuilder"]
    class dLux_builders_SparseApertureBuilder["SparseApertureBuilder"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_builders_GridBuilder
    click dLux_builders_GridBuilder href "#dLux.builders.GridBuilder" "Methods: validate(), build()"
    zodiax_base_Base <|-- dLux_builders_OPDDef
    click dLux_builders_OPDDef href "#dLux.builders.OPDDef" "Methods: calculate()"
    zodiax_base_Base <|-- dLux_builders_ApertureData
    click dLux_builders_ApertureData href "#dLux.builders.ApertureData" "Attributes: transmission, support, diameter, centers"
    zodiax_base_Base <|-- dLux_builders_Norm
    click dLux_builders_Norm href "#dLux.builders.Norm" "Attributes: mode, scale"
    dLux_builders_OPDDef <|-- dLux_builders_ZernikeDef
    click dLux_builders_ZernikeDef href "#dLux.builders.ZernikeDef" "Attributes: nolls, oversize, norm · Methods: calculate()"
    dLux_builders_GridBuilder <|-- dLux_builders_ApertureBuilder
    click dLux_builders_ApertureBuilder href "#dLux.builders.ApertureBuilder" "Attributes: primary, obscurations, opd, oversample · Methods: build(), aperture_data()"
    dLux_builders_ApertureBuilder <|-- dLux_builders_SparseApertureBuilder
    click dLux_builders_SparseApertureBuilder href "#dLux.builders.SparseApertureBuilder" "Attributes: centers, global_obscurations · Methods: aperture_data()"
```

???+ info "GridBuilder"
    ::: dLux.builders.GridBuilder

???+ info "OPDDef"
    ::: dLux.builders.OPDDef

???+ info "ApertureData"
    ::: dLux.builders.ApertureData

???+ info "Norm"
    ::: dLux.builders.Norm

???+ info "ZernikeDef"
    ::: dLux.builders.ZernikeDef

???+ info "ApertureBuilder"
    ::: dLux.builders.ApertureBuilder

???+ info "SparseApertureBuilder"
    ::: dLux.builders.SparseApertureBuilder
