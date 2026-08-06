# Prebuilt

## Inheritance

```mermaid
classDiagram
    class dLux_prebuilt_SimpleCircular["SimpleCircular"]
    class dLux_prebuilt_SegmentedHex["SegmentedHex"]
    class dLux_prebuilt_NRMLike["NRMLike"]
    class dLux_prebuilt_HSTLike["HSTLike"]
    class dLux_prebuilt_JWSTLike["JWSTLike"]
    class dLux_prebuilt_JWSTNRMLike["JWSTNRMLike"]
    class dLux_prebuilt_EuclidLike["EuclidLike"]
    class dLux_builders_ApertureBuilder["ApertureBuilder"]
    class dLux_builders_SparseApertureBuilder["SparseApertureBuilder"]
    dLux_builders_ApertureBuilder <|-- dLux_prebuilt_SimpleCircular
    click dLux_prebuilt_SimpleCircular href "#dLux.prebuilt.SimpleCircular" "No direct public attributes or methods"
    dLux_builders_SparseApertureBuilder <|-- dLux_prebuilt_SegmentedHex
    click dLux_prebuilt_SegmentedHex href "#dLux.prebuilt.SegmentedHex" "No direct public attributes or methods"
    dLux_builders_SparseApertureBuilder <|-- dLux_prebuilt_NRMLike
    click dLux_prebuilt_NRMLike href "#dLux.prebuilt.NRMLike" "No direct public attributes or methods"
    dLux_prebuilt_SimpleCircular <|-- dLux_prebuilt_HSTLike
    click dLux_prebuilt_HSTLike href "#dLux.prebuilt.HSTLike" "No direct public attributes or methods"
    dLux_prebuilt_SegmentedHex <|-- dLux_prebuilt_JWSTLike
    click dLux_prebuilt_JWSTLike href "#dLux.prebuilt.JWSTLike" "No direct public attributes or methods"
    dLux_prebuilt_NRMLike <|-- dLux_prebuilt_JWSTNRMLike
    click dLux_prebuilt_JWSTNRMLike href "#dLux.prebuilt.JWSTNRMLike" "No direct public attributes or methods"
    dLux_builders_ApertureBuilder <|-- dLux_prebuilt_EuclidLike
    click dLux_prebuilt_EuclidLike href "#dLux.prebuilt.EuclidLike" "No direct public attributes or methods"
```

???+ info "SimpleCircular"
    ::: dLux.prebuilt.SimpleCircular

???+ info "SegmentedHex"
    ::: dLux.prebuilt.SegmentedHex

???+ info "NRMLike"
    ::: dLux.prebuilt.NRMLike

???+ info "HSTLike"
    ::: dLux.prebuilt.HSTLike

???+ info "JWSTLike"
    ::: dLux.prebuilt.JWSTLike

???+ info "JWSTNRMLike"
    ::: dLux.prebuilt.JWSTNRMLike

???+ info "EuclidLike"
    ::: dLux.prebuilt.EuclidLike
