# Grids

## Inheritance

```mermaid
classDiagram
    Base <|-- CoordTransform
    BaseGridSpec <|-- GridSpec
    BaseGridSpec <|-- ResizeSpec
    CoordTransform <|-- Affine
    CoordTransform <|-- AffineMap
    CoordTransform <|-- DistortCoords
    CoordTransform <|-- TransformChain
```

???+ info "GridSpec"
    ::: dLux.grids.GridSpec

???+ info "ResizeSpec"
    ::: dLux.grids.ResizeSpec

???+ info "CoordTransform"
    ::: dLux.grids.CoordTransform

???+ info "Affine"
    ::: dLux.grids.Affine

???+ info "AffineMap"
    ::: dLux.grids.AffineMap

???+ info "TransformChain"
    ::: dLux.grids.TransformChain

???+ info "DistortCoords"
    ::: dLux.grids.DistortCoords
