# Grids

## Inheritance

```mermaid
classDiagram
    class dLux_grids_GridSpec["GridSpec"]
    class dLux_grids_ResizeSpec["ResizeSpec"]
    class dLux_grids_CoordTransform["CoordTransform"]
    class dLux_grids_Affine["Affine"]
    class dLux_grids_AffineMap["AffineMap"]
    class dLux_grids_TransformChain["TransformChain"]
    class dLux_grids_DistortCoords["DistortCoords"]
    class zodiax_base_Base["Base"]
    class dLux_grids_BaseGridSpec["BaseGridSpec"]
    dLux_grids_BaseGridSpec <|-- dLux_grids_GridSpec
    click dLux_grids_GridSpec href "#dLux.grids.GridSpec" "Attributes: n, d, c, unit · Methods: ndim(), broadcast(), shape(), scale(), axes(), axes_for(), coordinates(), coordinates_for(), xs(), xs_for(), fov(), extent()"
    dLux_grids_BaseGridSpec <|-- dLux_grids_ResizeSpec
    click dLux_grids_ResizeSpec href "#dLux.grids.ResizeSpec" "Attributes: n, pad_factor, crop_factor, c · Methods: broadcast(), explicit(), padding(), output_size(), crop_size(), pad(), crop(), resize()"
    zodiax_base_Base <|-- dLux_grids_CoordTransform
    click dLux_grids_CoordTransform href "#dLux.grids.CoordTransform" "Methods: get_coordinates(), apply()"
    dLux_grids_CoordTransform <|-- dLux_grids_Affine
    click dLux_grids_Affine href "#dLux.grids.Affine" "Attributes: translation, rotation, scale, shear, order · Methods: coefficients()"
    dLux_grids_CoordTransform <|-- dLux_grids_AffineMap
    click dLux_grids_AffineMap href "#dLux.grids.AffineMap" "Attributes: matrix, offset"
    dLux_grids_CoordTransform <|-- dLux_grids_TransformChain
    click dLux_grids_TransformChain href "#dLux.grids.TransformChain" "Attributes: transformations"
    dLux_grids_CoordTransform <|-- dLux_grids_DistortCoords
    click dLux_grids_DistortCoords href "#dLux.grids.DistortCoords" "Attributes: powers, distortion, shift_invariant"
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
