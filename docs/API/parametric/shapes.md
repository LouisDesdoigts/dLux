# Shapes

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_shapes_Shape["Shape"]
    class dLux_parametric_shapes_InvertibleShape["InvertibleShape"]
    class dLux_parametric_shapes_Soft["Soft"]
    class dLux_parametric_shapes_Circle["Circle"]
    class dLux_parametric_shapes_Square["Square"]
    class dLux_parametric_shapes_Rectangle["Rectangle"]
    class dLux_parametric_shapes_RegularPolygon["RegularPolygon"]
    class dLux_parametric_shapes_Spider["Spider"]
    class dLux_parametric_shapes_Complement["Complement"]
    class dLux_parametric_shapes_TransformedShape["TransformedShape"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    class zodiax_base_Base["Base"]
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_shapes_Shape
    click dLux_parametric_shapes_Shape href "#dLux.parametric.shapes.Shape" "Properties: extent"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_InvertibleShape
    click dLux_parametric_shapes_InvertibleShape href "#dLux.parametric.shapes.InvertibleShape" "Attributes: edge, invert · Methods: evaluate(), evaluate_hard(), evaluate_soft()"
    zodiax_base_Base <|-- dLux_parametric_shapes_Soft
    click dLux_parametric_shapes_Soft href "#dLux.parametric.shapes.Soft" "Attributes: pixels · Methods: clip()"
    dLux_parametric_shapes_InvertibleShape <|-- dLux_parametric_shapes_Circle
    click dLux_parametric_shapes_Circle href "#dLux.parametric.shapes.Circle" "Attributes: diameter · Properties: extent · Methods: evaluate_hard(), evaluate_soft()"
    dLux_parametric_shapes_InvertibleShape <|-- dLux_parametric_shapes_Square
    click dLux_parametric_shapes_Square href "#dLux.parametric.shapes.Square" "Attributes: width · Properties: extent · Methods: evaluate_hard(), evaluate_soft()"
    dLux_parametric_shapes_InvertibleShape <|-- dLux_parametric_shapes_Rectangle
    click dLux_parametric_shapes_Rectangle href "#dLux.parametric.shapes.Rectangle" "Attributes: width, height · Properties: extent · Methods: evaluate_hard(), evaluate_soft()"
    dLux_parametric_shapes_InvertibleShape <|-- dLux_parametric_shapes_RegularPolygon
    click dLux_parametric_shapes_RegularPolygon href "#dLux.parametric.shapes.RegularPolygon" "Attributes: diameter, nsides · Properties: extent · Methods: evaluate_hard(), evaluate_soft()"
    dLux_parametric_shapes_InvertibleShape <|-- dLux_parametric_shapes_Spider
    click dLux_parametric_shapes_Spider href "#dLux.parametric.shapes.Spider" "Attributes: width, angles · Methods: evaluate_hard(), evaluate_soft()"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_Complement
    click dLux_parametric_shapes_Complement href "#dLux.parametric.shapes.Complement" "Attributes: shape · Properties: extent · Methods: evaluate()"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_TransformedShape
    click dLux_parametric_shapes_TransformedShape href "#dLux.parametric.shapes.TransformedShape" "Attributes: shape, transformation · Properties: extent · Methods: evaluate()"
```

???+ info "Shape"
    ::: dLux.parametric.shapes.Shape

???+ info "InvertibleShape"
    ::: dLux.parametric.shapes.InvertibleShape

???+ info "Soft"
    ::: dLux.parametric.shapes.Soft

???+ info "Circle"
    ::: dLux.parametric.shapes.Circle

???+ info "Square"
    ::: dLux.parametric.shapes.Square

???+ info "Rectangle"
    ::: dLux.parametric.shapes.Rectangle

???+ info "RegularPolygon"
    ::: dLux.parametric.shapes.RegularPolygon

???+ info "Spider"
    ::: dLux.parametric.shapes.Spider

???+ info "Complement"
    ::: dLux.parametric.shapes.Complement

???+ info "TransformedShape"
    ::: dLux.parametric.shapes.TransformedShape
