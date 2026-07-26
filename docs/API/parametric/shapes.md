# Shapes

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_shapes_Shape["Shape"]
    class dLux_parametric_shapes_SoftShape["SoftShape"]
    class dLux_parametric_shapes_RadialShape["RadialShape"]
    class dLux_parametric_shapes_Circle["Circle"]
    class dLux_parametric_shapes_Square["Square"]
    class dLux_parametric_shapes_Rectangle["Rectangle"]
    class dLux_parametric_shapes_RegularPolygon["RegularPolygon"]
    class dLux_parametric_shapes_Spider["Spider"]
    class dLux_parametric_shapes_Complement["Complement"]
    class dLux_parametric_shapes_TransformedShape["TransformedShape"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_shapes_Shape
    click dLux_parametric_shapes_Shape href "#dLux.parametric.shapes.Shape" "Methods: extent()"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_SoftShape
    click dLux_parametric_shapes_SoftShape href "#dLux.parametric.shapes.SoftShape" "Attributes: softening · Methods: clip()"
    dLux_parametric_shapes_SoftShape <|-- dLux_parametric_shapes_RadialShape
    click dLux_parametric_shapes_RadialShape href "#dLux.parametric.shapes.RadialShape" "Attributes: softening, diameter · Methods: extent()"
    dLux_parametric_shapes_RadialShape <|-- dLux_parametric_shapes_Circle
    click dLux_parametric_shapes_Circle href "#dLux.parametric.shapes.Circle" "Attributes: softening, diameter · Methods: evaluate()"
    dLux_parametric_shapes_SoftShape <|-- dLux_parametric_shapes_Square
    click dLux_parametric_shapes_Square href "#dLux.parametric.shapes.Square" "Attributes: softening, width · Methods: extent(), evaluate()"
    dLux_parametric_shapes_SoftShape <|-- dLux_parametric_shapes_Rectangle
    click dLux_parametric_shapes_Rectangle href "#dLux.parametric.shapes.Rectangle" "Attributes: softening, width, height · Methods: extent(), evaluate()"
    dLux_parametric_shapes_RadialShape <|-- dLux_parametric_shapes_RegularPolygon
    click dLux_parametric_shapes_RegularPolygon href "#dLux.parametric.shapes.RegularPolygon" "Attributes: softening, diameter, nsides · Methods: evaluate()"
    dLux_parametric_shapes_SoftShape <|-- dLux_parametric_shapes_Spider
    click dLux_parametric_shapes_Spider href "#dLux.parametric.shapes.Spider" "Attributes: softening, width, angles · Methods: evaluate()"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_Complement
    click dLux_parametric_shapes_Complement href "#dLux.parametric.shapes.Complement" "Attributes: shape · Methods: extent(), evaluate()"
    dLux_parametric_shapes_Shape <|-- dLux_parametric_shapes_TransformedShape
    click dLux_parametric_shapes_TransformedShape href "#dLux.parametric.shapes.TransformedShape" "Attributes: shape, transformation · Methods: extent(), evaluate()"
```

???+ info "Shape"
    ::: dLux.parametric.shapes.Shape

???+ info "SoftShape"
    ::: dLux.parametric.shapes.SoftShape

???+ info "RadialShape"
    ::: dLux.parametric.shapes.RadialShape

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
