# Parametrics

## Inheritance

```mermaid
classDiagram
    Base <|-- Parametric
    Base <|-- ParametricHolder
    Parametric <|-- Combination
    Parametric <|-- DynamicParametric
    Parametric <|-- Interpolation
    Parametric <|-- Transform
```

???+ info "Parametric"
    ::: dLux.parametric.parametrics.Parametric

???+ info "ParametricHolder"
    ::: dLux.parametric.parametrics.ParametricHolder

???+ info "resolve_parametric"
    ::: dLux.parametric.parametrics.resolve_parametric

???+ info "Transform"
    ::: dLux.parametric.parametrics.Transform

???+ info "Interpolation"
    ::: dLux.parametric.parametrics.Interpolation

???+ info "DynamicParametric"
    ::: dLux.parametric.parametrics.DynamicParametric

???+ info "Combination"
    ::: dLux.parametric.parametrics.Combination
