# Refractive

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_refractive_CauchyIndex["CauchyIndex"]
    class dLux_parametric_refractive_PolynomialIndex["PolynomialIndex"]
    class dLux_parametric_refractive_InterpolatedIndex["InterpolatedIndex"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_refractive_CauchyIndex
    click dLux_parametric_refractive_CauchyIndex href "#dLux.parametric.refractive.CauchyIndex" "Attributes: coeffs, scale · Properties: coefficients · Methods: evaluate()"
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_refractive_PolynomialIndex
    click dLux_parametric_refractive_PolynomialIndex href "#dLux.parametric.refractive.PolynomialIndex" "Attributes: coeffs, scale · Properties: coefficients · Methods: evaluate()"
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_refractive_InterpolatedIndex
    click dLux_parametric_refractive_InterpolatedIndex href "#dLux.parametric.refractive.InterpolatedIndex" "Attributes: wavelengths, indices, method, extrapolate · Methods: evaluate()"
```

???+ info "CauchyIndex"
    ::: dLux.parametric.refractive.CauchyIndex

???+ info "PolynomialIndex"
    ::: dLux.parametric.refractive.PolynomialIndex

???+ info "InterpolatedIndex"
    ::: dLux.parametric.refractive.InterpolatedIndex
