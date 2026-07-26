# Polynomials

## Inheritance

```mermaid
classDiagram
    Base <|-- DynamicZernike
    CoordBasis <|-- DynamicZernikeBasis
    ExplicitBasis <|-- ExplicitPolynomial
    ExplicitBasis <|-- ZernikeBasis
    ParametricBasis <|-- Polynomial
    Polynomial <|-- CoordinatePolynomial
    _ZernikeBasis <|-- DynamicZernikeBasis
    _ZernikeBasis <|-- ZernikeBasis
```

???+ info "DynamicZernike"
    ::: dLux.parametric.polynomials.DynamicZernike

???+ info "ZernikeBasis"
    ::: dLux.parametric.polynomials.ZernikeBasis

???+ info "DynamicZernikeBasis"
    ::: dLux.parametric.polynomials.DynamicZernikeBasis

???+ info "Polynomial"
    ::: dLux.parametric.polynomials.Polynomial

???+ info "ExplicitPolynomial"
    ::: dLux.parametric.polynomials.ExplicitPolynomial

???+ info "CoordinatePolynomial"
    ::: dLux.parametric.polynomials.CoordinatePolynomial
