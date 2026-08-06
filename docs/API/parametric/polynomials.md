# Polynomials

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_polynomials_DynamicZernike["DynamicZernike"]
    class dLux_parametric_polynomials_ZernikeBasis["ZernikeBasis"]
    class dLux_parametric_polynomials_DynamicZernikeBasis["DynamicZernikeBasis"]
    class dLux_parametric_polynomials_Polynomial["Polynomial"]
    class dLux_parametric_polynomials_ExplicitPolynomial["ExplicitPolynomial"]
    class dLux_parametric_polynomials_CoordinatePolynomial["CoordinatePolynomial"]
    class dLux_parametric_bases_Basis["Basis"]
    class dLux_parametric_bases_CoordBasis["CoordBasis"]
    class dLux_parametric_bases_ParametricBasis["ParametricBasis"]
    class dLux_parametric_polynomials__ZernikeBasis["_ZernikeBasis"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_parametric_polynomials_DynamicZernike
    click dLux_parametric_polynomials_DynamicZernike href "#dLux.parametric.polynomials.DynamicZernike" "Attributes: j, n, m, name, _c, _k · Methods: calculate()"
    dLux_parametric_polynomials__ZernikeBasis <|-- dLux_parametric_polynomials_ZernikeBasis
    dLux_parametric_bases_Basis <|-- dLux_parametric_polynomials_ZernikeBasis
    click dLux_parametric_polynomials_ZernikeBasis href "#dLux.parametric.polynomials.ZernikeBasis" "Attributes: coefficients, shape, basis"
    dLux_parametric_polynomials__ZernikeBasis <|-- dLux_parametric_polynomials_DynamicZernikeBasis
    dLux_parametric_bases_CoordBasis <|-- dLux_parametric_polynomials_DynamicZernikeBasis
    click dLux_parametric_polynomials_DynamicZernikeBasis href "#dLux.parametric.polynomials.DynamicZernikeBasis" "Attributes: coefficients, shape, zernikes, nsides, diameter · Methods: calculate_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_polynomials_Polynomial
    click dLux_parametric_polynomials_Polynomial href "#dLux.parametric.polynomials.Polynomial" "Attributes: coefficients, shape, powers · Methods: calculate_basis(), evaluate(), solve_basis()"
    dLux_parametric_bases_Basis <|-- dLux_parametric_polynomials_ExplicitPolynomial
    click dLux_parametric_polynomials_ExplicitPolynomial href "#dLux.parametric.polynomials.ExplicitPolynomial" "Attributes: coefficients, shape, basis, powers"
    dLux_parametric_polynomials_Polynomial <|-- dLux_parametric_polynomials_CoordinatePolynomial
    click dLux_parametric_polynomials_CoordinatePolynomial href "#dLux.parametric.polynomials.CoordinatePolynomial" "Attributes: coefficients, shape, powers, ndim · Methods: calculate_basis()"
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
