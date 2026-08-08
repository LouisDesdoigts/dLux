# Bases

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_bases_ParametricBasis["ParametricBasis"]
    class dLux_parametric_bases_Basis["Basis"]
    class dLux_parametric_bases_PastedBasis["PastedBasis"]
    class dLux_parametric_bases_ImplicitBasis["ImplicitBasis"]
    class dLux_parametric_bases_CoordBasis["CoordBasis"]
    class dLux_parametric_bases_CLIMBBasis["CLIMBBasis"]
    class dLux_parametric_bases_FourierBasis["FourierBasis"]
    class dLux_parametric_bases_SplineBasis["SplineBasis"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_bases_ParametricBasis
    click dLux_parametric_bases_ParametricBasis href "#dLux.parametric.bases.ParametricBasis" "Attributes: coeffs, shape · Properties: coefficients, c, alpha · Methods: evaluate_basis(), solve_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_bases_Basis
    click dLux_parametric_bases_Basis href "#dLux.parametric.bases.Basis" "Attributes: coeffs, shape, basis · Methods: evaluate(), solve_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_bases_PastedBasis
    click dLux_parametric_bases_PastedBasis href "#dLux.parametric.bases.PastedBasis" "Attributes: coeffs, shape, basis, spec, method · Methods: evaluate(), solve_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_bases_ImplicitBasis
    click dLux_parametric_bases_ImplicitBasis href "#dLux.parametric.bases.ImplicitBasis" "Attributes: coeffs, shape · Methods: calculate_basis(), evaluate(), solve_basis()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_CoordBasis
    click dLux_parametric_bases_CoordBasis href "#dLux.parametric.bases.CoordBasis" "Attributes: coeffs, shape · Methods: get_coordinates()"
    dLux_parametric_bases_Basis <|-- dLux_parametric_bases_CLIMBBasis
    click dLux_parametric_bases_CLIMBBasis href "#dLux.parametric.bases.CLIMBBasis" "Attributes: coeffs, shape, basis, values, oversample · Methods: evaluate_latent(), evaluate()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_FourierBasis
    click dLux_parametric_bases_FourierBasis href "#dLux.parametric.bases.FourierBasis" "Attributes: coeffs, shape, kernels · Methods: calculate_basis(), evaluate(), resize()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_SplineBasis
    click dLux_parametric_bases_SplineBasis href "#dLux.parametric.bases.SplineBasis" "Attributes: coeffs, shape, knot_coords, sample_coords, method · Methods: calculate_basis(), evaluate()"
```

???+ info "ParametricBasis"
    ::: dLux.parametric.bases.ParametricBasis

???+ info "Basis"
    ::: dLux.parametric.bases.Basis

???+ info "PastedBasis"
    ::: dLux.parametric.bases.PastedBasis

???+ info "ImplicitBasis"
    ::: dLux.parametric.bases.ImplicitBasis

???+ info "CoordBasis"
    ::: dLux.parametric.bases.CoordBasis

???+ info "CLIMBBasis"
    ::: dLux.parametric.bases.CLIMBBasis

???+ info "FourierBasis"
    ::: dLux.parametric.bases.FourierBasis

???+ info "SplineBasis"
    ::: dLux.parametric.bases.SplineBasis
