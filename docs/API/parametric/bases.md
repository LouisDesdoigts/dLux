# Bases

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_bases_ParametricBasis["ParametricBasis"]
    class dLux_parametric_bases_ExplicitBasis["ExplicitBasis"]
    class dLux_parametric_bases_ImplicitBasis["ImplicitBasis"]
    class dLux_parametric_bases_CoordBasis["CoordBasis"]
    class dLux_parametric_bases_CLIMBBasis["CLIMBBasis"]
    class dLux_parametric_bases_FourierBasis["FourierBasis"]
    class dLux_parametric_bases_SplineBasis["SplineBasis"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_bases_ParametricBasis
    click dLux_parametric_bases_ParametricBasis href "#dLux.parametric.bases.ParametricBasis" "Attributes: coefficients, basis_shape · Methods: coeffs(), c(), alpha(), coefficient_shape(), evaluate_basis(), solve_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_bases_ExplicitBasis
    click dLux_parametric_bases_ExplicitBasis href "#dLux.parametric.bases.ExplicitBasis" "Attributes: coefficients, basis_shape, basis · Methods: evaluate(), solve_basis()"
    dLux_parametric_bases_ParametricBasis <|-- dLux_parametric_bases_ImplicitBasis
    click dLux_parametric_bases_ImplicitBasis href "#dLux.parametric.bases.ImplicitBasis" "Attributes: coefficients, basis_shape · Methods: calculate_basis(), evaluate(), solve_basis()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_CoordBasis
    click dLux_parametric_bases_CoordBasis href "#dLux.parametric.bases.CoordBasis" "Attributes: coefficients, basis_shape · Methods: get_coordinates()"
    dLux_parametric_bases_ExplicitBasis <|-- dLux_parametric_bases_CLIMBBasis
    click dLux_parametric_bases_CLIMBBasis href "#dLux.parametric.bases.CLIMBBasis" "Attributes: coefficients, basis_shape, basis, values, oversample · Methods: evaluate_latent(), evaluate()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_FourierBasis
    click dLux_parametric_bases_FourierBasis href "#dLux.parametric.bases.FourierBasis" "Attributes: coefficients, basis_shape, kernels · Methods: calculate_basis(), evaluate(), resize()"
    dLux_parametric_bases_ImplicitBasis <|-- dLux_parametric_bases_SplineBasis
    click dLux_parametric_bases_SplineBasis href "#dLux.parametric.bases.SplineBasis" "Attributes: coefficients, basis_shape, knot_coords, sample_coords, method · Methods: calculate_basis(), evaluate()"
```

???+ info "ParametricBasis"
    ::: dLux.parametric.bases.ParametricBasis

???+ info "ExplicitBasis"
    ::: dLux.parametric.bases.ExplicitBasis

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
