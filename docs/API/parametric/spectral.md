# Spectral

## Inheritance

```mermaid
classDiagram
    class dLux_parametric_spectral_SpectralPolynomial["SpectralPolynomial"]
    class dLux_parametric_spectral_SpectralBasis["SpectralBasis"]
    class dLux_parametric_spectral_Blackbody["Blackbody"]
    class dLux_parametric_bases_Basis["Basis"]
    class dLux_parametric_parametrics_Parametric["Parametric"]
    class dLux_parametric_polynomials_Polynomial["Polynomial"]
    dLux_parametric_polynomials_Polynomial <|-- dLux_parametric_spectral_SpectralPolynomial
    click dLux_parametric_spectral_SpectralPolynomial href "#dLux.parametric.spectral.SpectralPolynomial" "Attributes: normalise · Methods: evaluate()"
    dLux_parametric_bases_Basis <|-- dLux_parametric_spectral_SpectralBasis
    click dLux_parametric_spectral_SpectralBasis href "#dLux.parametric.spectral.SpectralBasis" "Attributes: normalise · Methods: evaluate()"
    dLux_parametric_parametrics_Parametric <|-- dLux_parametric_spectral_Blackbody
    click dLux_parametric_spectral_Blackbody href "#dLux.parametric.spectral.Blackbody" "Attributes: temperature, normalise · Methods: evaluate()"
```

???+ info "SpectralPolynomial"
    ::: dLux.parametric.spectral.SpectralPolynomial

???+ info "SpectralBasis"
    ::: dLux.parametric.spectral.SpectralBasis

???+ info "Blackbody"
    ::: dLux.parametric.spectral.Blackbody
