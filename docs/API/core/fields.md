# Fields

## Inheritance

```mermaid
classDiagram
    class dLux_fields_BaseField["BaseField"]
    class dLux_fields_ContinuousField["ContinuousField"]
    class dLux_fields_DiscreteField["DiscreteField"]
    class dLux_fields_Wavefront["Wavefront"]
    class dLux_fields_PolarisedWavefront["PolarisedWavefront"]
    class dLux_fields_PSF["PSF"]
    class dLux_fields_Image["Image"]
    class zodiax_base_Base["Base"]
    zodiax_base_Base <|-- dLux_fields_BaseField
    click dLux_fields_BaseField href "#dLux.fields.BaseField" "Attributes: spec · Methods: field(), set_field(), spatial_shape(), axes(), coordinates(), xs(), npixels(), pixel_scale(), center(), diameter(), normalise(), convolve(), resize(), downsample(), flip()"
    dLux_fields_BaseField <|-- dLux_fields_ContinuousField
    click dLux_fields_ContinuousField href "#dLux.fields.ContinuousField" "Attributes: spec · Methods: scale_to(), interpolate(), rotate()"
    dLux_fields_BaseField <|-- dLux_fields_DiscreteField
    click dLux_fields_DiscreteField href "#dLux.fields.DiscreteField" "Attributes: spec, variance, read_noise · Methods: error(), fourier_transform(), amplitude_spectrum(), power_spectrum(), add_poisson_noise(), add_read_noise(), log_likelihood()"
    dLux_fields_ContinuousField <|-- dLux_fields_Wavefront
    click dLux_fields_Wavefront href "#dLux.fields.Wavefront" "Attributes: spec, phasor, wavelength · Methods: real(), imaginary(), amplitude(), phase(), complex(), polar(), psf(), wavenumber(), batch_ndim(), is_chromatic(), is_polarised(), power(), add_phase(), add_opd(), tilt(), normalise(), apply_jones(), psf_from_stokes()"
    dLux_fields_Wavefront <|-- dLux_fields_PolarisedWavefront
    click dLux_fields_PolarisedWavefront href "#dLux.fields.PolarisedWavefront" "Attributes: spec, phasor, wavelength · Methods: is_polarised(), batch_ndim(), from_wavefront(), psf(), psf_from_stokes(), stokes(), apply_jones()"
    dLux_fields_ContinuousField <|-- dLux_fields_PSF
    click dLux_fields_PSF href "#dLux.fields.PSF" "Attributes: data, spec · Methods: batch_ndim()"
    dLux_fields_DiscreteField <|-- dLux_fields_Image
    click dLux_fields_Image href "#dLux.fields.Image" "Attributes: data, spec, variance, read_noise"
```

???+ info "BaseField"
    ::: dLux.fields.BaseField

???+ info "ContinuousField"
    ::: dLux.fields.ContinuousField

???+ info "DiscreteField"
    ::: dLux.fields.DiscreteField

???+ info "Wavefront"
    ::: dLux.fields.Wavefront

???+ info "PolarisedWavefront"
    ::: dLux.fields.PolarisedWavefront

???+ info "PSF"
    ::: dLux.fields.PSF

???+ info "Image"
    ::: dLux.fields.Image
