# Core API

This diagram is generated from the public API. Hover over a class for its direct attributes and methods, or select it to open the full reference.

```mermaid
classDiagram
    class dLux_builders_GridBuilder["GridBuilder"]
    class dLux_builders_OPDDef["OPDDef"]
    class dLux_builders_ApertureData["ApertureData"]
    class dLux_builders_Norm["Norm"]
    class dLux_builders_ZernikeDef["ZernikeDef"]
    class dLux_builders_ApertureBuilder["ApertureBuilder"]
    class dLux_builders_SparseApertureBuilder["SparseApertureBuilder"]
    class dLux_fields_BaseField["BaseField"]
    class dLux_fields_ContinuousField["ContinuousField"]
    class dLux_fields_DiscreteField["DiscreteField"]
    class dLux_fields_Wavefront["Wavefront"]
    class dLux_fields_PolarisedWavefront["PolarisedWavefront"]
    class dLux_fields_PSF["PSF"]
    class dLux_fields_Image["Image"]
    class dLux_grids_GridSpec["GridSpec"]
    class dLux_grids_ResizeSpec["ResizeSpec"]
    class dLux_grids_CoordTransform["CoordTransform"]
    class dLux_grids_Affine["Affine"]
    class dLux_grids_AffineMap["AffineMap"]
    class dLux_grids_TransformChain["TransformChain"]
    class dLux_grids_DistortCoords["DistortCoords"]
    class dLux_prebuilt_SimpleCircular["SimpleCircular"]
    class dLux_prebuilt_SegmentedHex["SegmentedHex"]
    class dLux_prebuilt_NRMLike["NRMLike"]
    class dLux_prebuilt_HSTLike["HSTLike"]
    class dLux_prebuilt_JWSTLike["JWSTLike"]
    class dLux_prebuilt_JWSTNRMLike["JWSTNRMLike"]
    class dLux_prebuilt_EuclidLike["EuclidLike"]
    class dLux_sources_BaseSource["BaseSource"]
    class dLux_sources_Spectrum["Spectrum"]
    class dLux_sources_Source["Source"]
    class dLux_sources_BinarySource["BinarySource"]
    class dLux_systems_LayeredSystem["LayeredSystem"]
    class dLux_systems_OpticalSystem["OpticalSystem"]
    class dLux_systems_DetectorSystem["DetectorSystem"]
    click dLux_builders_GridBuilder href "../builders/#dLux.builders.GridBuilder" "Methods: validate(), build()"
    click dLux_builders_OPDDef href "../builders/#dLux.builders.OPDDef" "Methods: calculate()"
    click dLux_builders_ApertureData href "../builders/#dLux.builders.ApertureData" "Attributes: transmission, support, diameter, centers"
    click dLux_builders_Norm href "../builders/#dLux.builders.Norm" "Attributes: mode, scale"
    dLux_builders_OPDDef <|-- dLux_builders_ZernikeDef
    click dLux_builders_ZernikeDef href "../builders/#dLux.builders.ZernikeDef" "Attributes: nolls, oversize, norm · Methods: calculate()"
    dLux_builders_GridBuilder <|-- dLux_builders_ApertureBuilder
    click dLux_builders_ApertureBuilder href "../builders/#dLux.builders.ApertureBuilder" "Attributes: primary, obscurations, opd, oversample · Methods: build(), aperture_data()"
    dLux_builders_ApertureBuilder <|-- dLux_builders_SparseApertureBuilder
    click dLux_builders_SparseApertureBuilder href "../builders/#dLux.builders.SparseApertureBuilder" "Attributes: centers, global_obscurations · Methods: aperture_data()"
    click dLux_fields_BaseField href "../fields/#dLux.fields.BaseField" "Attributes: spec · Properties: field, spatial_shape, axes, coordinates, xs, npixels, pixel_scale, center, diameter · Methods: normalise(), convolve(), resize(), downsample(), flip()"
    dLux_fields_BaseField <|-- dLux_fields_ContinuousField
    click dLux_fields_ContinuousField href "../fields/#dLux.fields.ContinuousField" "Attributes: spec · Methods: scale_to(), interpolate(), rotate()"
    dLux_fields_BaseField <|-- dLux_fields_DiscreteField
    click dLux_fields_DiscreteField href "../fields/#dLux.fields.DiscreteField" "Attributes: spec, variance, read_noise · Properties: error, fourier_transform, amplitude_spectrum, power_spectrum · Methods: add_poisson_noise(), add_read_noise(), log_likelihood()"
    dLux_fields_ContinuousField <|-- dLux_fields_Wavefront
    click dLux_fields_Wavefront href "../fields/#dLux.fields.Wavefront" "Attributes: spec, phasor, wavelength · Properties: field, real, imaginary, amplitude, phase, complex, polar, psf, wavenumber, batch_ndim, is_chromatic, is_polarised, _mapped_axis, power · Methods: from_phasor(), add_phase(), add_opd(), tilt(), normalise(), apply_jones(), psf_from_stokes()"
    dLux_fields_Wavefront <|-- dLux_fields_PolarisedWavefront
    click dLux_fields_PolarisedWavefront href "../fields/#dLux.fields.PolarisedWavefront" "Attributes: spec, phasor, wavelength · Properties: is_polarised, batch_ndim, psf · Methods: from_phasor(), from_wavefront(), psf_from_stokes(), stokes(), apply_jones()"
    dLux_fields_ContinuousField <|-- dLux_fields_PSF
    click dLux_fields_PSF href "../fields/#dLux.fields.PSF" "Attributes: data, spec · Properties: field, batch_ndim · Methods: from_wavefront()"
    dLux_fields_DiscreteField <|-- dLux_fields_Image
    click dLux_fields_Image href "../fields/#dLux.fields.Image" "Attributes: data, spec, variance, read_noise · Properties: field"
    click dLux_grids_GridSpec href "../grids/#dLux.grids.GridSpec" "Attributes: n, d, c, unit · Properties: ndim, shape, scale, axes, xs, coordinates, fov · Methods: broadcast(), resize(), downsample(), oversample(), resample(), build(), axes_for(), xs_for(), transformed(), coordinates_for(), extent()"
    click dLux_grids_ResizeSpec href "../grids/#dLux.grids.ResizeSpec" "Attributes: n, pad, crop, c · Properties: explicit, padding · Methods: broadcast(), output_size(), crop_size(), pad_array(), crop_array(), resize()"
    click dLux_grids_CoordTransform href "../grids/#dLux.grids.CoordTransform" "Methods: get_coordinates(), apply()"
    dLux_grids_CoordTransform <|-- dLux_grids_Affine
    click dLux_grids_Affine href "../grids/#dLux.grids.Affine" "Attributes: translation, rotation, scale, shear, order · Methods: coefficients()"
    dLux_grids_CoordTransform <|-- dLux_grids_AffineMap
    click dLux_grids_AffineMap href "../grids/#dLux.grids.AffineMap" "Attributes: matrix, offset"
    dLux_grids_CoordTransform <|-- dLux_grids_TransformChain
    click dLux_grids_TransformChain href "../grids/#dLux.grids.TransformChain" "Attributes: transformations"
    dLux_grids_CoordTransform <|-- dLux_grids_DistortCoords
    click dLux_grids_DistortCoords href "../grids/#dLux.grids.DistortCoords" "Attributes: powers, distortion, shift_invariant"
    dLux_builders_ApertureBuilder <|-- dLux_prebuilt_SimpleCircular
    click dLux_prebuilt_SimpleCircular href "../prebuilt/#dLux.prebuilt.SimpleCircular" "No direct public attributes or methods"
    dLux_builders_SparseApertureBuilder <|-- dLux_prebuilt_SegmentedHex
    click dLux_prebuilt_SegmentedHex href "../prebuilt/#dLux.prebuilt.SegmentedHex" "No direct public attributes or methods"
    dLux_builders_SparseApertureBuilder <|-- dLux_prebuilt_NRMLike
    click dLux_prebuilt_NRMLike href "../prebuilt/#dLux.prebuilt.NRMLike" "No direct public attributes or methods"
    dLux_prebuilt_SimpleCircular <|-- dLux_prebuilt_HSTLike
    click dLux_prebuilt_HSTLike href "../prebuilt/#dLux.prebuilt.HSTLike" "No direct public attributes or methods"
    dLux_prebuilt_SegmentedHex <|-- dLux_prebuilt_JWSTLike
    click dLux_prebuilt_JWSTLike href "../prebuilt/#dLux.prebuilt.JWSTLike" "No direct public attributes or methods"
    dLux_prebuilt_NRMLike <|-- dLux_prebuilt_JWSTNRMLike
    click dLux_prebuilt_JWSTNRMLike href "../prebuilt/#dLux.prebuilt.JWSTNRMLike" "No direct public attributes or methods"
    dLux_builders_ApertureBuilder <|-- dLux_prebuilt_EuclidLike
    click dLux_prebuilt_EuclidLike href "../prebuilt/#dLux.prebuilt.EuclidLike" "No direct public attributes or methods"
    click dLux_sources_BaseSource href "../sources/#dLux.sources.BaseSource" "Attributes: flux, distribution, units · Methods: source_params(), flux_params(), distribution_params(), wavefront(), model()"
    click dLux_sources_Spectrum href "../sources/#dLux.sources.Spectrum" "Attributes: wavelengths, weights, units · Methods: spectrum_params(), model()"
    dLux_sources_BaseSource <|-- dLux_sources_Source
    dLux_sources_Spectrum <|-- dLux_sources_Source
    click dLux_sources_Source href "../sources/#dLux.sources.Source" "Attributes: wavelengths, weights, position, flux, distribution, units · Methods: params()"
    dLux_sources_BaseSource <|-- dLux_sources_BinarySource
    dLux_sources_Spectrum <|-- dLux_sources_BinarySource
    click dLux_sources_BinarySource href "../sources/#dLux.sources.BinarySource" "Attributes: wavelengths, weights, centre, separation, position_angle, contrast, flux, distribution, units · Methods: params()"
    click dLux_systems_LayeredSystem href "../systems/#dLux.systems.LayeredSystem" "Attributes: layers · Methods: apply(), debug(), insert_layer(), remove_layer()"
    dLux_systems_LayeredSystem <|-- dLux_systems_OpticalSystem
    click dLux_systems_OpticalSystem href "../systems/#dLux.systems.OpticalSystem" "Attributes: layers, spec · Methods: initialise_wavefront(), propagate_mono(), propagate(), model(), debug_propagate_mono()"
    dLux_systems_LayeredSystem <|-- dLux_systems_DetectorSystem
    click dLux_systems_DetectorSystem href "../systems/#dLux.systems.DetectorSystem" "Attributes: layers · Methods: model()"
```
