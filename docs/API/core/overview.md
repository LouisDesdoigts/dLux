# Core API

This diagram is generated from the public API. Hover over a class for its direct attributes and methods, or select it to open the full reference.

```mermaid
classDiagram
    class dLux_base_Base["Base"]
    class dLux_builders_GridBuilder["GridBuilder"]
    class dLux_builders_OPDDef["OPDDef"]
    class dLux_builders_ApertureData["ApertureData"]
    class dLux_builders_Norm["Norm"]
    class dLux_builders_ZernikeDef["ZernikeDef"]
    class dLux_builders_ApertureBuilder["ApertureBuilder"]
    class dLux_builders_SparseApertureBuilder["SparseApertureBuilder"]
    class dLux_compatibility_ABCDConjugatePlane["ABCDConjugatePlane"]
    class dLux_compatibility_ASMPropagator["ASMPropagator"]
    class dLux_compatibility_AberratedAperture["AberratedAperture"]
    class dLux_compatibility_AngularOpticalSystem["AngularOpticalSystem"]
    class dLux_grids_CoordTransform["CoordTransform"]
    class dLux_compatibility_BaseDetector["BaseDetector"]
    class dLux_compatibility_BaseOpticalSystem["BaseOpticalSystem"]
    class dLux_compatibility_BaseSpectrum["BaseSpectrum"]
    class dLux_compatibility_BasisLayer["BasisLayer"]
    class dLux_compatibility_BasisOptic["BasisOptic"]
    class dLux_compatibility_CartesianOpticalSystem["CartesianOpticalSystem"]
    class dLux_compatibility_CircularAperture["CircularAperture"]
    class dLux_compatibility_CompoundAperture["CompoundAperture"]
    class dLux_compatibility_CoordSpec["CoordSpec"]
    class dLux_compatibility_DistortedCoords["DistortedCoords"]
    class dLux_compatibility_Dither["Dither"]
    class dLux_compatibility_FFT["FFT"]
    class dLux_compatibility_FFTPropagator["FFTPropagator"]
    class dLux_compatibility_Instrument["Instrument"]
    class dLux_compatibility_LayeredDetector["LayeredDetector"]
    class dLux_compatibility_LayeredOpticalSystem["LayeredOpticalSystem"]
    class dLux_compatibility_MFT["MFT"]
    class dLux_compatibility_MFTPropagator["MFTPropagator"]
    class dLux_compatibility_MultiAperture["MultiAperture"]
    class dLux_compatibility_PadSpec["PadSpec"]
    class dLux_compatibility_ParametricOpticalSystem["ParametricOpticalSystem"]
    class dLux_compatibility_PointResolvedSource["PointResolvedSource"]
    class dLux_compatibility_PointSource["PointSource"]
    class dLux_compatibility_PointSources["PointSources"]
    class dLux_compatibility_PolySpectrum["PolySpectrum"]
    class dLux_compatibility_RectangularAperture["RectangularAperture"]
    class dLux_compatibility_RegPolyAperture["RegPolyAperture"]
    class dLux_compatibility_ResolvedSource["ResolvedSource"]
    class dLux_compatibility_Rotate["Rotate"]
    class dLux_compatibility_Scene["Scene"]
    class dLux_grids_BaseGridSpec["BaseGridSpec"]
    class dLux_compatibility_SquareAperture["SquareAperture"]
    class dLux_compatibility_Telescope["Telescope"]
    class dLux_compatibility_Zernike["Zernike"]
    class dLux_fields_BaseField["BaseField"]
    class dLux_fields_ContinuousField["ContinuousField"]
    class dLux_fields_DiscreteField["DiscreteField"]
    class dLux_fields_Wavefront["Wavefront"]
    class dLux_fields_PolarisedWavefront["PolarisedWavefront"]
    class dLux_fields_PSF["PSF"]
    class dLux_fields_Image["Image"]
    class dLux_grids_GridSpec["GridSpec"]
    class dLux_grids_ResizeSpec["ResizeSpec"]
    class dLux_grids_PasteSpec["PasteSpec"]
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
    click dLux_base_Base href "../base/#dLux.base.Base" "Methods: get(), set(), add(), multiply(), divide(), power(), min(), max()"
    dLux_base_Base <|-- dLux_builders_GridBuilder
    click dLux_builders_GridBuilder href "../builders/#dLux.builders.GridBuilder" "Methods: validate(), build()"
    dLux_base_Base <|-- dLux_builders_OPDDef
    click dLux_builders_OPDDef href "../builders/#dLux.builders.OPDDef" "Methods: calculate()"
    dLux_base_Base <|-- dLux_builders_ApertureData
    click dLux_builders_ApertureData href "../builders/#dLux.builders.ApertureData" "Attributes: transmission, support, diameter, centers"
    dLux_base_Base <|-- dLux_builders_Norm
    click dLux_builders_Norm href "../builders/#dLux.builders.Norm" "Attributes: mode, scale"
    dLux_builders_OPDDef <|-- dLux_builders_ZernikeDef
    click dLux_builders_ZernikeDef href "../builders/#dLux.builders.ZernikeDef" "Attributes: nolls, groups, order, oversize, norm, method · Methods: calculate()"
    dLux_builders_GridBuilder <|-- dLux_builders_ApertureBuilder
    click dLux_builders_ApertureBuilder href "../builders/#dLux.builders.ApertureBuilder" "Attributes: primary, obscurations, opd, oversample · Methods: build(), aperture_data()"
    dLux_builders_ApertureBuilder <|-- dLux_builders_SparseApertureBuilder
    click dLux_builders_SparseApertureBuilder href "../builders/#dLux.builders.SparseApertureBuilder" "Attributes: centers, global_obscurations · Methods: aperture_data()"
    click dLux_compatibility_ABCDConjugatePlane href "../compatibility/#dLux.compatibility.ABCDConjugatePlane" "No direct public attributes or methods"
    click dLux_compatibility_ASMPropagator href "../compatibility/#dLux.compatibility.ASMPropagator" "No direct public attributes or methods"
    click dLux_compatibility_AberratedAperture href "../compatibility/#dLux.compatibility.AberratedAperture" "No direct public attributes or methods"
    click dLux_compatibility_AngularOpticalSystem href "../compatibility/#dLux.compatibility.AngularOpticalSystem" "No direct public attributes or methods"
    dLux_base_Base <|-- dLux_grids_CoordTransform
    click dLux_grids_CoordTransform href "../grids/#dLux.grids.CoordTransform" "Methods: get_coordinates(), apply()"
    click dLux_compatibility_BaseDetector href "../compatibility/#dLux.compatibility.BaseDetector" "No direct public attributes or methods"
    click dLux_compatibility_BaseOpticalSystem href "../compatibility/#dLux.compatibility.BaseOpticalSystem" "No direct public attributes or methods"
    click dLux_compatibility_BaseSpectrum href "../compatibility/#dLux.compatibility.BaseSpectrum" "No direct public attributes or methods"
    click dLux_compatibility_BasisLayer href "../compatibility/#dLux.compatibility.BasisLayer" "No direct public attributes or methods"
    click dLux_compatibility_BasisOptic href "../compatibility/#dLux.compatibility.BasisOptic" "No direct public attributes or methods"
    click dLux_compatibility_CartesianOpticalSystem href "../compatibility/#dLux.compatibility.CartesianOpticalSystem" "No direct public attributes or methods"
    click dLux_compatibility_CircularAperture href "../compatibility/#dLux.compatibility.CircularAperture" "No direct public attributes or methods"
    click dLux_compatibility_CompoundAperture href "../compatibility/#dLux.compatibility.CompoundAperture" "No direct public attributes or methods"
    dLux_grids_GridSpec <|-- dLux_compatibility_CoordSpec
    click dLux_compatibility_CoordSpec href "../compatibility/#dLux.compatibility.CoordSpec" "Properties: xs, fov, extent"
    dLux_grids_DistortCoords <|-- dLux_compatibility_DistortedCoords
    click dLux_compatibility_DistortedCoords href "../compatibility/#dLux.compatibility.DistortedCoords" "Methods: calculate()"
    click dLux_compatibility_Dither href "../compatibility/#dLux.compatibility.Dither" "No direct public attributes or methods"
    click dLux_compatibility_FFT href "../compatibility/#dLux.compatibility.FFT" "No direct public attributes or methods"
    click dLux_compatibility_FFTPropagator href "../compatibility/#dLux.compatibility.FFTPropagator" "No direct public attributes or methods"
    click dLux_compatibility_Instrument href "../compatibility/#dLux.compatibility.Instrument" "No direct public attributes or methods"
    dLux_systems_DetectorSystem <|-- dLux_compatibility_LayeredDetector
    click dLux_compatibility_LayeredDetector href "../compatibility/#dLux.compatibility.LayeredDetector" "Methods: model()"
    dLux_systems_OpticalSystem <|-- dLux_compatibility_LayeredOpticalSystem
    click dLux_compatibility_LayeredOpticalSystem href "../compatibility/#dLux.compatibility.LayeredOpticalSystem" "Properties: wf_npixels, diameter"
    click dLux_compatibility_MFT href "../compatibility/#dLux.compatibility.MFT" "No direct public attributes or methods"
    click dLux_compatibility_MFTPropagator href "../compatibility/#dLux.compatibility.MFTPropagator" "No direct public attributes or methods"
    click dLux_compatibility_MultiAperture href "../compatibility/#dLux.compatibility.MultiAperture" "No direct public attributes or methods"
    dLux_grids_ResizeSpec <|-- dLux_compatibility_PadSpec
    click dLux_compatibility_PadSpec href "../compatibility/#dLux.compatibility.PadSpec" "No direct public attributes or methods"
    click dLux_compatibility_ParametricOpticalSystem href "../compatibility/#dLux.compatibility.ParametricOpticalSystem" "No direct public attributes or methods"
    click dLux_compatibility_PointResolvedSource href "../compatibility/#dLux.compatibility.PointResolvedSource" "No direct public attributes or methods"
    dLux_sources_Source <|-- dLux_compatibility_PointSource
    click dLux_compatibility_PointSource href "../compatibility/#dLux.compatibility.PointSource" "No direct public attributes or methods"
    dLux_sources_Source <|-- dLux_compatibility_PointSources
    click dLux_compatibility_PointSources href "../compatibility/#dLux.compatibility.PointSources" "No direct public attributes or methods"
    click dLux_compatibility_PolySpectrum href "../compatibility/#dLux.compatibility.PolySpectrum" "No direct public attributes or methods"
    click dLux_compatibility_RectangularAperture href "../compatibility/#dLux.compatibility.RectangularAperture" "No direct public attributes or methods"
    click dLux_compatibility_RegPolyAperture href "../compatibility/#dLux.compatibility.RegPolyAperture" "No direct public attributes or methods"
    dLux_sources_Source <|-- dLux_compatibility_ResolvedSource
    click dLux_compatibility_ResolvedSource href "../compatibility/#dLux.compatibility.ResolvedSource" "No direct public attributes or methods"
    click dLux_compatibility_Rotate href "../compatibility/#dLux.compatibility.Rotate" "No direct public attributes or methods"
    click dLux_compatibility_Scene href "../compatibility/#dLux.compatibility.Scene" "No direct public attributes or methods"
    dLux_base_Base <|-- dLux_grids_BaseGridSpec
    click dLux_grids_BaseGridSpec href "../grids/#dLux.grids.BaseGridSpec" "No direct public attributes or methods"
    click dLux_compatibility_SquareAperture href "../compatibility/#dLux.compatibility.SquareAperture" "No direct public attributes or methods"
    click dLux_compatibility_Telescope href "../compatibility/#dLux.compatibility.Telescope" "No direct public attributes or methods"
    click dLux_compatibility_Zernike href "../compatibility/#dLux.compatibility.Zernike" "No direct public attributes or methods"
    dLux_base_Base <|-- dLux_fields_BaseField
    click dLux_fields_BaseField href "../fields/#dLux.fields.BaseField" "Attributes: grid · Properties: field, spatial_shape, axes, coordinates, xs, npixels, pixel_scale, center, diameter · Methods: normalise(), convolve(), resize(), downsample(), flip()"
    dLux_fields_BaseField <|-- dLux_fields_ContinuousField
    click dLux_fields_ContinuousField href "../fields/#dLux.fields.ContinuousField" "Attributes: grid · Methods: scale_to(), interpolate(), rotate()"
    dLux_fields_BaseField <|-- dLux_fields_DiscreteField
    click dLux_fields_DiscreteField href "../fields/#dLux.fields.DiscreteField" "Attributes: grid · Properties: fourier_transform, amplitude_spectrum, power_spectrum"
    dLux_fields_ContinuousField <|-- dLux_fields_Wavefront
    click dLux_fields_Wavefront href "../fields/#dLux.fields.Wavefront" "Attributes: grid, phasor, wavelength · Properties: field, real, imaginary, amplitude, phase, complex, polar, psf, wavenumber, batch_ndim, is_chromatic, is_polarised, _mapped_axis, power · Methods: from_phasor(), add_phase(), add_opd(), tilt(), normalise(), apply_jones(), psf_from_stokes()"
    dLux_fields_Wavefront <|-- dLux_fields_PolarisedWavefront
    click dLux_fields_PolarisedWavefront href "../fields/#dLux.fields.PolarisedWavefront" "Attributes: grid, phasor, wavelength · Properties: is_polarised, batch_ndim, psf · Methods: from_phasor(), from_wavefront(), psf_from_stokes(), stokes(), apply_jones()"
    dLux_fields_ContinuousField <|-- dLux_fields_PSF
    click dLux_fields_PSF href "../fields/#dLux.fields.PSF" "Attributes: data, grid · Properties: field, batch_ndim · Methods: from_wavefront()"
    dLux_fields_DiscreteField <|-- dLux_fields_Image
    click dLux_fields_Image href "../fields/#dLux.fields.Image" "Attributes: data, grid, variance, read_noise · Properties: field, error · Methods: z_score(), add_poisson_noise(), add_read_noise(), simulate(), log_likelihood()"
    dLux_grids_BaseGridSpec <|-- dLux_grids_GridSpec
    click dLux_grids_GridSpec href "../grids/#dLux.grids.GridSpec" "Attributes: n, d, c, unit · Properties: ndim, shape, scale, axes, xs, coordinates, fov · Methods: broadcast(), resize(), downsample(), oversample(), resample(), from_axes(), build(), axes_for(), xs_for(), transformed(), coordinates_for(), extent()"
    dLux_grids_BaseGridSpec <|-- dLux_grids_ResizeSpec
    click dLux_grids_ResizeSpec href "../grids/#dLux.grids.ResizeSpec" "Attributes: n, pad, crop, c · Properties: explicit, padding · Methods: broadcast(), output_size(), crop_size(), pad_array(), crop_array(), crop_axes(), resize()"
    dLux_grids_BaseGridSpec <|-- dLux_grids_PasteSpec
    click dLux_grids_PasteSpec href "../grids/#dLux.grids.PasteSpec" "Attributes: n, shape, starts, offsets, d · Properties: coordinates · Methods: from_grid(), paste(), extract()"
    dLux_base_Base <|-- dLux_grids_CoordTransform
    click dLux_grids_CoordTransform href "../grids/#dLux.grids.CoordTransform" "Methods: get_coordinates(), apply()"
    dLux_grids_CoordTransform <|-- dLux_grids_Affine
    click dLux_grids_Affine href "../grids/#dLux.grids.Affine" "Attributes: translation, rotation, scale, shear, order · Properties: coeffs"
    dLux_grids_CoordTransform <|-- dLux_grids_AffineMap
    click dLux_grids_AffineMap href "../grids/#dLux.grids.AffineMap" "Attributes: matrix, offset"
    dLux_grids_CoordTransform <|-- dLux_grids_TransformChain
    click dLux_grids_TransformChain href "../grids/#dLux.grids.TransformChain" "Attributes: transformations"
    dLux_grids_CoordTransform <|-- dLux_grids_DistortCoords
    click dLux_grids_DistortCoords href "../grids/#dLux.grids.DistortCoords" "Attributes: powers, distortion, shift_invariant"
    dLux_builders_ApertureBuilder <|-- dLux_prebuilt_SimpleCircular
    click dLux_prebuilt_SimpleCircular href "../prebuilt/#dLux.prebuilt.SimpleCircular" "No direct public attributes or methods"
    dLux_builders_SparseApertureBuilder <|-- dLux_prebuilt_SegmentedHex
    click dLux_prebuilt_SegmentedHex href "../prebuilt/#dLux.prebuilt.SegmentedHex" "Attributes: paste_method · Methods: build()"
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
    dLux_base_Base <|-- dLux_systems_LayeredSystem
    click dLux_systems_LayeredSystem href "../systems/#dLux.systems.LayeredSystem" "Attributes: layers · Methods: debug(), insert_layer(), remove_layer()"
    dLux_systems_LayeredSystem <|-- dLux_systems_OpticalSystem
    click dLux_systems_OpticalSystem href "../systems/#dLux.systems.OpticalSystem" "Attributes: layers, grid · Methods: apply_mono(), apply(), initialise_wavefront(), propagate_mono(), propagate(), model(), debug_propagate_mono()"
    dLux_systems_LayeredSystem <|-- dLux_systems_DetectorSystem
    click dLux_systems_DetectorSystem href "../systems/#dLux.systems.DetectorSystem" "Attributes: layers · Methods: model()"
```
