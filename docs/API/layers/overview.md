# Layers API

This diagram is generated from the public API. Hover over a class for its direct attributes and methods, or select it to open the full reference.

```mermaid
classDiagram
    class dLux_layers_coronagraphy_SoummerFPM["SoummerFPM"]
    class dLux_layers_detector_BaseDetectorLayer["BaseDetectorLayer"]
    class dLux_layers_detector_DetectorLayer["DetectorLayer"]
    class dLux_layers_detector_ApplyPixelResponse["ApplyPixelResponse"]
    class dLux_layers_detector_ApplyJitter["ApplyJitter"]
    class dLux_layers_detector_ApplySaturation["ApplySaturation"]
    class dLux_layers_detector_AddConstant["AddConstant"]
    class dLux_layers_dynamic_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_dynamic_DynamicTransmissiveLayer["DynamicTransmissiveLayer"]
    class dLux_layers_dynamic_DynamicAberratedLayer["DynamicAberratedLayer"]
    class dLux_layers_dynamic_DynamicOptic["DynamicOptic"]
    class dLux_layers_optical_BaseLayer["BaseLayer"]
    class dLux_layers_optical_BaseOpticalLayer["BaseOpticalLayer"]
    class dLux_layers_optical_OpticalLayer["OpticalLayer"]
    class dLux_layers_optical_TransmissiveLayer["TransmissiveLayer"]
    class dLux_layers_optical_AberratedLayer["AberratedLayer"]
    class dLux_layers_optical_Optic["Optic"]
    class dLux_layers_optical_Tilt["Tilt"]
    class dLux_layers_polarised_PolarisationLayer["PolarisationLayer"]
    class dLux_layers_polarised_PolarisingOptic["PolarisingOptic"]
    class dLux_layers_polarised_UniformPolarisingOptic["UniformPolarisingOptic"]
    class dLux_layers_polarised_LinearPolariser["LinearPolariser"]
    class dLux_layers_polarised_Retarder["Retarder"]
    class dLux_layers_propagation_ABCDElement["ABCDElement"]
    class dLux_layers_propagation_ABCDFreeSpace["ABCDFreeSpace"]
    class dLux_layers_propagation_ABCDLens["ABCDLens"]
    class dLux_layers_propagation_ABCDMirror["ABCDMirror"]
    class dLux_layers_propagation_ABCDFraunhofer["ABCDFraunhofer"]
    class dLux_layers_propagation_Propagator["Propagator"]
    class dLux_layers_propagation_FocalPropagator["FocalPropagator"]
    class dLux_layers_propagation_ABCDPropagator["ABCDPropagator"]
    class dLux_layers_propagation_FreeSpace["FreeSpace"]
    class dLux_layers_propagation_Fraunhofer["Fraunhofer"]
    class dLux_layers_propagation_Fresnel["Fresnel"]
    class dLux_layers_refractive_RefractiveOptic["RefractiveOptic"]
    class dLux_layers_refractive_Wedge["Wedge"]
    class dLux_layers_sparse_Interfere["Interfere"]
    class dLux_layers_sparse_SparseOptic["SparseOptic"]
    class dLux_layers_sparse_SparseDynamicOptic["SparseDynamicOptic"]
    class dLux_layers_unified_UnifiedLayer["UnifiedLayer"]
    class dLux_layers_unified_Resize["Resize"]
    class dLux_layers_unified_Downsample["Downsample"]
    class dLux_layers_unified_Flip["Flip"]
    class dLux_layers_unified_Interpolate["Interpolate"]
    class dLux_layers_unified_Normalise["Normalise"]
    class dLux_layers_unified_Lambda["Lambda"]
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_coronagraphy_SoummerFPM
    click dLux_layers_coronagraphy_SoummerFPM href "../coronagraphy/#dLux.layers.coronagraphy.SoummerFPM" "Attributes: optic, propagator · Methods: context(), apply_mono()"
    dLux_layers_optical_BaseLayer <|-- dLux_layers_detector_BaseDetectorLayer
    click dLux_layers_detector_BaseDetectorLayer href "../detector/#dLux.layers.detector.BaseDetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_BaseDetectorLayer <|-- dLux_layers_detector_DetectorLayer
    click dLux_layers_detector_DetectorLayer href "../detector/#dLux.layers.detector.DetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplyPixelResponse
    click dLux_layers_detector_ApplyPixelResponse href "../detector/#dLux.layers.detector.ApplyPixelResponse" "Attributes: pixel_response"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplyJitter
    click dLux_layers_detector_ApplyJitter href "../detector/#dLux.layers.detector.ApplyJitter" "Attributes: sigma, kernel_size, oversample · Properties: kernel"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_ApplySaturation
    click dLux_layers_detector_ApplySaturation href "../detector/#dLux.layers.detector.ApplySaturation" "Attributes: threshold"
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_detector_AddConstant
    click dLux_layers_detector_AddConstant href "../detector/#dLux.layers.detector.AddConstant" "Attributes: value"
    dLux_layers_optical_BaseOpticalLayer <|-- dLux_layers_dynamic_BaseDynamicLayer
    click dLux_layers_dynamic_BaseDynamicLayer href "../dynamic/#dLux.layers.dynamic.BaseDynamicLayer" "Attributes: coordinates, transformation · Methods: context()"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicTransmissiveLayer
    dLux_layers_optical_TransmissiveLayer <|-- dLux_layers_dynamic_DynamicTransmissiveLayer
    click dLux_layers_dynamic_DynamicTransmissiveLayer href "../dynamic/#dLux.layers.dynamic.DynamicTransmissiveLayer" "Attributes: coordinates, transformation, transmission, normalise"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicAberratedLayer
    dLux_layers_optical_AberratedLayer <|-- dLux_layers_dynamic_DynamicAberratedLayer
    click dLux_layers_dynamic_DynamicAberratedLayer href "../dynamic/#dLux.layers.dynamic.DynamicAberratedLayer" "Attributes: coordinates, transformation, opd, phase"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_dynamic_DynamicOptic
    dLux_layers_optical_Optic <|-- dLux_layers_dynamic_DynamicOptic
    click dLux_layers_dynamic_DynamicOptic href "../dynamic/#dLux.layers.dynamic.DynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise"
    click dLux_layers_optical_BaseLayer href "../optical/#dLux.layers.optical.BaseLayer" "No direct public attributes or methods"
    dLux_layers_optical_BaseLayer <|-- dLux_layers_optical_BaseOpticalLayer
    click dLux_layers_optical_BaseOpticalLayer href "../optical/#dLux.layers.optical.BaseOpticalLayer" "Methods: apply_mono(), apply()"
    dLux_layers_optical_BaseOpticalLayer <|-- dLux_layers_optical_OpticalLayer
    click dLux_layers_optical_OpticalLayer href "../optical/#dLux.layers.optical.OpticalLayer" "Methods: context()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_optical_TransmissiveLayer
    click dLux_layers_optical_TransmissiveLayer href "../optical/#dLux.layers.optical.TransmissiveLayer" "Attributes: transmission, normalise · Methods: apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_optical_AberratedLayer
    click dLux_layers_optical_AberratedLayer href "../optical/#dLux.layers.optical.AberratedLayer" "Attributes: opd, phase · Methods: apply_mono()"
    dLux_layers_optical_TransmissiveLayer <|-- dLux_layers_optical_Optic
    dLux_layers_optical_AberratedLayer <|-- dLux_layers_optical_Optic
    click dLux_layers_optical_Optic href "../optical/#dLux.layers.optical.Optic" "Attributes: transmission, opd, phase, normalise · Methods: phasor(), apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_optical_Tilt
    click dLux_layers_optical_Tilt href "../optical/#dLux.layers.optical.Tilt" "Attributes: angles, unit · Methods: apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_polarised_PolarisationLayer
    click dLux_layers_polarised_PolarisationLayer href "../polarised/#dLux.layers.polarised.PolarisationLayer" "Attributes: polarisation · Methods: apply_mono()"
    click dLux_layers_polarised_PolarisingOptic href "../polarised/#dLux.layers.polarised.PolarisingOptic" "Attributes: jones"
    dLux_layers_polarised_PolarisingOptic <|-- dLux_layers_polarised_UniformPolarisingOptic
    click dLux_layers_polarised_UniformPolarisingOptic href "../polarised/#dLux.layers.polarised.UniformPolarisingOptic" "Attributes: jones, orientation · Methods: apply_mono()"
    click dLux_layers_polarised_LinearPolariser href "../polarised/#dLux.layers.polarised.LinearPolariser" "Attributes: angle · Properties: jones · Methods: apply_mono()"
    click dLux_layers_polarised_Retarder href "../polarised/#dLux.layers.polarised.Retarder" "Attributes: retardance, angle · Properties: jones · Methods: apply_mono()"
    click dLux_layers_propagation_ABCDElement href "../propagation/#dLux.layers.propagation.ABCDElement" "No direct public attributes or methods"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDFreeSpace
    click dLux_layers_propagation_ABCDFreeSpace href "../propagation/#dLux.layers.propagation.ABCDFreeSpace" "Attributes: distance · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDLens
    click dLux_layers_propagation_ABCDLens href "../propagation/#dLux.layers.propagation.ABCDLens" "Attributes: focal_length · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDMirror
    click dLux_layers_propagation_ABCDMirror href "../propagation/#dLux.layers.propagation.ABCDMirror" "Attributes: radius · Properties: abcd"
    dLux_layers_propagation_ABCDElement <|-- dLux_layers_propagation_ABCDFraunhofer
    click dLux_layers_propagation_ABCDFraunhofer href "../propagation/#dLux.layers.propagation.ABCDFraunhofer" "Attributes: focal_length · Properties: abcd"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_propagation_Propagator
    click dLux_layers_propagation_Propagator href "../propagation/#dLux.layers.propagation.Propagator" "Attributes: grid · Methods: apply(), validate()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_FocalPropagator
    click dLux_layers_propagation_FocalPropagator href "../propagation/#dLux.layers.propagation.FocalPropagator" "Attributes: grid, focal_length, inverse · Methods: validate()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_ABCDPropagator
    click dLux_layers_propagation_ABCDPropagator href "../propagation/#dLux.layers.propagation.ABCDPropagator" "Attributes: grid, ABCDs, method · Properties: abcd · Methods: validate(), apply_mono()"
    dLux_layers_propagation_Propagator <|-- dLux_layers_propagation_FreeSpace
    click dLux_layers_propagation_FreeSpace href "../propagation/#dLux.layers.propagation.FreeSpace" "Attributes: grid, distance, crop · Methods: apply_mono()"
    dLux_layers_propagation_FocalPropagator <|-- dLux_layers_propagation_Fraunhofer
    click dLux_layers_propagation_Fraunhofer href "../propagation/#dLux.layers.propagation.Fraunhofer" "Attributes: grid, focal_length, inverse, method · Methods: apply_mono()"
    dLux_layers_propagation_FocalPropagator <|-- dLux_layers_propagation_Fresnel
    click dLux_layers_propagation_Fresnel href "../propagation/#dLux.layers.propagation.Fresnel" "Attributes: grid, focal_length, inverse, defocus, method · Methods: apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_refractive_RefractiveOptic
    click dLux_layers_refractive_RefractiveOptic href "../refractive/#dLux.layers.refractive.RefractiveOptic" "Attributes: thickness, n · Methods: apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_refractive_Wedge
    click dLux_layers_refractive_Wedge href "../refractive/#dLux.layers.refractive.Wedge" "Attributes: angle, n · Methods: apply_mono()"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_sparse_Interfere
    click dLux_layers_sparse_Interfere href "../sparse/#dLux.layers.sparse.Interfere" "Methods: apply(), apply_mono()"
    dLux_layers_optical_Optic <|-- dLux_layers_sparse_SparseOptic
    click dLux_layers_sparse_SparseOptic href "../sparse/#dLux.layers.sparse.SparseOptic" "Attributes: transmission, opd, phase, normalise, centers · Properties: n_apertures · Methods: phasor(), localise(), apply_mono()"
    dLux_layers_dynamic_BaseDynamicLayer <|-- dLux_layers_sparse_SparseDynamicOptic
    dLux_layers_sparse_SparseOptic <|-- dLux_layers_sparse_SparseDynamicOptic
    click dLux_layers_sparse_SparseDynamicOptic href "../sparse/#dLux.layers.sparse.SparseDynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise, centers"
    dLux_layers_optical_OpticalLayer <|-- dLux_layers_unified_UnifiedLayer
    dLux_layers_detector_DetectorLayer <|-- dLux_layers_unified_UnifiedLayer
    click dLux_layers_unified_UnifiedLayer href "../unified/#dLux.layers.unified.UnifiedLayer" "No direct public attributes or methods"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Resize
    click dLux_layers_unified_Resize href "../unified/#dLux.layers.unified.Resize" "Attributes: npixels · Methods: apply_mono()"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Downsample
    click dLux_layers_unified_Downsample href "../unified/#dLux.layers.unified.Downsample" "Attributes: n · Methods: apply_mono()"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Flip
    click dLux_layers_unified_Flip href "../unified/#dLux.layers.unified.Flip" "Attributes: axes · Methods: apply_mono()"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Interpolate
    click dLux_layers_unified_Interpolate href "../unified/#dLux.layers.unified.Interpolate" "Attributes: transformation, method, complex, fill · Methods: apply_mono()"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Normalise
    click dLux_layers_unified_Normalise href "../unified/#dLux.layers.unified.Normalise" "Attributes: mode, value · Methods: apply_mono()"
    dLux_layers_unified_UnifiedLayer <|-- dLux_layers_unified_Lambda
    click dLux_layers_unified_Lambda href "../unified/#dLux.layers.unified.Lambda" "Methods: apply_mono()"
```
