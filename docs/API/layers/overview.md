# Layers API

This diagram is generated from the public API. Hover over a class for its direct attributes and methods, or select it to open the full reference.

```mermaid
classDiagram
    class dLux_layers_detector_layers_BaseDetectorLayer["BaseDetectorLayer"]
    class dLux_layers_detector_layers_DetectorLayer["DetectorLayer"]
    class dLux_layers_detector_layers_ApplyPixelResponse["ApplyPixelResponse"]
    class dLux_layers_detector_layers_ApplyJitter["ApplyJitter"]
    class dLux_layers_detector_layers_ApplySaturation["ApplySaturation"]
    class dLux_layers_detector_layers_AddConstant["AddConstant"]
    class dLux_layers_dynamic_layers_BaseDynamicLayer["BaseDynamicLayer"]
    class dLux_layers_dynamic_layers_DynamicTransmissiveLayer["DynamicTransmissiveLayer"]
    class dLux_layers_dynamic_layers_DynamicAberratedLayer["DynamicAberratedLayer"]
    class dLux_layers_dynamic_layers_DynamicOptic["DynamicOptic"]
    class dLux_layers_optical_layers_BaseLayer["BaseLayer"]
    class dLux_layers_optical_layers_BaseOpticalLayer["BaseOpticalLayer"]
    class dLux_layers_optical_layers_OpticalLayer["OpticalLayer"]
    class dLux_layers_optical_layers_TransmissiveLayer["TransmissiveLayer"]
    class dLux_layers_optical_layers_AberratedLayer["AberratedLayer"]
    class dLux_layers_optical_layers_Optic["Optic"]
    class dLux_layers_optical_layers_Tilt["Tilt"]
    class dLux_layers_polarised_layers_PolarisationLayer["PolarisationLayer"]
    class dLux_layers_polarised_layers_PolarisingOptic["PolarisingOptic"]
    class dLux_layers_polarised_layers_UniformPolarisingOptic["UniformPolarisingOptic"]
    class dLux_layers_polarised_layers_LinearPolariser["LinearPolariser"]
    class dLux_layers_polarised_layers_Retarder["Retarder"]
    class dLux_layers_propagation_layers_ABCDElement["ABCDElement"]
    class dLux_layers_propagation_layers_ABCDFreeSpace["ABCDFreeSpace"]
    class dLux_layers_propagation_layers_ABCDLens["ABCDLens"]
    class dLux_layers_propagation_layers_ABCDMirror["ABCDMirror"]
    class dLux_layers_propagation_layers_ABCDFraunhofer["ABCDFraunhofer"]
    class dLux_layers_propagation_layers_Propagator["Propagator"]
    class dLux_layers_propagation_layers_FocalPropagator["FocalPropagator"]
    class dLux_layers_propagation_layers_ABCDPropagator["ABCDPropagator"]
    class dLux_layers_propagation_layers_FreeSpace["FreeSpace"]
    class dLux_layers_propagation_layers_Fraunhofer["Fraunhofer"]
    class dLux_layers_propagation_layers_Fresnel["Fresnel"]
    class dLux_layers_refractive_layers_RefractiveOptic["RefractiveOptic"]
    class dLux_layers_refractive_layers_Wedge["Wedge"]
    class dLux_layers_sparse_layers_Interfere["Interfere"]
    class dLux_layers_sparse_layers_SparseOptic["SparseOptic"]
    class dLux_layers_sparse_layers_SparseDynamicOptic["SparseDynamicOptic"]
    class dLux_layers_unified_layers_UnifiedLayer["UnifiedLayer"]
    class dLux_layers_unified_layers_Resize["Resize"]
    class dLux_layers_unified_layers_Downsample["Downsample"]
    class dLux_layers_unified_layers_Flip["Flip"]
    class dLux_layers_unified_layers_Interpolate["Interpolate"]
    class dLux_layers_unified_layers_Normalise["Normalise"]
    class dLux_layers_unified_layers_Lambda["Lambda"]
    dLux_layers_optical_layers_BaseLayer <|-- dLux_layers_detector_layers_BaseDetectorLayer
    click dLux_layers_detector_layers_BaseDetectorLayer href "../detector_layers/#dLux.layers.detector_layers.BaseDetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_layers_BaseDetectorLayer <|-- dLux_layers_detector_layers_DetectorLayer
    click dLux_layers_detector_layers_DetectorLayer href "../detector_layers/#dLux.layers.detector_layers.DetectorLayer" "No direct public attributes or methods"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplyPixelResponse
    click dLux_layers_detector_layers_ApplyPixelResponse href "../detector_layers/#dLux.layers.detector_layers.ApplyPixelResponse" "Attributes: pixel_response"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplyJitter
    click dLux_layers_detector_layers_ApplyJitter href "../detector_layers/#dLux.layers.detector_layers.ApplyJitter" "Attributes: sigma, kernel_size, oversample · Methods: kernel()"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_ApplySaturation
    click dLux_layers_detector_layers_ApplySaturation href "../detector_layers/#dLux.layers.detector_layers.ApplySaturation" "Attributes: threshold"
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_detector_layers_AddConstant
    click dLux_layers_detector_layers_AddConstant href "../detector_layers/#dLux.layers.detector_layers.AddConstant" "Attributes: value"
    dLux_layers_optical_layers_BaseOpticalLayer <|-- dLux_layers_dynamic_layers_BaseDynamicLayer
    click dLux_layers_dynamic_layers_BaseDynamicLayer href "../dynamic_layers/#dLux.layers.dynamic_layers.BaseDynamicLayer" "Attributes: coordinates, transformation · Methods: context()"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicTransmissiveLayer
    dLux_layers_optical_layers_TransmissiveLayer <|-- dLux_layers_dynamic_layers_DynamicTransmissiveLayer
    click dLux_layers_dynamic_layers_DynamicTransmissiveLayer href "../dynamic_layers/#dLux.layers.dynamic_layers.DynamicTransmissiveLayer" "Attributes: coordinates, transformation, transmission, normalise"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicAberratedLayer
    dLux_layers_optical_layers_AberratedLayer <|-- dLux_layers_dynamic_layers_DynamicAberratedLayer
    click dLux_layers_dynamic_layers_DynamicAberratedLayer href "../dynamic_layers/#dLux.layers.dynamic_layers.DynamicAberratedLayer" "Attributes: coordinates, transformation, opd, phase"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_dynamic_layers_DynamicOptic
    dLux_layers_optical_layers_Optic <|-- dLux_layers_dynamic_layers_DynamicOptic
    click dLux_layers_dynamic_layers_DynamicOptic href "../dynamic_layers/#dLux.layers.dynamic_layers.DynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise"
    click dLux_layers_optical_layers_BaseLayer href "../optical_layers/#dLux.layers.optical_layers.BaseLayer" "Methods: apply()"
    dLux_layers_optical_layers_BaseLayer <|-- dLux_layers_optical_layers_BaseOpticalLayer
    click dLux_layers_optical_layers_BaseOpticalLayer href "../optical_layers/#dLux.layers.optical_layers.BaseOpticalLayer" "No direct public attributes or methods"
    dLux_layers_optical_layers_BaseOpticalLayer <|-- dLux_layers_optical_layers_OpticalLayer
    click dLux_layers_optical_layers_OpticalLayer href "../optical_layers/#dLux.layers.optical_layers.OpticalLayer" "Methods: context()"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_TransmissiveLayer
    click dLux_layers_optical_layers_TransmissiveLayer href "../optical_layers/#dLux.layers.optical_layers.TransmissiveLayer" "Attributes: transmission, normalise"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_AberratedLayer
    click dLux_layers_optical_layers_AberratedLayer href "../optical_layers/#dLux.layers.optical_layers.AberratedLayer" "Attributes: opd, phase"
    dLux_layers_optical_layers_TransmissiveLayer <|-- dLux_layers_optical_layers_Optic
    dLux_layers_optical_layers_AberratedLayer <|-- dLux_layers_optical_layers_Optic
    click dLux_layers_optical_layers_Optic href "../optical_layers/#dLux.layers.optical_layers.Optic" "Attributes: transmission, opd, phase, normalise · Methods: phasor()"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_optical_layers_Tilt
    click dLux_layers_optical_layers_Tilt href "../optical_layers/#dLux.layers.optical_layers.Tilt" "Attributes: angles, unit"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_polarised_layers_PolarisationLayer
    click dLux_layers_polarised_layers_PolarisationLayer href "../polarised_layers/#dLux.layers.polarised_layers.PolarisationLayer" "Attributes: polarisation"
    click dLux_layers_polarised_layers_PolarisingOptic href "../polarised_layers/#dLux.layers.polarised_layers.PolarisingOptic" "Attributes: jones"
    dLux_layers_polarised_layers_PolarisingOptic <|-- dLux_layers_polarised_layers_UniformPolarisingOptic
    click dLux_layers_polarised_layers_UniformPolarisingOptic href "../polarised_layers/#dLux.layers.polarised_layers.UniformPolarisingOptic" "Attributes: jones, orientation"
    click dLux_layers_polarised_layers_LinearPolariser href "../polarised_layers/#dLux.layers.polarised_layers.LinearPolariser" "Attributes: angle · Methods: jones()"
    click dLux_layers_polarised_layers_Retarder href "../polarised_layers/#dLux.layers.polarised_layers.Retarder" "Attributes: retardance, angle · Methods: jones()"
    click dLux_layers_propagation_layers_ABCDElement href "../propagation_layers/#dLux.layers.propagation_layers.ABCDElement" "No direct public attributes or methods"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDFreeSpace
    click dLux_layers_propagation_layers_ABCDFreeSpace href "../propagation_layers/#dLux.layers.propagation_layers.ABCDFreeSpace" "Attributes: distance · Methods: abcd()"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDLens
    click dLux_layers_propagation_layers_ABCDLens href "../propagation_layers/#dLux.layers.propagation_layers.ABCDLens" "Attributes: focal_length · Methods: abcd()"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDMirror
    click dLux_layers_propagation_layers_ABCDMirror href "../propagation_layers/#dLux.layers.propagation_layers.ABCDMirror" "Attributes: radius · Methods: abcd()"
    dLux_layers_propagation_layers_ABCDElement <|-- dLux_layers_propagation_layers_ABCDFraunhofer
    click dLux_layers_propagation_layers_ABCDFraunhofer href "../propagation_layers/#dLux.layers.propagation_layers.ABCDFraunhofer" "Attributes: focal_length · Methods: abcd()"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_propagation_layers_Propagator
    click dLux_layers_propagation_layers_Propagator href "../propagation_layers/#dLux.layers.propagation_layers.Propagator" "Attributes: spec · Methods: validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_FocalPropagator
    click dLux_layers_propagation_layers_FocalPropagator href "../propagation_layers/#dLux.layers.propagation_layers.FocalPropagator" "Attributes: spec, focal_length · Methods: validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_ABCDPropagator
    click dLux_layers_propagation_layers_ABCDPropagator href "../propagation_layers/#dLux.layers.propagation_layers.ABCDPropagator" "Attributes: spec, ABCDs, method · Methods: abcd(), validate()"
    dLux_layers_propagation_layers_Propagator <|-- dLux_layers_propagation_layers_FreeSpace
    click dLux_layers_propagation_layers_FreeSpace href "../propagation_layers/#dLux.layers.propagation_layers.FreeSpace" "Attributes: spec, distance, crop"
    dLux_layers_propagation_layers_FocalPropagator <|-- dLux_layers_propagation_layers_Fraunhofer
    click dLux_layers_propagation_layers_Fraunhofer href "../propagation_layers/#dLux.layers.propagation_layers.Fraunhofer" "Attributes: spec, focal_length, method"
    dLux_layers_propagation_layers_FocalPropagator <|-- dLux_layers_propagation_layers_Fresnel
    click dLux_layers_propagation_layers_Fresnel href "../propagation_layers/#dLux.layers.propagation_layers.Fresnel" "Attributes: spec, focal_length, defocus, method"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_refractive_layers_RefractiveOptic
    click dLux_layers_refractive_layers_RefractiveOptic href "../refractive_layers/#dLux.layers.refractive_layers.RefractiveOptic" "Attributes: thickness, n"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_refractive_layers_Wedge
    click dLux_layers_refractive_layers_Wedge href "../refractive_layers/#dLux.layers.refractive_layers.Wedge" "Attributes: angle, n"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_sparse_layers_Interfere
    click dLux_layers_sparse_layers_Interfere href "../sparse_layers/#dLux.layers.sparse_layers.Interfere" "No direct public attributes or methods"
    dLux_layers_optical_layers_Optic <|-- dLux_layers_sparse_layers_SparseOptic
    click dLux_layers_sparse_layers_SparseOptic href "../sparse_layers/#dLux.layers.sparse_layers.SparseOptic" "Attributes: transmission, opd, phase, normalise, centers · Methods: n_apertures(), phasor(), wavefronts()"
    dLux_layers_dynamic_layers_BaseDynamicLayer <|-- dLux_layers_sparse_layers_SparseDynamicOptic
    dLux_layers_sparse_layers_SparseOptic <|-- dLux_layers_sparse_layers_SparseDynamicOptic
    click dLux_layers_sparse_layers_SparseDynamicOptic href "../sparse_layers/#dLux.layers.sparse_layers.SparseDynamicOptic" "Attributes: coordinates, transformation, transmission, opd, phase, normalise, centers"
    dLux_layers_optical_layers_OpticalLayer <|-- dLux_layers_unified_layers_UnifiedLayer
    dLux_layers_detector_layers_DetectorLayer <|-- dLux_layers_unified_layers_UnifiedLayer
    click dLux_layers_unified_layers_UnifiedLayer href "../unified_layers/#dLux.layers.unified_layers.UnifiedLayer" "No direct public attributes or methods"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Resize
    click dLux_layers_unified_layers_Resize href "../unified_layers/#dLux.layers.unified_layers.Resize" "Attributes: npixels"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Downsample
    click dLux_layers_unified_layers_Downsample href "../unified_layers/#dLux.layers.unified_layers.Downsample" "Attributes: n"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Flip
    click dLux_layers_unified_layers_Flip href "../unified_layers/#dLux.layers.unified_layers.Flip" "Attributes: axes"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Interpolate
    click dLux_layers_unified_layers_Interpolate href "../unified_layers/#dLux.layers.unified_layers.Interpolate" "Attributes: transformation, method, complex, fill"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Normalise
    click dLux_layers_unified_layers_Normalise href "../unified_layers/#dLux.layers.unified_layers.Normalise" "Attributes: mode, value"
    dLux_layers_unified_layers_UnifiedLayer <|-- dLux_layers_unified_layers_Lambda
    click dLux_layers_unified_layers_Lambda href "../unified_layers/#dLux.layers.unified_layers.Lambda" "No direct public attributes or methods"
```
