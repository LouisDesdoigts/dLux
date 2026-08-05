# API

This package map is generated from cross-module inheritance in the public API. Hover over a module to see its public classes, or select it to open its reference.

```mermaid
classDiagram
    class dLux_fields["fields"]
    click dLux_fields href "../core/fields/" "Public classes: BaseField, ContinuousField, DiscreteField, Wavefront, PolarisedWavefront, PSF, Image"
    class dLux_grids["grids"]
    click dLux_grids href "../core/grids/" "Public classes: GridSpec, ResizeSpec, CoordTransform, Affine, AffineMap, TransformChain, DistortCoords"
    class dLux_sources["sources"]
    click dLux_sources href "../core/sources/" "Public classes: BaseSource, Spectrum, Source, BinarySource"
    class dLux_systems["systems"]
    click dLux_systems href "../core/systems/" "Public classes: LayeredSystem, OpticalSystem, DetectorSystem"
    class dLux_layers_detector_layers["layers.detector_layers"]
    click dLux_layers_detector_layers href "../layers/detector_layers/" "Public classes: BaseDetectorLayer, DetectorLayer, ApplyPixelResponse, ApplyJitter, ApplySaturation, AddConstant"
    class dLux_layers_dynamic_layers["layers.dynamic_layers"]
    click dLux_layers_dynamic_layers href "../layers/dynamic_layers/" "Public classes: BaseDynamicLayer, DynamicTransmissiveLayer, DynamicAberratedLayer, DynamicOptic"
    class dLux_layers_optical_layers["layers.optical_layers"]
    click dLux_layers_optical_layers href "../layers/optical_layers/" "Public classes: BaseLayer, BaseOpticalLayer, OpticalLayer, TransmissiveLayer, AberratedLayer, Optic, Tilt"
    class dLux_layers_polarised_layers["layers.polarised_layers"]
    click dLux_layers_polarised_layers href "../layers/polarised_layers/" "Public classes: PolarisationLayer, PolarisingOptic, UniformPolarisingOptic, LinearPolariser, Retarder"
    class dLux_layers_propagation_layers["layers.propagation_layers"]
    click dLux_layers_propagation_layers href "../layers/propagation_layers/" "Public classes: ABCDElement, ABCDFreeSpace, ABCDLens, ABCDMirror, ABCDFraunhofer, Propagator, FocalPropagator, ABCDPropagator, FreeSpace, Fraunhofer, Fresnel"
    class dLux_layers_refractive_layers["layers.refractive_layers"]
    click dLux_layers_refractive_layers href "../layers/refractive_layers/" "Public classes: RefractiveOptic, Wedge"
    class dLux_layers_sparse_layers["layers.sparse_layers"]
    click dLux_layers_sparse_layers href "../layers/sparse_layers/" "Public classes: Interfere, SparseOptic, SparseDynamicOptic"
    class dLux_layers_unified_layers["layers.unified_layers"]
    click dLux_layers_unified_layers href "../layers/unified_layers/" "Public classes: UnifiedLayer, Resize, Downsample, Flip, Interpolate, Normalise, Lambda"
    class dLux_parametric_bases["parametric.bases"]
    click dLux_parametric_bases href "../parametric/bases/" "Public classes: ParametricBasis, ExplicitBasis, ImplicitBasis, CoordBasis, CLIMBBasis, FourierBasis, SplineBasis"
    class dLux_parametric_parametrics["parametric.parametrics"]
    click dLux_parametric_parametrics href "../parametric/parametrics/" "Public classes: Parametric, ParametricHolder, Transform, Interpolation, DynamicParametric, Combination"
    class dLux_parametric_polynomials["parametric.polynomials"]
    click dLux_parametric_polynomials href "../parametric/polynomials/" "Public classes: DynamicZernike, ZernikeBasis, DynamicZernikeBasis, Polynomial, ExplicitPolynomial, CoordinatePolynomial"
    class dLux_parametric_refractive["parametric.refractive"]
    click dLux_parametric_refractive href "../parametric/refractive/" "Public classes: CauchyIndex, PolynomialIndex, InterpolatedIndex"
    class dLux_parametric_shapes["parametric.shapes"]
    click dLux_parametric_shapes href "../parametric/shapes/" "Public classes: Shape, SoftShape, RadialShape, Circle, Square, Rectangle, RegularPolygon, Spider, Complement, TransformedShape"
    dLux_layers_detector_layers <|-- dLux_layers_unified_layers
    dLux_layers_dynamic_layers <|-- dLux_layers_sparse_layers
    dLux_layers_optical_layers <|-- dLux_layers_detector_layers
    dLux_layers_optical_layers <|-- dLux_layers_dynamic_layers
    dLux_layers_optical_layers <|-- dLux_layers_polarised_layers
    dLux_layers_optical_layers <|-- dLux_layers_propagation_layers
    dLux_layers_optical_layers <|-- dLux_layers_refractive_layers
    dLux_layers_optical_layers <|-- dLux_layers_sparse_layers
    dLux_layers_optical_layers <|-- dLux_layers_unified_layers
    dLux_layers_optical_layers <|-- dLux_systems
    dLux_parametric_bases <|-- dLux_parametric_polynomials
    dLux_parametric_parametrics <|-- dLux_layers_optical_layers
    dLux_parametric_parametrics <|-- dLux_parametric_bases
    dLux_parametric_parametrics <|-- dLux_parametric_refractive
    dLux_parametric_parametrics <|-- dLux_parametric_shapes
    dLux_parametric_parametrics <|-- dLux_sources
```
