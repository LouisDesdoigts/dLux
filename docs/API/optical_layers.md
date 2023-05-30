# Optical Layers: optical_layers.py

This module contains the classes that define the behaviour of OpticalLayers in dLux.

These classes provide a simple set of classes used to perform basic transformations of wavefronts.

There are 7 public classes:

- `Optic`
- `PhaseOptic`
- `BasisOptic`
- `PhaseBasisOptic`
- `Tilt`
- `Normalise`
- `Rotate`

The 'Optic' (`Optic`, `PhaseOptic`, `BasisOptic`, `PhaseBasisOptic`) classes are quite general and are used to perform basic modifications to both the wavefront amplitude and phase. They all have a `transmission` and `normalise` attribute that modify the amplitude of the wavefront, and optionally normalise the wavefront after applying the Optic.

The `Optic` class holds a static OPD array that is added to the wavefront. The `PhaseOptic` class holds a static phase array that is added to the wavefront.

??? info "Optics API"
    ::: dLux.optical_layers.Optic

??? info "PhaseOptic API"
    ::: dLux.optical_layers.PhaseOptic

The `BasisOptic` class holds a set of basis vectors and coefficients that are used to calculate the OPD array that is added to the wavefront. The `PhaseBasisOptic` class holds a set of basis vectors and coefficients that are used to calculate the phase array that is added to the wavefront.

??? info "BasisOptic API"
    ::: dLux.optical_layers.BasisOptic

??? info "PhaseBasisOptic API"
    ::: dLux.optical_layers.PhaseBasisOptic

The `Tilt` class tilts the wavefront by the input angles.

??? info "Tilt API"
    ::: dLux.optical_layers.Tilt

The `Normalise` class normalises the wavefront.

??? info "Normalise API"
    ::: dLux.optical_layers.Normalise

The `Rotate` class rotates the wavefront by the input angle.

??? info "Rotate API"
    ::: dLux.optical_layers.Rotate