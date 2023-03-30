
# Core classes

The `core.py` class contains the core classes that users will interact with. These classes are designed to be modular and allow for the creation of complex instruments. The `core.py` script contains the following classes:

- `Instrument`
- `Optics`
- `Detector`

## Instrument

The `Instrument` class is a high level class that is designed to control the interaction and modelling of various other classes. It has the following attributes:

- `optics`: The `Optics` class that controls the diffraction of wavefronts.
- `detector`: The `Detector` class that controls the transformation applied by a detector.
- `sources`: The `Source` class that controls the source.
- `observation`: The `Observation` class that controls custom observation stratergies.

It only has a two methods:

1. `model`

??? info "model API"
    :::dLux.core.Instrument.model

2. `observe`

??? info "observe API"
    :::dLux.core.Instrument.model
