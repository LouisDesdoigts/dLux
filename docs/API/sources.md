# Sources

The sources module contains different parametrisations of source objects. These are used to model the light sources that are being observed. Each source class must have at least a `position` `flux` and `spectrum` attribute. The position defaults to the center of the image, ie aligned with the optical axis. Flux defaults to one, giving a unit psf. The only parameter that must be specific is the the `spectrum`, however we can instead pass in an array of wavelengths and an appropriate spectrum object will be generated. Resolved sources can also be modelled and are applied via a convolution.

All source positions are defined in radians and all fluxes are in units of photons.

!!! tip "Acessing Parameters"
    The `Source` class has an in-built `__getattr__` class that allows for the spectrum parameters to be accessed from the `Source` object. That means if we wanted to access the `wavelengths` parameter  of a `PointSource` we could do so like this:

    ```python
    wavelengths = PointSource.wavelengths
    ```

    As opposed to the longer:

    ```python
    wavelengths = PointSource.spectrum.wavelengths
    ```

There are also a series of different parametrisation of source objects, lets have a look:

## Point Source

A simple point source objects.

??? info "Point Source API"
    :::dLux.sources.PointSource

## Multi-Point Source

Models a series of point sources at different positions and fluxes with a single spectrum.

??? info "Multi-Point Source API"
    :::dLux.sources.MultiPointSource

## Binary Source

Models a binary source with a differing spectra. Parametrised by separation, position angle, and flux ratio.

??? info "Binary Source API"
    :::dLux.sources.BinarySource

## Array Distribution

Models a single resolved source source object stored as an array of relative intensities.

??? info "Array Distribution API"
    :::dLux.sources.ArrayDistribution

## Point Extended Source

Models a single unresolved source and a single resolved source with a shared spectrum. An example would be a star and its dust-shell.

??? info "Point Extended Source API"
    :::dLux.sources.PointExtendedSource

## Point and Extended Source

Models a single unresolved source and a single resolved source with a unique spectrums. An example would be a quasar and its host galaxy.

??? info "Point and Extended Source API"
    :::dLux.sources.PointAndExtendedSource