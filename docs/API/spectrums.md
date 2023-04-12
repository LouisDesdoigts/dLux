# Spectra

The spectrums module just provides some simple parametrisations of spectra. These are used to model the spectra of the sources that are being observed. Each spectrum class must have an array of `wavelengths` in units of meters.

## Array Spectrum

A simple spectrum object that takes in an array of wavelengths and an array of relative intensities.

??? info "Array Spectrum API"
    :::dLux.spectrums.ArraySpectrum

## Combined Spectra

Multiple array spectra combined together defined on the same set of wavelenghts. This is primarily used for `MultiPointSource`, `BinarySource` and `PointAndExtendedSource` objects.

??? info "Combined Spectrum API"
    :::dLux.spectrums.CombinedSpectrum