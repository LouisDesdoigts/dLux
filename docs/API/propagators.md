# Propagators

The propagators module contains all the optical layers that transform wavefront between pupil and focal planes. There are several different types of propagators that are combined. Angular and Cartesian propagators define their output units in either angular or cartesian coordinates. MFT and FFT based propagators have either a dynamic or fixed sampling in the output plane respectively. There is also a FarFieldFresnel propagator that is used to propagate near-to the focal plane.

This gives a total of 5 classes: `AngularMFT`, `AngularFFT`, `CartesianMFT`, `CartesianFFT` and `CartesianFresnel`. `AngularMFT` is the most commonly used propagator as it can easily model chromatic effects without interpolation and most optical systems are designed in angular coordinates.

All of the propagators have an `inverse` parameter that used to designate the type of propagation. If `inverse=False` then the propagation will from the pupil plane to the focal plane, and vice versa if `inverse=True`.

The MFT propagators all have an `npixels_out` and `pixel_scale_out` parameters that define the size and sampling of the output plane. They also have a `shift` and `pixel_shift` parameter which can be used to shift the output plane by a given amount. By default the shift values is in angular units, but if `shift_units='pixels'` then the shift will be in pixels. FFT propagators by their nature have a fixed sampling and output size, nor can they be shifted so these parameters are not used.

Cartesian propagators also have a `focal_length` parameter that is required to calculate the correct sampling in the output plane.

??? info "Angular MFT API"
    :::dLux.propagators.AngularMFT

??? info "Angular FFT API"
    :::dLux.propagators.AngularFFT

??? info "Cartesian MFT API"
    :::dLux.propagators.CartesianMFT

??? info "Cartesian FFT API"
    :::dLux.propagators.CartesianFFT

??? info "Cartesian Fresnel API"
    :::dLux.propagators.CartesianFresnel
