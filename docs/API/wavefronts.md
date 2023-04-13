# Wavefronts

The wavefront classes primarily track all the relevant parameters of a wavefront and provide a number of methods for manipulating them. The wavefront class will rarely be interacted with directly by users unless they are creating their own custom classes. The bulk of the code is in the `Wavefront` class, which is the base class for all wavefronts. The concrete wavefront classes `AngularWavefront`, `CartesianWavefront` and `FarFieldFresnelWavefront` are very minimal and mostly just provide functions to track the different parameters units through the propagations.

??? info "Wavefront API"
    :::dLux.wavefronts.Wavefront

??? info "Angular Wavefront API"
    :::dLux.wavefronts.AngularWavefront

??? info "Cartesian Wavefront API"
    :::dLux.wavefronts.CartesianWavefront

??? info "Far Field Fresnel Wavefront API"
    :::dLux.wavefronts.FarFieldFresnelWavefront
