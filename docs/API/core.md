
# Core classes

The `core.py` class contains the core classes that users will interact with. These classes are designed to be modular and allow for the creation of complex instruments. The `core.py` script contains the following classes:

- `Instrument`
- `Optics`
- `Detector`

Lets examine the `Optics` class first as it is the core diffraction engine of dLux.

---

## Optics

The `Optics` class is the core class that controls the diffraction of wavefronts. It has only a single attribute, `layers` which is stored as an ordered dictionary, but input as a list.

!!! tip "Construction"
    The `Optics` class only has a single input, `layers`. This is a list of `Layer` classes that will be applied to the wavefront in the order they are given. The layers input is taken as a list and automatically converted to an ordered dictionary, populating the dictionary with the `name` parameter of each layer as the key. If there are mulitple layers with the same name parameter, the names will be modified to "original_name_0", "original_name_1" etc.

    Here is how we would create a basic optics class:

    ```python
    import dLux as dl
    
    wf_npix = 256
    det_npix = 128
    aperture_diameter = 1.0
    det_pixel_size = 1e-6

    layers = [
        dl.CreateWavefront(wf_npix, aperture_diameter),
        dl.ApertureFactory(wf_npix, name="Aperture"),
        dl.NormaliseWavefront(),
        dl.AngularMFT(det_npix, det_pixel_size, name="Propagator")
    ]

    optics = dl.Optics(layers)
    ```

!!! info "A Basic Optical Setup"
    In dLux the four layers shown above form a very basic optical train and these layers are used in almost every example.

    `CreateWavefont` should be the first layer in all optical layers as this is used to actually initialse a wavefront object that is operated on by the other layers. 

    `ApertureFactory` is used to create an aperture that is applied to the wavefront. While this layer is not _strictly_ nessecary, almost all models will have some form of non-square aperture and so will likely require some form of aperture layer.

    `NormaliseWavefront` is used to normalise the wavefront to a unit intensity. This simply controls at what point the wavefront is normalised to unity power. Since most calculations in optics use the collecting area of the aperture, most use cases will require this layer directly after the aperture.

    `AngularMFT` is used to propagate the wavefront from the pupil to the focal plane as is common in most optical systems. Typically at least one propagator type layer will be needed in an optical train.

    To a degree these processes could be automated and made parameters of the `Optics` class itself, but in the goal of flexibilty and generalisability dLux instead chooses to be verbose to give users full control in order to be able to create any optical system they want.

!!! tip "Accessing Parameters"
    The Optics class is set up with a `__getattr__` method that is able to raise parameters from the `layers` attribute such that they can be accessed as class parameters via their name. Taking the example from above, we can access `Aperture` layer via:

    ```python
    aperture = optics.Aperture
    ```

    Without the `__getattr__` method we would have to use the following path to access the 'Aperture' layer:

    ```python
    aperture = optics.layers['Aperture']
    ```

    Which is a lot less nice!

The `Optics` has a simple API with `model` being its primary method. The `model` method takes in a source or list of sources and propagates them through the layers of the optics class.

??? info "model API"
    :::dLux.core.Optics.model

It also has both `propagate` and `propagate_mono` methods, which propagate a broadband and monochromatic wavelengths through the optics class respectively. They take in either an array of wavelengths or a single wavelength and return the calcualted PSF.

??? info "propagate API"
    :::dLux.core.Optics.propagate

??? info "propagate_mono API"
    :::dLux.core.Optics.propagate_mono

<!-- It also has the methods `plot` and `summarise`. `summarise` prints a brief summary of the objects it contains and `plot` will propagate a single wavelength through the instrument and show it at intemediate states, although `plot` and `summarise` and both considered to experimental and are subject to change in the future. -->
<!-- 
??? info "plot API"
    :::dLux.core.Optics.plot -->

<!-- ??? info "summarise API"
    :::dLux.core.Optics.summarise -->

<!-- ??? info "Optics API"
    :::dLux.core.Optics -->

---

## Instrument

The `Instrument` class is a high level class that is designed to control the interaction and modelling of various other classes. It has the following attributes:

- `optics`: The `Optics` class that controls the diffraction of wavefronts.
- `detector`: The `Detector` class that controls the transformation applied by a detector.
- `sources`: The `Source` class that controls the source.
- `observation`: The `Observation` class that controls custom observation stratergies.

It has a two primary methods: `Instrument.model()` & `Instrument.observe()`. The `model` method will automatically propagate all the of the sources through the optics and detector if it is present. The `observe` method will instead call the `observe` method of the `Observation` class, which is designed to give fine-grained control over how the modelling is done in order to create a set of data over multiple filters, for example.

It also has the methods `normalise`, `plot` and `summarise`. `normalise` will normalise all sources stored within the class, `summarise` prints a brief summary of the objects it contains and `plot` will propagate a single wavelength through the instrument and show it at intemediate states, although `plot` and `summarise` and both considered to experimental and are subject to change in the future.

!!! tip "Construction"
    The `Instrument` class requires both an `Optics` and `Source` object, with `Detector` and `Observation` classes being optional.

    Here is how we would create a basic instrument:

    ```python
    import dLux as dl

    instrument = dl.Instrument(optics, source)
    ```

    We can also pass in a list of sources if we want to model multiple sources:

    ```python
    import dLux as dl

    instrument = dl.Instrument(optics, [source1, source2])
    ```

!!! tip "Accessing Parameters"
    The Instrument class is set up with a `__getattr__` method that is able to raise parameters from the `optics`, `detector`, `sources` and `observation` classes if they are present. Say we have a Instrument with an `Optics` class that applies optical aberrations via a layer called `Aberrations` set via its name parameter. We would normally need to use the following path to access the `name` parameter:

    ```python
    aberration_layer = instrument.optics.layers['Aberrations']
    ```

    Which is quite long. Since all instrument classes have an optics class and all optics classes have a layers attribute, we can instead use the following:

    ```python
    aberration_layer = instrument.Aberrations
    ```

    Which is much simpler! Dont forget to set the name parameter of layers you will need to access often!

??? info "model API"
    :::dLux.core.Instrument.model

??? info "observe API"
    :::dLux.core.Instrument.observe

<!-- ??? info "Instrument API"
    :::dLux.core.Instrument -->

---

## Detector

The `Detector` class is a high level class that is designed to control the transformation applied by a detector. It has only a single attribute, `layers` which is stored as an ordered dictionary, but input as a list.

!!! tip "Construction"
    The `Detector` class only has a single input, `layers`. This is a list of `Layer` classes that will be applied to the psf in the order they are given. The layers input is taken as a list and automatically converted to an ordered dictionary, populating the dictionary with the `name` parameter of each layer as the key. If there are mulitple layers with the same name parameter, the names will be modified to "original_name_0", "original_name_1" etc.

    Here is how we would create a basic detector class:

    ```python
    import dLux as dl

    jitter = 0.5 # Pixels
    saturation = 1e5 # ADU

    layers = [
        dl.ApplyJitter(jitter, name='Jitter'),
        dl.ApplySaturation(saturation, name='Saturation'),
        ]

    detector = dl.Detector(layers)
    ```

!!! tip "Accessing Parameters"
    The Detector class is set up with a `__getattr__` method that is able to raise parameters from the `layers` attribute such that they can be accessed as class parameters via their name. Taking the example from above, we can access `Jitter` layer via:

    ```python
    jitter_layer = detector.Jitter
    ```

    Without the `__getattr__` method we would have to use the following path to access the 'Jitter' layer:

    ```python
    jitter_layer = detector.layers['Jitter']
    ```

    Which is a lot less nice!

The `Detector` has a simple API with two primary methods: `model` and `apply_detector`. The `model` method takes in an Optics object and source or list of sources, propagates them through the layers of the optics class and then applies the detector layers. The `apply_detector` method simply applies the detector layers to a given input array.

??? info "model API"
    :::dLux.core.Detector.model

??? info "apply_detector API"
    :::dLux.core.Detector.apply_detector
