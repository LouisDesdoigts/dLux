from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


def test_model(
        create_optics: callable,
        create_detector: callable,
        create_point_source: callable) -> None:
    """
    Test the model function
    """
    optics = create_optics()
    detector = create_detector()
    source = create_point_source()
    sources = [source, source]

    # Test non-optics input
    with pytest.raises(AssertionError):
        dLux.core.model([], [])

    # Test non detector input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, sources, detector=[])

    # Test no source inputs
    with pytest.raises(TypeError):
        dLux.core.model(optics)

    # Test sources with non-source input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, sources=[source, 1.])

    # Test different combinations of inputs, detector
    out = dLux.core.model(optics, sources=sources, detector=detector)

    # Test normalise function for scene
    out = dLux.core.model(optics, sources=sources)

    # Test normalise function for sources
    out = dLux.core.model(optics, sources=sources, normalise=False)

    # Test return_tree with different structures, list
    out = dLux.core.model(optics, sources=sources, return_tree=True)

    # Test return_tree with different structures, dict
    out = dLux.core.model(optics, sources={"source": source}, return_tree=True)

    # Test return_tree with different structures, source
    out = dLux.core.model(optics, sources=sources, return_tree=True)

    # Test return_tree with detector input
    out = dLux.core.model(optics, sources=sources, detector=detector, return_tree=True)

    # Test with flatten
    out = dLux.core.model(optics, sources=sources, flatten=True)

    # Test with flatten and return tree
    out = dLux.core.model(optics, sources=sources, return_tree=True, flatten=True)


class TestOptics(object):
    """
    Tests the Optics class.
    """


    def test_constructor(self, 
        create_optics: callable, 
        create_create_wavefront: callable) -> None:
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(ValueError):
            create_optics(layers={})

        # Test fist layer not CreateWavefront input
        with pytest.raises(ValueError):
            create_optics(layers=[10.])
        
        # Test non layer input
        with pytest.raises(ValueError):
            create_optics(layers=[create_create_wavefront(), 10.])
        
        # Test no propagator warning
        with pytest.warns():
            create_optics(layers=[create_create_wavefront()])
    

    # def _test_getattr(self, create_optics: callable) -> None:
    #     """
    #     Tests the __getattr__ method.
    #     """
    #     osys = create_optics()

    #     # Test non-existant attribute
    #     with pytest.raises(AttributeError):
    #         osys.non_existant

    #     # Test existing attribute
    #     osys.layers

    #     # Test name referece
    #     osys.CartesianMFT


    def test_propagate_mono(self, create_optics: callable) -> None:
        """
        Tests the propagate_mono method.
        """
        osys = create_optics() 

        # Test inputs
        with pytest.raises(AssertionError):
            osys.propagate_mono([1e-6])

        with pytest.raises(AssertionError):
            osys.propagate_mono(1e-6, [0.])

        # Test propagation
        psf = osys.propagate_mono(4e-6)

        # Test with return WF
        osys.propagate_mono(4e-6, return_wf=True)

        # Test with return_all
        osys.propagate_mono(4e-6, return_all=True)


    def test_propagate(self, create_optics: callable) -> None:
        """
        Tests the propagate method.
        """
        osys = create_optics()

        with pytest.raises(ValueError):
            osys.propagate([1e-6], [0.])

        with pytest.raises(ValueError):
            osys.propagate([1e-6, 2e-6], weights=[0.])

        # Test propagation
        osys.propagate([4e-6, 5e-6])
        osys.propagate([4e-6, 5e-6], weights=np.ones(2))


    def test_model(self, 
            create_optics: callable, 
            create_point_source: callable) -> None:
        """
        Tests the model method
        """
        osys = create_optics()
        psf = osys.model(sources=create_point_source())
    

    def test_getattr(self, create_optics: callable) -> None:
        """
        Tests the __getattr__ method.
        """
        osys = create_optics()

        # Test non-existant attribute
        with pytest.raises(AttributeError):
            osys.non_existant

        # Test existing attribute
        osys.layers

        # Test name referece
        osys.CartesianMFT


class TestSimpleOptics():

    def test_constructor(self, 
        create_simple_optics,
        create_circular_aperture,
        create_aberrated_aperture,
        create_static_aberrated_aperture,
        create_add_opd):
        """
        Tests the constructor
        """
        create_simple_optics()
        create_simple_optics(diameter=1)

        with pytest.raises(ValueError):
            create_simple_optics(diameter=np.array([1]))

        with pytest.raises(TypeError):
            create_simple_optics(diameter="1")

        with pytest.raises(TypeError):
            create_simple_optics(aberrations="1")

        with pytest.raises(TypeError):
            create_simple_optics(aperture="1")
        
        with pytest.raises(TypeError):
            create_simple_optics(propagator="1")

        with pytest.raises(ValueError):
            ap = create_circular_aperture()
            aber_ap = create_aberrated_aperture(aperture=ap)
            aber = create_static_aberrated_aperture(npixels=16, 
                aperture=aber_ap)
            create_simple_optics(aperture=np.ones((8, 8)), aberrations=aber)

        with pytest.raises(ValueError):
            opd_layer = create_add_opd(opd=np.ones((16, 16)))
            create_simple_optics(aperture=np.ones((8, 8)), 
                aberrations=opd_layer)


    def test_propagate(self, create_simple_optics):
        """
        Tests the propagate method
        """
        osys = create_simple_optics()
        psf = osys.propagate([4e-6, 5e-6])
        psf = osys.propagate(4e-6)
    
    def test_propagate_mono(self, create_simple_optics):
        """
        Tests the propagate_mono method
        """
        osys = create_simple_optics()
        psf = osys.propagate_mono(4e-6)
        wf = osys.propagate_mono(4e-6, return_wf=True)

    def test_fresnel_wavefront(self, create_simple_optics, 
        create_cartesian_fresnel):
        """
        Tests the propagate_mono method
        """
        propagator = create_cartesian_fresnel()
        osys = create_simple_optics(propagator=propagator)
        psf = osys.propagate_mono(4e-6)
        wf = osys.propagate_mono(4e-6, return_wf=True)
    
    def test_getattr(self, create_simple_optics):
        """
        Tests the __getattr__ method.
        """
        osys = create_simple_optics()

        # Test non-existant attribute
        with pytest.raises(AttributeError):
            osys.non_existant

        # Test existing attributes
        osys.diameter

        # Test existing attributes
        osys.pixel_scale
        
        # Test name referece
        osys.Propagator


class TestMaskedOptics():

    def test_constructor(self, 
        create_masked_optics,
        create_transmissive_optic,
        create_add_opd,
        create_add_phase):
        """
        Tests the constructor
        """
        create_masked_optics()

        # Test not array or optical layer input
        with pytest.raises(ValueError):
            create_masked_optics(mask='mask')

        # Test wrong array size
        with pytest.raises(ValueError):
            create_masked_optics(mask=np.ones((2, 2)))
        
        # Test wrong shape transmissive optics input
        with pytest.raises(ValueError):
            mask = create_transmissive_optic(np.ones((8, 8)))
            create_masked_optics(mask=mask)
        
        # Test wrong shape AddOPD input
        with pytest.raises(ValueError):
            mask = create_add_opd(np.ones((8, 8)))
            create_masked_optics(mask=mask)
        
        # Test wrong shape AddPhase input
        with pytest.raises(ValueError):
            mask = create_add_phase(np.ones((8, 8)))
            create_masked_optics(mask=mask)


    def test_propagate(self, create_masked_optics):
        """
        Tests the propagate method
        """
        osys = create_masked_optics()
        psf = osys.propagate([4e-6, 5e-6])
        psf = osys.propagate(4e-6)
    

    def test_propagate_mono(self, create_masked_optics):
        """
        Tests the propagate_mono method
        """
        osys = create_masked_optics()
        psf = osys.propagate_mono(4e-6)
        wf = osys.propagate_mono(4e-6, return_wf=True)


class TestDetector(object):
    """
    Tests the Detector class.
    """


    def test_constructor(self, create_detector: callable) -> None:
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            create_detector(layers={})

        # Test list input with non Optics Layer input
        with pytest.raises(AssertionError):
            create_detector(layers=[10.])
    

    def test_getattr(self, create_detector):
        """
        
        """
        detector = create_detector()
        detector.AddConstant


    def test_apply_detector(self, create_detector: callable) -> None:
        """
        Tests the apply_detector method.
        """
        detector = create_detector()

        # Test inputs
        with pytest.raises(AssertionError):
            detector.apply_detector([1e-6])

        # Test 1d input
        with pytest.raises(AssertionError):
            detector.apply_detector(np.array([1.]))

        # Test 3d input
        with pytest.raises(AssertionError):
            detector.apply_detector(np.array([[[1.]]]))

        # Test propagation
        image = detector.apply_detector(np.ones((5, 5)))


    def test_debug_apply_detector(self, create_detector: callable) -> None:
        """
        Tests the debug_apply_detector method.
        """
        detector = create_detector()

        # Test inputs
        with pytest.raises(AssertionError):
            detector.debug_apply_detector([1e-6])

        # Test 1d input
        with pytest.raises(AssertionError):
            detector.debug_apply_detector(np.array([1.]))

        # Test 3d input
        with pytest.raises(AssertionError):
            detector.debug_apply_detector(np.array([[[1.]]]))

        # Test propagation
        image, _, _ = detector.debug_apply_detector(np.ones((5, 5)))


    def test_model(self,
            create_detector: callable,
            create_point_source: callable,
            create_optics: callable) -> None:
        """
        Tests the model method
        """
        detector = create_detector()
        source = create_point_source()
        psf = create_optics().model(sources=source)
        psf = detector.model(psf)


class TestInstrument(object):
    """
    Tests the Optics class.
    """


    def test_constructor(self, create_instrument: callable) -> None:
        """
        Tests the constructor.
        """
        # Test non optics input
        with pytest.raises(ValueError):
            create_instrument(optics=[])

        # Test non detector input
        with pytest.raises(ValueError):
            create_instrument(detector=[])

        # Test non source input
        with pytest.raises(ValueError):
            create_instrument(sources=[1.])
        
        # Test non-observation input
        with pytest.raises(ValueError):
            create_instrument(observation=1.)


    def test_normalise(self, create_instrument: callable) -> None:
        """
        Tests the normalise method.
        """
        # Test all sources in the scene are normalised
        instrument = create_instrument()
        normalised_instrument = instrument.normalise()
        for source in normalised_instrument.sources.values():
            assert np.allclose(source.get_weights().sum(), 1.)
            if hasattr(source, 'get_distribution'):
                assert np.allclose(source.get_distribution(), 1.)
    

    def test_observe(self, create_instrument):
        """
        
        """
        instrument = create_instrument()
        psf = instrument.observe()
    

    def test_getattr(self, create_instrument):
        """
        
        """
        instrument = create_instrument()

        # Test non existent attribute
        with pytest.raises(AttributeError):
            instrument.not_a_attribute
        
        # Test optics attribute
        instrument.CreateWavefront

        # Test source attribute
        instrument.PointSource

        # Test detector attribute
        instrument.AddConstant

        # Test observation attribute
        instrument.Dither


    def test_model(self, 
            create_instrument: callable, 
            create_point_source: callable) -> None:
        """
        Tests the model method.
        """
        instrument = create_instrument()
        sources = [create_point_source()]
        source = create_point_source()

        # Test modelling
        psf = instrument.model()


class TestFilter(object):
    """
    Tests the Filter class.
    """


    def test_constructor(self, create_filter: callable) -> None:
        """
        Tests the constructor.
        """
        # Test adding filter name
        with pytest.raises(NotImplementedError):
            create_filter(filter_name='Test')

        # Test 2d wavelengths input
        with pytest.raises(AssertionError):
            create_filter(2,2)

        # Test 2d throughput input
        with pytest.raises(AssertionError):
            create_filter(2,2)

        # Test different shape wavelengths and throughput
        with pytest.raises(AssertionError):
            create_filter(4)

        # Test negative wavelengths
        with pytest.raises(AssertionError):
            create_filter([-1, 1])

        # Test negative throughputs
        with pytest.raises(AssertionError):
            create_filter([-1, 1])

        # Test throughputs greater than 1
        with pytest.raises(AssertionError):
            create_filter([0, 1.5])

        # Test reverse order wavelengths
        with pytest.raises(AssertionError):
            create_filter([1, 0.5])


    def test_get_throughput(self, create_filter: callable) -> None:
        """
        Test the get_throughput method.
        """
        filt = create_filter()

        # Test scalar input
        throughput = filt.get_throughput(np.array([1e-6, 2e-6]))
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()

        # test array input
        throughput = filt.get_throughput(np.array([1e-6, 2e-6]))
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()

        # Test 1d array inputs
        throughput = filt.get_throughput(1e-6*np.linspace(1, 5, 5))
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()