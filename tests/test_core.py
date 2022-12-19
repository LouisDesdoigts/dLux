from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from test_sources import PointSourceUtility
from jax import config
config.update("jax_debug_nans", True)

Array = np.ndarray

def test_model(
        create_optics: callable,
        create_detector: callable,
        create_scene: callable,
        create_filter: callable, 
        create_source: callable) -> None:
    """
    Test the model function
    """
    optics = create_optics()
    detector = create_detector()
    scene = create_scene()
    filter = create_filter()
    source = create_pointsource()
    sources = [source, source]

    # Test non-optics input
    with pytest.raises(AssertionError):
        dLux.core.model([])

    # Test non detector input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, detector=[])

    # Test non filter input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, filter=[])

    # Test no source inputs
    with pytest.raises(AssertionError):
        dLux.core.model(optics)

    # Test scene & source input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, scene=scene, source=source)

    # Test scene & sources input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, scene=scene, sources=sources)

    # Test sources & source input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, sources=sources, source=source)

    # Test non-scene input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, scene=[])

    # Test sources with non-source input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, sources=[source, 1.])

    # Test source with non-source input
    with pytest.raises(AssertionError):
        dLux.core.model(optics, source=1.)

    # Test different combinations of inputs, detector
    out = dLux.core.model(optics, scene=scene, detector=detector)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

#     # Test different combinations of inputs, filter
#     out = dLux.core.model(optics, scene=scene, filter=filter)
#     assert not np.isnan(out).all()
#     assert not np.isinf(out).all()

#     # Test different combinations of inputs, filter and detector
#     out = dLux.core.model(optics, scene=scene, detector=detector, filter=filter)
#     assert not np.isnan(out).all()
#     assert not np.isinf(out).all()

    # Normalisation testing
    # Test normalise function for scene
    out = dLux.core.model(optics, scene=scene)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test normalise function for scene
    out = dLux.core.model(optics, sources=sources)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test normalise function for scene
    out = dLux.core.model(optics, source=source)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test normalise function for scene
    out = dLux.core.model(optics, scene=scene, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test normalise function for scene
    out = dLux.core.model(optics, sources=sources, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test normalise function for scene
    out = dLux.core.model(optics, source=source, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test flatten and tree_out combinations, flatten
    out = dLux.core.model(optics, scene=scene, flatten=True)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test flatten and tree_out combinations, return_tree
    out = dLux.core.model(optics, scene=scene, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test flatten and tree_out combinations, return_tree and flatten
    out = dLux.core.model(optics, scene=scene, flatten=True, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test return_tree with different structures, list
    out = dLux.core.model(optics, sources=sources, return_tree=True)
    out = np.array(out)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test return_tree with different structures, dict
    out = dLux.core.model(optics, sources={"source": source}, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test return_tree with different structures, source
    out = dLux.core.model(optics, source=source, return_tree=True)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()

    # Test return_tree with detector input
    out = dLux.core.model(optics, scene=scene, detector=detector, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()


class TestOptics(object):
    """
    Tests the Optics class.
    """


    def test_constructor(self, create_optics: callable) -> None:
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            create_optics(layers={})

        # Test list input with non Optics Layer input
        with pytest.raises(AssertionError):
            create_optics(layers=[10.])


    def test_propagate_mono(self, create_optic: callable) -> None:
        """
        Tests the propagate_mono method.
        """
        osys = create_optic() 

        # Test inputs
        with pytest.raises(AssertionError):
            osys.propagate_mono([1e-6])

        with pytest.raises(AssertionError):
            osys.propagate_mono(1e-6, [0.])

        with pytest.raises(AssertionError):
            osys.propagate_mono(1e-6, weight=[0.])

        # Test propagation
        psf = osys.propagate_mono(4e-6)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


    def test_propagate_multi(self, create_optic: callable) -> None:
        """
        Tests the propagate_multi method.
        """
        osys = create_optic()

        # Test inputs
        with pytest.raises(AssertionError):
            osys.propagate_multi(1e-6)

        with pytest.raises(AssertionError):
            osys.propagate_multi([[1e-6]])

        with pytest.raises(AssertionError):
            osys.propagate_multi([1e-6], [0.])

        with pytest.raises(AssertionError):
            osys.propagate_multi([1e-6, 2e-6], weights=[0.])

        # Test propagation
        psf = osys.propagate_multi([4e-6, 5e-6])
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


    def test_debug_prop(self, create_optic: callable) -> None:
        """
        Tests the debug_prop method.
        """
        osys = create_optic()

        # Test inputs
        with pytest.raises(AssertionError):
            osys.debug_prop([1e-6])

        with pytest.raises(AssertionError):
            osys.debug_prop(1e-6, [0.])

        with pytest.raises(AssertionError):
            osys.debug_prop(1e-6, weight=[0.])

        # Test propagation
        psf, _, _ = osys.debug_prop(4e-6)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


    def test_model(self, 
            create_optic: callable, 
            create_point_source: callable) -> None:
        """
        Tests the model method
        """
        osys = create_optic()
        psf = osys.model(source=create_point_source())
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


class TestScene(object):
    """
    Tests the Scene class.
    """


    def test_constructor(self, create_scene: callable) -> None:
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            create_scene(sources={})

        # Test list input with non Source input
        with pytest.raises(AssertionError):
            create_scene(sources=[10.])


    def test_normalise(self, create_scene: callable) -> None:
        """
        Tests the normalise method.
        """
        # Test all sources in the scene are normalised
        scene = create_scene()
        normalised_scene = scene.normalise()
        for source in normalised_scene.sources.values():
            assert np.allclose(source.get_weights().sum(), 1.)
            if hasattr(source, 'get_distribution'):
                assert np.allclose(source.get_distribution(), 1.)


    def test_model(self, 
            create_scene: callable, 
            create_optics: callable) -> None:
        """
        Tests the model method
        """
        scene = create_scene()
        psf = scene.model(create_optics())
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


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
        assert not np.isnan(image).all()
        assert not np.isinf(image).all()


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
        assert not np.isnan(image).all()
        assert not np.isinf(image).all()


    def test_model(self,
            create_detector: callable,
            create_scene: callable,
            create_optics: callable) -> None:
        """
        Tests the model method
        """
        detector = create_detector()
        scene = create_scene()
        psf = detector.model(create_optics(), scene=scene)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


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
            create_filter(2,2)))

        # Test 2d throughput input
        with pytest.raises(AssertionError):
            create_filter(2,2)))

        # Test different shape wavelengths and throughput
        with pytest.raises(AssertionError):
            create_filter(4))

        # Test negative wavelengths
        with pytest.raises(AssertionError):
            create_filter([-1, 1]))

        # Test negative throughputs
        with pytest.raises(AssertionError):
            create_filter([-1, 1]))

        # Test throughputs greater than 1
        with pytest.raises(AssertionError):
            create_filter([0, 1.5]))

        # Test reverse order wavelengths
        with pytest.raises(AssertionError):
            create_filter([1, 0.5]))


    def test_get_throughput(self, create_filter: callable) -> None:
        """
        Test the get_throughput method.
        """
        filt = create_filter()

        # Test scalar input
        throughput = filt.get_throughput(np.array([1e-6, 2e-6]))
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()

        # test array input
        throughput = filt.get_throughput(np.array([1e-6, 2e-6]))
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()

        # Test 1d array inputs
        throughput = filt.get_throughput(1e-6*np.linspace(1, 5, 5))
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()


    # def test_model(self):
    #     """
    #     Tests the model method
    #     """
    #     filter_in = self.utility.construct()
    #     scene = SceneUtility().construct()
    #     psf = filter_in.model(OpticsUtility().construct(), scene=scene)
    #     assert not np.isnan(psf).all()
    #     assert not np.isinf(psf).all()


class TestInstrument(object):
    """
    Tests the Optics class.
    """


    def test_constructor(self, create_instrument: callable) -> None:
        """
        Tests the constructor.
        """
        # Test optic and optical_layers input
        with pytest.raises(ValueError):
            create_instrument(optics=[], optical_layers=[], input_both=True)

        # Test detector and detector_layers input
        with pytest.raises(ValueError):
            create_instrument(detector=[], detector_layers=[], input_both=True)

        # Test scene and sources input
        with pytest.raises(ValueError):
            create_instrument(scene=[], sources=[], input_both=True)

        # Test non optics input
        with pytest.raises(AssertionError):
            create_instrument(optics=[])

        # Test non detector input
        with pytest.raises(AssertionError):
            create_instrument(detector=[])

        # Test non scene input
        with pytest.raises(AssertionError):
            create_instrument(scene=[])

        # Test non filter input
        with pytest.raises(AssertionError):
            create_instrument(filter=[])

        # Test non list optical_layers input
        with pytest.raises(AssertionError):
            create_instrument(optical_layers={}, input_layers=True)

        # Test non list detector_layers input
        with pytest.raises(AssertionError):
            create_instrument(detector_layers={}, input_layers=True)

        # Test non list sources input
        with pytest.raises(AssertionError):
            create_instrument(sources={}, input_layers=True)


    def test_normalise(self, create_instrument: callable) -> None:
        """
        Tests the normalise method.
        """
        # Test all sources in the scene are normalised
        instrument = create_instrument()
        normalised_instrument = instrument.normalise()
        for source in normalised_instrument.scene.sources.values():
            assert np.allclose(source.get_weights().sum(), 1.)
            if hasattr(source, 'get_distribution'):
                assert np.allclose(source.get_distribution(), 1.)


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
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()

        # Test input sources
        psf = instrument.model(sources=sources)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()

        # Test input source
        psf = instrument.model(source=source)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()
