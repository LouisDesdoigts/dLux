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


class TestOptics(UtilityUser):
    """
    Tests the Optics class.
    """
    utility : OpticsUtility = OpticsUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            self.utility.construct(layers={})

        # Test list input with non Optics Layer input
        with pytest.raises(AssertionError):
            self.utility.construct(layers=[10.])


    def test_propagate_mono(self):
        """
        Tests the propagate_mono method.
        """
        osys = self.utility.construct()

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


    def test_propagate_multi(self):
        """
        Tests the propagate_multi method.
        """
        osys = self.utility.construct()

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


    def test_debug_prop(self):
        """
        Tests the debug_prop method.
        """
        osys = self.utility.construct()

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


    def test_model(self):
        """
        Tests the model method
        """
        osys = self.utility.construct()
        psf = osys.model(source=PointSourceUtility().construct())
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


class TestScene(UtilityUser):
    """
    Tests the Scene class.
    """
    utility : SceneUtility = SceneUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            self.utility.construct(sources={})

        # Test list input with non Source input
        with pytest.raises(AssertionError):
            self.utility.construct(sources=[10.])


    def test_normalise(self):
        """
        Tests the normalise method.
        """
        # Test all sources in the scene are normalised
        scene = self.utility.construct()
        normalised_scene = scene.normalise()
        for source in normalised_scene.sources.values():
            assert np.allclose(source.get_weights().sum(), 1.)
            if hasattr(source, 'get_distribution'):
                assert np.allclose(source.get_distribution(), 1.)


    def test_model(self):
        """
        Tests the model method
        """
        scene = self.utility.construct()
        psf = scene.model(OpticsUtility().construct())
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


class TestDetector(UtilityUser):
    """
    Tests the Detector class.
    """
    utility : DetectorUtility = DetectorUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test non-list inputs
        with pytest.raises(AssertionError):
            self.utility.construct(layers={})

        # Test list input with non Optics Layer input
        with pytest.raises(AssertionError):
            self.utility.construct(layers=[10.])


    def test_apply_detector(self):
        """
        Tests the apply_detector method.
        """
        detector = self.utility.construct()

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


    def test_debug_apply_detector(self):
        """
        Tests the debug_apply_detector method.
        """
        detector = self.utility.construct()

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


    def test_model(self):
        """
        Tests the model method
        """
        detector = self.utility.construct()
        scene = SceneUtility().construct()
        psf = detector.model(OpticsUtility().construct(), scene=scene)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


class TestFilter(UtilityUser):
    """
    Tests the Filter class.
    """
    utility : FilterUtility = FilterUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test adding filter name
        with pytest.raises(NotImplementedError):
            self.utility.construct(filter_name='Test')

        # Test 2d wavelengths input
        with pytest.raises(AssertionError):
            self.utility.construct(np.ones((2,2)))

        # Test 2d throughput input
        with pytest.raises(AssertionError):
            self.utility.construct(throughput=np.ones((2,2)))

        # Test different shape wavelengths and throughput
        with pytest.raises(AssertionError):
            self.utility.construct(np.ones(5), np.ones(4))

        # Test negative wavelengths
        with pytest.raises(AssertionError):
            self.utility.construct(np.array([-1, 1]))

        # Test negative throughputs
        with pytest.raises(AssertionError):
            self.utility.construct(throughput=np.array([-1, 1]))

        # Test throughputs greater than 1
        with pytest.raises(AssertionError):
            self.utility.construct(throughput=np.array([0, 1.5]))

        # Test reverse order wavelengths
        with pytest.raises(AssertionError):
            self.utility.construct(wavelengths=np.array([1, 0.5]))


    def test_get_throughput(self):
        """
        Test the get_throughput method.
        """
        filt = self.utility.construct()

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


class TestInstrument(UtilityUser):
    """
    Tests the Optics class.
    """
    utility : InstrumentUtility = InstrumentUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test optic and optical_layers input
        with pytest.raises(ValueError):
            self.utility.construct(optics=[], optical_layers=[], 
                                   input_both=True)

        # Test detector and detector_layers input
        with pytest.raises(ValueError):
            self.utility.construct(detector=[], detector_layers=[], 
                                   input_both=True)

        # Test scene and sources input
        with pytest.raises(ValueError):
            self.utility.construct(scene=[], sources=[], input_both=True)

        # Test non optics input
        with pytest.raises(AssertionError):
            self.utility.construct(optics=[])

        # Test non detector input
        with pytest.raises(AssertionError):
            self.utility.construct(detector=[])

        # Test non scene input
        with pytest.raises(AssertionError):
            self.utility.construct(scene=[])

        # Test non filter input
        with pytest.raises(AssertionError):
            self.utility.construct(filter=[])

        # Test non list optical_layers input
        with pytest.raises(AssertionError):
            self.utility.construct(optical_layers={}, input_layers=True)

        # Test non list detector_layers input
        with pytest.raises(AssertionError):
            self.utility.construct(detector_layers={}, input_layers=True)

        # Test non list sources input
        with pytest.raises(AssertionError):
            self.utility.construct(sources={}, input_layers=True)


    def test_normalise(self):
        """
        Tests the normalise method.
        """
        # Test all sources in the scene are normalised
        instrument = self.utility.construct()
        normalised_instrument = instrument.normalise()
        for source in normalised_instrument.scene.sources.values():
            assert np.allclose(source.get_weights().sum(), 1.)
            if hasattr(source, 'get_distribution'):
                assert np.allclose(source.get_distribution(), 1.)


    def test_model(self):
        """
        Tests the model method.
        """
        instrument = self.utility.construct()
        sources = [PointSourceUtility().construct()]
        source = PointSourceUtility().construct()

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
