import pytest
import jax.numpy as np
from utilities import *
from optax import GradientTransformation


def test_model():
    """
    Test the model function
    """
    optics = OpticsUtility().construct()
    detector = DetectorUtility().construct()
    scene = SceneUtility().construct()
    filter = FilterUtility().construct()
    source = PointSourceUtility().construct()
    sources = [source, source]
    # instrument = InstrumentUtility().construct()
    
    # Test non-optics input
    with pytest.raises(AssertionError):
        dLux.base.model([])
        
    # Test non detector input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, detector=[])
        
    # Test non filter input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, filter=[])
        
    # Test no source inputs
    with pytest.raises(AssertionError):
        dLux.base.model(optics)
        
    # Test scene & source input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, scene=scene, source=source)
        
    # Test scene & sources input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, scene=scene, sources=sources)
        
    # Test sources & source input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, sources=sources, source=source)
        
    # Test non-scene input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, scene=[])
        
    # Test sources with non-source input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, sources=[source, 1.])
        
    # Test source with non-source input
    with pytest.raises(AssertionError):
        dLux.base.model(optics, source=1.)
        
    # Test different combinations of inputs, detector
    out = dLux.base.model(optics, scene=scene, detector=detector)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test different combinations of inputs, filter
    out = dLux.base.model(optics, scene=scene, filter=filter)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test different combinations of inputs, filter and detector
    out = dLux.base.model(optics, scene=scene, detector=detector, filter=filter)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
        
    # Normalisation testing
    # Test normalise function for scene
    out = dLux.base.model(optics, scene=scene)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test normalise function for scene
    out = dLux.base.model(optics, sources=sources)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test normalise function for scene
    out = dLux.base.model(optics, source=source)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test normalise function for scene
    out = dLux.base.model(optics, scene=scene, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test normalise function for scene
    out = dLux.base.model(optics, sources=sources, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test normalise function for scene
    out = dLux.base.model(optics, source=source, normalise=False)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    

    # Test flatten and tree_out combinations, flatten
    out = dLux.base.model(optics, scene=scene, flatten=True)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test flatten and tree_out combinations, return_tree
    out = dLux.base.model(optics, scene=scene, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test flatten and tree_out combinations, return_tree and flatten
    out = dLux.base.model(optics, scene=scene, flatten=True, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test return_tree with different structures, list
    out = dLux.base.model(optics, sources=sources, return_tree=True)
    out = np.array(out)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test return_tree with different structures, dict
    out = dLux.base.model(optics, sources={"source": source}, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test return_tree with different structures, source
    out = dLux.base.model(optics, source=source, return_tree=True)
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()
    
    # Test return_tree with detector input
    out = dLux.base.model(optics, scene=scene, detector=detector, return_tree=True)
    out = np.array(list(out.values()))
    assert not np.isnan(out).all()
    assert not np.isinf(out).all()


class TestBase(UtilityUser):
    """
    Tests the Base class.
    """
    utility : BaseUtility = BaseUtility()
    
    
    def test_get_leaf(self):
        """
        tests the get_leaf method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct(param1, param2)
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        
        assert base.get_leaf(param1_path) == param1
        assert base.get_leaf(param2_path) == param2
        
        assert base.get_leaf('1', path_dict=path_dict) == param1
        assert base.get_leaf('2', path_dict=path_dict) == param2
    
    
    def test_get_leaves(self):
        """
        tests the get_leaves method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct(param1, param2)
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        
        assert base.get_leaves([param1_path, param2_path]) == [param1, param2]
        assert base.get_leaves(['1', '2'], path_dict=path_dict) == \
                                                              [param1, param2]
    
    
    def test_update_leaves(self):
        """
        tests the update_leaves method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        
        # Test paths
        new_base = base.update_leaves([param1_path, param2_path], 
                                      [param1, param2])
        assert new_base.param   == param1
        assert new_base.b.param == param2
        
        # Test path_dict
        new_base = base.update_leaves(['1', '2'], [param1, param2], 
                                      path_dict=path_dict)
        assert new_base.param   == param1
        assert new_base.b.param == param2
    
    
    def test_apply_to_leaves(self):
        """
        tests the apply_to_leaves method
        """
        # Define parameters and construct base
        param1_fn = lambda x: x*5
        param2_fn = lambda x: x*10
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        
        # Test paths
        new_base = base.apply_to_leaves([param1_path, param2_path], 
                                        [param1_fn, param2_fn])
        assert new_base.param   == 5.
        assert new_base.b.param == 10.
        
        # Test path_dict
        new_base = base.apply_to_leaves(['1', '2'], [param1_fn, param2_fn], 
                                      path_dict=path_dict)
        assert new_base.param   == 5.
        assert new_base.b.param == 10.
    
    
    def test_get_filter_spec(self):
        """
        tests the get_filter_spec method
        """
        # Define parameters and construct base
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        
        # Test paths
        filter_spec = base.get_filter_spec([param1_path, param2_path])
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
        
        # Test path_dict
        filter_spec = base.get_filter_spec(['1', '2'], path_dict=path_dict)
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
    
    
    def test_get_param_spec(self):
        """
        tests the get_param_spec method
        """
        # Define parameters and construct base
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        groups = ['group1', 'group2']
        
        # Test paths
        param_spec = base.get_param_spec([param1_path, param2_path], groups)
        assert param_spec.param   == groups[0]
        assert param_spec.b.param == groups[1]
        
        # Test paths with filter_spec
        _, filter_spec = base.get_param_spec([param1_path, param2_path], groups, 
                                             get_filter_spec=True)
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
        
        # Test path_dict
        param_spec = base.get_param_spec(['1', '2'], groups, 
                                                          path_dict=path_dict)
        assert param_spec.param   == groups[0]
        assert param_spec.b.param == groups[1]
        
        # Test path_dict
        _, filter_spec = base.get_param_spec(['1', '2'], groups, 
                                             get_filter_spec=True, 
                                             path_dict=path_dict)
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
    
    
    def test_get_optimiser(self):
        """
        tests the get_optimiser method
        """
        # Define parameters and construct base
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        optimisers = ['group1', 'group2'] # This is arbitrary?
        
        # Test paths
        optim = base.get_optimiser([param1_path, param2_path], optimisers)
        assert isinstance(optim, GradientTransformation)
        
        # Test paths with filter_spec
        _, filter_spec = base.get_optimiser([param1_path, param2_path], 
                                            optimisers, get_filter_spec=True)
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
        
        # Test paths
        optim = base.get_optimiser(['1', '2'], optimisers, path_dict=path_dict)
        assert isinstance(optim, GradientTransformation)
        
        # Test paths with filter_spec
        _, filter_spec = base.get_optimiser(['1', '2'], optimisers, 
                                            path_dict=path_dict, 
                                            get_filter_spec=True)
        assert filter_spec.param   == True
        assert filter_spec.b.param == True
    
    
    def test_update_and_model(self):
        """
        tests the update_and_model method
        """
        # Define parameters and construct base
        param1 = 2.
        param2 = 4.
        base = self.utility.construct()
        
        # Define paths and path_dict
        param1_path = ['param']
        param2_path = ['b', 'param']
        path_dict = {'1': param1_path,
                     '2': param2_path}
        values = [param1, param2]
        
        # Test paths
        out = base.update_and_model("model", [param1_path, param2_path], values)
        assert out == param1**2 + param2**2
        
        # Test paths with filter_spec
        out = base.update_and_model("model", ['1', '2'], values, 
                                    path_dict=path_dict)
        assert out == param1**2 + param2**2


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
            
        # Test order outside of (1, 2, 3)
        with pytest.raises(AssertionError):
            self.utility.construct(order=4)
    
    
    def test_get_throughput(self):
        """
        Test the get_throughput method.
        """
        filt = self.utility.construct()
        
        # Test scalar input
        throughput = filt.get_throughput(1e-6)
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()
        
        # test array input
        throughput = filt.get_throughput(np.array(1e-6))
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
        
        # Test scalar input
        throughput = filt.get_throughput(1e-6, integrate=False)
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()
        
        # test array input
        throughput = filt.get_throughput(np.array(1e-6), integrate=False)
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()
        
        # Test 1d array inputs
        throughput = filt.get_throughput(1e-6*np.linspace(1, 5, 5), integrate=False)
        assert not np.isnan(throughput).all()
        assert not np.isinf(throughput).all()
        assert (throughput >= 0).all()
        assert (throughput <= 1).all()
    
    
    def test_model(self):
        """
        Tests the model method
        """
        filter_in = self.utility.construct()
        scene = SceneUtility().construct()
        psf = filter_in.model(OpticsUtility().construct(), scene=scene)
        assert not np.isnan(psf).all()
        assert not np.isinf(psf).all()


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