from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


def _test_apertured_optics_constructor(constructor):
    """Tests the constructor of the AperturedOptics class"""
    constructor()
    with pytest.raises(TypeError):
        constructor(aperture="1")
    with pytest.raises(TypeError):
        constructor(mask="1")


def _test_propagate(optics):
    """Tests the propagate method and its sub function"""
    # Test wavelengths
    optics.propagate(1e-6)
    optics.propagate(np.array(1e-6))
    optics.propagate([1e-6, 2e-6])

    # Test weights
    optics.propagate(1e-6, weights=1e-6)
    optics.propagate(1e-6, weights=np.array(1e-6))
    with pytest.raises(ValueError):
        optics.propagate(1e-6, weights=np.array([1e-6, 2e-6]))

    # Test offset
    optics.propagate(wavelengths=1e-6, offset=[0, 0])
    with pytest.raises(ValueError):
        optics.propagate(1e-6, weights=np.array([1e-6, 2e-6]))


def _test_propagate_mono(optics):
    """Tests the propagate method and its sub function"""
    # Test wavelengths
    optics.propagate_mono(1e-6)
    optics.propagate_mono(1e-6, return_wf=True)


def _test_model(optics, source):
    """Tests the source input type checking of the BaseOptics.model method"""
    optics.model(sources=source)
    optics.model(sources=[source])
    with pytest.raises(TypeError):
        optics.model(sources=1)


class TestAngularOptics():
    """Tests the AngularOptics class."""

    def test_constructor(self, create_angular_optics):
        """Tests the constructor"""
        _test_apertured_optics_constructor(create_angular_optics)

    def test_propagate(self, create_angular_optics):
        """Tests the propagate method"""
        _test_propagate(create_angular_optics())

    def test_model(self, create_angular_optics, create_point_source):
        """Tests the model method"""
        _test_model(create_angular_optics(), create_point_source())
    
    def test_propagate_mono(self, create_angular_optics):
        """Tests the propagate_mono method"""
        _test_propagate_mono(create_angular_optics())


class TestCartesianOptics():
    """Tests the CartesianOptics class."""

    def test_constructor(self, create_cartesian_optics):
        """Tests the constructor"""
        _test_apertured_optics_constructor(create_cartesian_optics)
        with pytest.raises(TypeError):
            create_cartesian_optics(focal_length=[1.])

    def test_propagate(self, create_angular_optics):
        """Tests the propagate method"""
        _test_propagate(create_angular_optics())

    def test_model(self, create_cartesian_optics, create_point_source):
        """Tests the model method"""
        _test_model(create_cartesian_optics(), create_point_source())
    
    def test_propagate_mono(self, create_cartesian_optics):
        """Tests the propagate_mono method"""
        _test_propagate_mono(create_cartesian_optics())


class TestFlexibleOptics():
    """Tests the FlexibleOptics class."""

    def test_constructor(self, create_flexible_optics, create_mft):
        """Tests the constructor"""
        _test_apertured_optics_constructor(create_flexible_optics)
        with pytest.raises(TypeError):
            create_flexible_optics(propagator=1.)

    def test_propagate(self, create_flexible_optics):
        """Tests the propagate method"""
        _test_propagate(create_flexible_optics())

    def test_model(self, create_flexible_optics, create_point_source):
        """Tests the model method"""
        _test_model(create_flexible_optics(), create_point_source())
    
    def test_propagate_mono(self, create_flexible_optics):
        """Tests the propagate_mono method"""
        _test_propagate_mono(create_flexible_optics())


class TestLayeredOptics():
    """Tests the LayeredOptics class."""

    def test_constructor(self, create_layered_optics, create_mft):
        """Tests the constructor"""
        _test_apertured_optics_constructor(create_layered_optics)

    def test_propagate(self, create_layered_optics):
        """Tests the propagate method"""
        _test_propagate(create_layered_optics())

    def test_model(self, create_layered_optics, create_point_source):
        """Tests the model method"""
        _test_model(create_layered_optics(), create_point_source())
    
    def test_propagate_mono(self, create_layered_optics):
        """Tests the propagate_mono method"""
        _test_propagate_mono(create_layered_optics())