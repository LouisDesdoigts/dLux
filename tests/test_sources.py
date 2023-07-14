import jax.numpy as np
import pytest
from jax import config

config.update("jax_debug_nans", True)


def _test_source_constructor(constructor, spectrum_constructor):
    """Tests the consturctor for source classes"""
    constructor()
    constructor(spectrum=spectrum_constructor())
    with pytest.raises(ValueError):
        constructor(position=[1, 2, 3])
    with pytest.raises(ValueError):
        constructor(flux=[1])
    with pytest.raises(ValueError):
        constructor(spectrum="spectrum")


def _test_resolved_source_constructor(constructor):
    """tests the constructor for resolved source classes"""
    with pytest.raises(ValueError):
        constructor(distribution=[1])


def _test_normalise(constructor):
    """Tests the normalise method for source classes"""
    constructor().normalise()


def _test_model(constructor, optics_constructor):
    """Tests the model method for source classes"""
    constructor().model(optics_constructor())


def _test_rel_pos_constructor(constructor):
    """Tests the constructor for relative position classes"""
    with pytest.raises(ValueError):
        constructor(separation=[1])
    with pytest.raises(ValueError):
        constructor(position_angle=[1])


def _test_rel_flux_constructor(constructor):
    """Tests the constructor for relative flux classes"""
    with pytest.raises(ValueError):
        constructor(contrast=[1])


class TestPointSource:
    """Tests the Source class."""

    def test_constructor(self, create_point_source, create_spectrum):
        """Test the constructor class."""
        create_point_source()
        _test_source_constructor(create_point_source, create_spectrum)

    def test_normalise(self, create_point_source):
        """Test the normalise method."""
        _test_normalise(create_point_source)

    def test_model(self, create_point_source, create_angular_optics):
        """Test the model method."""
        _test_model(create_point_source, create_angular_optics)


class TestPointSources:
    """Tests the Sources class."""

    def test_constructor(self, create_point_sources, create_spectrum):
        """Test the constructor class."""
        create_point_sources()
        _test_source_constructor(create_point_sources, create_spectrum)
        create_point_sources(flux=None)
        with pytest.raises(ValueError):
            create_point_sources(flux=np.ones((2, 2)))

    def test_normalise(self, create_point_sources):
        """Test the normalise method."""
        _test_normalise(create_point_sources)

    def test_model(self, create_point_sources, create_angular_optics):
        """Test the model method."""
        _test_model(create_point_sources, create_angular_optics)


class TestResolvedSource:
    """Tests the ResolvedSource class."""

    def test_constructor(self, create_resolved_source, create_spectrum):
        """Test the constructor class."""
        create_resolved_source()
        _test_source_constructor(create_resolved_source, create_spectrum)
        _test_resolved_source_constructor(create_resolved_source)

    def test_normalise(self, create_resolved_source):
        """Test the normalise method."""
        _test_normalise(create_resolved_source)

    def test_model(self, create_resolved_source, create_angular_optics):
        """Test the model method."""
        _test_model(create_resolved_source, create_angular_optics)


class TestBinarySource:
    """Tests the BinarySource class."""

    def test_constructor(self, create_binary_source, create_spectrum):
        """Test the constructor class."""
        create_binary_source()
        _test_source_constructor(create_binary_source, create_spectrum)
        _test_rel_pos_constructor(create_binary_source)
        _test_rel_flux_constructor(create_binary_source)

    def test_normalise(self, create_binary_source):
        """Test the normalise method."""
        _test_normalise(create_binary_source)

    def test_model(self, create_binary_source, create_angular_optics):
        """Test the model method."""
        _test_model(create_binary_source, create_angular_optics)


class TestPointResolvedSource:
    """Tests the PointResolvedSource class."""

    def test_constructor(self, create_point_resolved_source, create_spectrum):
        """Test the constructor class."""
        create_point_resolved_source()
        _test_source_constructor(create_point_resolved_source, create_spectrum)
        _test_rel_flux_constructor(create_point_resolved_source)
        _test_resolved_source_constructor(create_point_resolved_source)

    def test_normalise(self, create_point_resolved_source):
        """Test the normalise method."""
        _test_normalise(create_point_resolved_source)

    def test_model(self, create_point_resolved_source, create_angular_optics):
        """Test the model method."""
        _test_model(create_point_resolved_source, create_angular_optics)
