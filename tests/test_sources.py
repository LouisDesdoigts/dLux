from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


class TestSource():
    """
    Tests the Source class.
    """


    def test_constructor(self, create_source : callable) -> None:
        """
        Test the constructor class.
        """
        # Position
        # Test string inputs
        with pytest.raises(ValueError):
            create_source(position="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_source(position=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_source(position=[])

        # Flux
        # Test string inputs
        with pytest.raises(ValueError):
            create_source(flux="")

        # Test 1d dimension input
        with pytest.raises(AssertionError):
            create_source(flux=[1.])

        # Test zero length input
        with pytest.raises(AssertionError):
            create_source(flux=[])

        # Spectrum
        # Test non-spectrum input
        with pytest.raises(AssertionError):
            create_source(spectrum=[])

        # Name
        # Test non-string input
        with pytest.raises(AssertionError):
            create_source(name=[])


    # Getters
    def test_get_position(self, create_source : callable) -> None:
        """
        Tests the get_position method.
        """
        source = create_source()
        assert (source.get_position() == source.position).all()


    def test_get_flux(self, create_source : callable) -> None:
        """
        Tests the get_flux method.
        """
        source = create_source()
        assert (source.get_flux() == source.flux).all()


    def test_get_wavelengths(self, create_source : callable) -> None:
        """
        Tests the get_wavelengths method.
        """
        source = create_source()
        assert (source.get_wavelengths() == source.spectrum.wavelengths).all()


    def test_get_weights(self, create_source : callable) -> None:
        """
        Tests the get_weights method.
        """
        source = create_source()
        assert (source.get_weights() == source.spectrum.weights).all()


class TestResolvedSource():
    """
    Tests the ResolvedSourve class.
    """
    pass


class TestRelativeFluxSource():
    """
    Tests the RelativeFluxSource class.
    """


    def test_constructor(self, create_relative_flux_source : callable) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            create_relative_flux_source(contrast="")

        # Test one dimension input
        with pytest.raises(AssertionError):
            create_relative_flux_source(contrast=[5.])

        # Test zero length input
        with pytest.raises(AssertionError):
            create_relative_flux_source(contrast=[])


    def test_get_flux(self, create_relative_flux_source : callable) -> None:
        """
        Tests the get_flux method.
        """
        source = create_relative_flux_source()
        flux_out = source.get_flux()
        assert flux_out.shape == (2,)
        assert np.allclose(flux_out[0]/flux_out[1], source.contrast)


class TestRelativePositionSource():
    """
    Tests the RelativePositionSource class.
    """


    def test_constructor(self, create_relative_position_source : callable) -> None:
        """
        Tests the constructor.
        """
        # Separation
        # Test string inputs
        with pytest.raises(ValueError):
            create_relative_position_source(separation="")

        # Test one dimension input
        with pytest.raises(AssertionError):
            create_relative_position_source(separation=[5.])

        # Test zero length input
        with pytest.raises(AssertionError):
            create_relative_position_source(separation=[])

        # position_angle
        # Test string inputs
        with pytest.raises(ValueError):
            create_relative_position_source(position_angle="")

        # Test one dimension input
        with pytest.raises(AssertionError):
            create_relative_position_source(position_angle=[5.])

        # Test zero length input
        with pytest.raises(AssertionError):
            create_relative_position_source(position_angle=[])


    def test_get_position(self, create_relative_position_source : callable) -> None:
        """
        Tests the get_position method.
        """
        source = create_relative_position_source()
        position_out = source.get_position()
        sep_vec = position_out[0] - position_out[1]
        separation = np.hypot(sep_vec[0], sep_vec[1])
        position_angle = np.arctan2(sep_vec[0], sep_vec[1])
        assert position_out.shape == (2,2)
        assert np.allclose(source.separation, separation).all()
        assert np.allclose(source.position_angle, position_angle).all()


class TestPointSource():
    """
    Tests the PointSource class.
    """


    def test_model(self, 
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        # TODO: this has no asserts?
        source = create_point_source()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


class TestMultiPointSource():
    """
    Tests the MultiPointSource class.
    """


    def test_constructor(self, create_multi_point_source : callable) -> None:
        """
        Test the constructor class.
        """
        # Position
        # Test string inputs
        with pytest.raises(ValueError):
            create_multi_point_source(position="")

        # Test zero dimension input
        with pytest.raises(AssertionError):
            create_multi_point_source(position=5.)

        # Test zero length input
        with pytest.raises(AssertionError):
            create_multi_point_source(position=[])

        # Test 1 dim input
        with pytest.raises(AssertionError):
            create_multi_point_source(position=np.ones(2))

        # Flux
        # Test string inputs
        with pytest.raises(ValueError):
            create_multi_point_source(flux="")

        # Test zero length input
        with pytest.raises(AssertionError):
            create_multi_point_source(flux=[])

        # Test 2 dim input
        with pytest.raises(AssertionError):
            create_multi_point_source(flux=np.ones((2, 2)))

        # Spectrum
        # Test non-spectrum input
        with pytest.raises(AssertionError):
            create_multi_point_source(spectrum=[])

        # Name
        # Test non-string input
        with pytest.raises(AssertionError):
            create_multi_point_source(name=[])


    def test_model(self, 
        create_multi_point_source : callable,
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        source = create_multi_point_source()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


class TestArrayDistribution():
    """
    Tests the ArrayDistribution class.
    """


    def test_constructor(self, create_array_distribution : callable) -> None:
        """
        Tests the constructor.
        """
        # Test string inputs
        with pytest.raises(ValueError):
            create_array_distribution(distribution="")

        # Test one dimension input
        with pytest.raises(AssertionError):
            create_array_distribution(distribution=[5.])

        # Test zero length input
        with pytest.raises(AssertionError):
            create_array_distribution(distribution=[])


    def test_get_distribution(self, create_array_distribution : callable) -> None:
        """
        Tests the get_distribution method.
        """
        source = create_array_distribution()
        assert (source.get_distribution() == source.distribution).all()


    def test_normalise(self, create_array_distribution : callable) -> None:
        """
        Tests the normalise method.
        """
        source = create_array_distribution()
        new_distribution = np.ones((4, 4))
        new_source = source.set('distribution', new_distribution).normalise()
        assert np.allclose(new_source.distribution.sum(), 1.)


    def test_model(self, 
        create_array_distribution : callable,
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        source = create_array_distribution()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)


class TestBinarySource():
    """
    Tests the BinarySource class.
    """


    def test_model(self, 
        create_binary_source : callable,
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        # TODO add tests
        source = create_binary_source()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)

class TestPointExtendedSource():
    """
    Tests the PointExtendedSource class.
    """

    def test_model(self, 
        create_point_extended_source : callable,
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        source = create_point_extended_source()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)



class TestPointAndExtendedSource():
    """
    Tests the PointAndExtendedSource class.
    """


    def test_constructor(self, create_point_and_extended_source : callable) -> None:
        """
        Tests the constructor.
        """
        # Test non Combined Spectrum input
        with pytest.raises(AssertionError):
            spec = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, 600e-9, 10))
            create_point_and_extended_source(spectrum=spec)


    def test_model(self, 
        create_point_and_extended_source : callable,
        create_point_source : callable, 
        create_optics : callable,
        create_detector) -> None:
        """
        Tests the model method.
        """
        source = create_point_and_extended_source()
        # optics = dLux.core.Optics([dLux.CreateWavefront(16, 1)])
        # detector = dLux.core.Detector([dLux.AddConstant(0.)])
        optics = create_optics()
        detector = create_detector()
        # filter_in = dLux.Filter()
        source.model(optics)
        source.model(optics, detector)
        # source.model(optics, detector, filter_in)
        # source.model(optics, filter_in=filter_in)
