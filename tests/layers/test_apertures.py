from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import (
    CircularAperture,
    RectangularAperture,
    RegPolyAperture,
    Spider,
    SquareAperture,
    AberratedAperture,
    CompoundAperture,
    MultiAperture,
)
from dLux import Wavefront, CoordTransform, DistortedCoords

wf = Wavefront(npixels=16, diameter=1, wavelength=1e-6)


@pytest.fixture
def rmax():
    return 0.1


@pytest.fixture
def n_sides():
    return 6


@pytest.fixture
def softening():
    return 1.0


@pytest.fixture
def angles():
    return np.linspace(0, 2 * np.pi, 6, endpoint=False)


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)


def _test_extent(aperture):
    assert isinstance(aperture.extent, (np.ndarray, float))


def _test_nsides(aperture):
    assert isinstance(aperture.nsides, int)


# Basic tests
@pytest.mark.parametrize("occulting", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
def test_circular_aperture(rmax, occulting, softening, normalise, transformation):
    ap = CircularAperture(rmax, transformation, occulting, softening, normalise)
    _test_apply(ap)
    _test_extent(ap)
    _test_nsides(ap)
    assert ap.softness.shape == ()
    assert ap.radius.shape == ()

    # Small hack here to test softening parameter
    with pytest.raises(ValueError):
        CircularAperture(rmax, softening=-1)


@pytest.mark.parametrize("occulting", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
def test_square_aperture(rmax, occulting, softening, normalise, transformation):
    ap = SquareAperture(rmax, transformation, occulting, softening, normalise)
    _test_apply(ap)
    _test_extent(ap)
    _test_nsides(ap)
    assert ap.width.shape == ()


@pytest.mark.parametrize("occulting", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
def test_rectangular_aperture(rmax, occulting, softening, normalise, transformation):
    ap = RectangularAperture(
        rmax, rmax, transformation, occulting, softening, normalise
    )
    _test_apply(ap)
    _test_extent(ap)
    _test_nsides(ap)
    assert ap.height.shape == ()
    assert ap.width.shape == ()


@pytest.mark.parametrize("occulting", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
def test_reg_poly_aperture(
    rmax, n_sides, occulting, softening, normalise, transformation
):
    ap = RegPolyAperture(n_sides, rmax, transformation, occulting, softening, normalise)
    _test_apply(ap)
    _test_extent(ap)
    _test_nsides(ap)
    assert ap.rmax.shape == ()


@pytest.mark.parametrize("occulting", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
def test_spider(rmax, angles, occulting, softening, normalise, transformation):
    ap = Spider(rmax, angles, transformation, occulting, softening, normalise)
    _test_apply(ap)
    assert ap.width.shape == ()
    with pytest.raises(TypeError):
        _test_extent(ap)
    with pytest.raises(TypeError):
        _test_nsides(ap)


# Testing other functionality
def test_getattr(rmax):
    aperture = CircularAperture(rmax, CoordTransform(translation=[1.0, 1.0]))
    assert np.allclose(aperture.translation, np.array([1.0, 1.0]))
    with pytest.raises(AttributeError):
        aperture.not_an_attr


def test_non_tf(rmax):
    with pytest.raises(TypeError):
        CircularAperture(rmax, transformation=2)


@pytest.mark.parametrize(
    "aperture",
    [CircularAperture(2.4), SquareAperture(4.8)],
)
def test_oversized_aperture_transmission(aperture):
    wavefront = Wavefront(npixels=16, diameter=1, wavelength=1e-6)
    transmission = aperture.transmission(wavefront.coordinates(), wavefront.pixel_scale)
    assert np.allclose(transmission, 1)


# Testing other aperture types separately, as they are more complex


@pytest.fixture
def noll_indices():
    return [1, 2, 3, 4]


@pytest.fixture
def aperture(rmax):
    return CircularAperture(rmax)


@pytest.mark.parametrize("coefficients", [None, np.zeros(4)])
@pytest.mark.parametrize("effect", ["opd", "phase", "amplitude"])
@pytest.mark.parametrize(
    "aperture",
    [
        CircularAperture(0.1, normalise=True),
        CircularAperture(0.1, CoordTransform()),
    ],
)
def test_aberrated_aperture(aperture, noll_indices, coefficients, effect):
    ap = AberratedAperture(aperture, noll_indices, coefficients, effect)
    _test_apply(ap)


def test_aberrated_aperture_error(noll_indices, rmax, angles):
    with pytest.raises(TypeError):
        AberratedAperture(CircularAperture(rmax, occulting=True), noll_indices)
    with pytest.raises(TypeError):
        AberratedAperture(Spider(rmax, angles, occulting=False), noll_indices)
    with pytest.raises(TypeError):
        AberratedAperture(None, noll_indices)
    with pytest.raises(NotImplementedError):
        AberratedAperture(CircularAperture(rmax), noll_indices).calculate()
    with pytest.raises(ValueError, match="effect"):
        AberratedAperture(CircularAperture(rmax), noll_indices, effect="invalid")


def test_aberrated_aperture_transformed_basis(noll_indices):
    aperture = CircularAperture(0.1, transformation=CoordTransform())
    aberrated = AberratedAperture(aperture, noll_indices)

    assert aberrated.calc_basis(wf.coordinates()).shape == (
        len(noll_indices),
        wf.npixels,
        wf.npixels,
    )


def test_aberrated_amplitude_is_perturbation_from_unity(noll_indices):
    aperture = CircularAperture(0.1)
    plain = aperture(wf)
    aberrated = AberratedAperture(aperture, noll_indices, effect="amplitude")(wf)

    assert np.allclose(aberrated.amplitude, plain.amplitude)


@pytest.fixture
def aperture_list(noll_indices, rmax):
    ap = CircularAperture(rmax)
    return [ap, AberratedAperture(ap, noll_indices)]


@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
@pytest.mark.parametrize("normalise", [True, False])
def test_compound_aperture(aperture_list, transformation, normalise, noll_indices):
    ap = CompoundAperture(aperture_list, transformation, normalise)
    _test_apply(ap)

    # Test getattr
    ap.CircularAperture
    with pytest.raises(AttributeError):
        ap.not_an_attr

    # Test too many aberrated apertures
    aber_ap = AberratedAperture(CircularAperture(1), noll_indices)
    with pytest.raises(TypeError):
        CompoundAperture([aber_ap, aber_ap])


def test_compound_aperture_without_aberrations():
    aperture = CompoundAperture([CircularAperture(0.4)])
    _test_apply(aperture)


def test_compound_aperture_normalises_after_aberrations():
    aperture = CircularAperture(2.4)
    aberrated = AberratedAperture(aperture, [2], coefficients=np.array([1e-7]))
    result = CompoundAperture([aberrated], normalise=True)(wf)

    assert np.allclose(result.power, 1)
    assert not np.allclose(result.phase, 0)


@pytest.mark.parametrize("transformation", [None, CoordTransform(), DistortedCoords()])
@pytest.mark.parametrize("normalise", [True, False])
def test_multi_aperture(aperture_list, transformation, normalise, noll_indices):
    ap = MultiAperture(aperture_list, transformation, normalise)
    _test_apply(ap)

    # Test MultiAperture with CompoundAperture
    comp_ap = CompoundAperture(aperture_list, transformation, normalise)
    ap = MultiAperture([comp_ap], transformation, normalise)
    ap._aberrated_apertures()
