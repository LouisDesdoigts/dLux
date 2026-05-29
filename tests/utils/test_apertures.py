import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import apertures as apertures_utils


# ============================================================================
# Fixtures - Reused parameters across tests
# ============================================================================
@pytest.fixture
def npixels():
    return 32


@pytest.fixture
def diameter():
    return 1.0


@pytest.fixture
def oversample():
    return 3


@pytest.fixture
def rmax():
    return 0.5


@pytest.fixture
def gap():
    return 0.01


@pytest.fixture
def nrings():
    return 2


@pytest.fixture
def segment_diameter():
    return 0.3


@pytest.fixture
def hole_diameter():
    return 0.2


@pytest.fixture
def centers():
    return np.array([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]])


# ============================================================================
# Tests for segmented_hex_cens
# ============================================================================
class TestSegmentedHexCens:
    """Tests for segmented hex center generation."""

    def test_single_ring(self):
        """Single ring returns just the center."""
        cens = apertures_utils.segmented_hex_cens(nrings=1, rmax=0.5)
        assert cens.shape == (1, 2)
        assert np.allclose(cens, np.array([[0.0, 0.0]]))

    def test_two_rings(self):
        """Two rings returns center + 6 neighbors."""
        cens = apertures_utils.segmented_hex_cens(nrings=2, rmax=0.5)
        assert cens.shape == (7, 2)

    def test_three_rings(self):
        """Three rings returns center + 6 + 12 middle points."""
        cens = apertures_utils.segmented_hex_cens(nrings=3, rmax=0.5)
        assert cens.shape == (19, 2)

    def test_with_gap(self, rmax):
        """Gap parameter affects spacing."""
        cens_no_gap = apertures_utils.segmented_hex_cens(nrings=2, rmax=rmax, gap=0.0)
        cens_with_gap = apertures_utils.segmented_hex_cens(
            nrings=2, rmax=rmax, gap=0.01
        )
        # Gap should increase spacing
        assert np.linalg.norm(cens_with_gap[1]) > np.linalg.norm(cens_no_gap[1])

    def test_invalid_nrings(self):
        """nrings < 1 raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            apertures_utils.segmented_hex_cens(nrings=0, rmax=0.5)

    def test_symmetry_two_rings(self):
        """Two ring configuration has 6-fold rotational symmetry."""
        cens = apertures_utils.segmented_hex_cens(nrings=2, rmax=0.5)
        # Center at [0, 0]
        assert np.allclose(cens[0], np.array([0.0, 0.0]))
        # Six neighbors should be roughly equidistant from center
        neighbor_distances = np.linalg.norm(cens[1:], axis=1)
        assert np.allclose(neighbor_distances, neighbor_distances[0])


# ============================================================================
# Tests for non_redundant_support
# ============================================================================
class TestNonRedundantSupport:
    """Tests for non-redundant aperture support mask."""

    def test_no_overlap_unchanged(self):
        """Non-overlapping apertures remain unchanged."""
        apertures = np.zeros((3, 8, 8))
        # Create non-overlapping circles
        apertures = apertures.at[0, 1:3, 1:3].set(1.0)
        apertures = apertures.at[1, 5:7, 1:3].set(1.0)
        apertures = apertures.at[2, 1:3, 5:7].set(1.0)

        support = apertures_utils.non_redundant_support(apertures)
        assert support.shape == apertures.shape
        assert (support <= 1).all()
        assert (support >= 0).all()

    def test_output_no_pixel_overlap(self):
        """No pixel is assigned to multiple segments."""
        apertures = np.zeros((3, 8, 8))
        # Create overlapping circles
        apertures = apertures.at[0, :, :].set(1.0)
        apertures = apertures.at[1, :, :].set(2.0)

        support = apertures_utils.non_redundant_support(apertures)
        # Sum along segment axis - each pixel should be in at most 1 segment
        pixel_sums = support.sum(axis=0)
        assert (pixel_sums <= 1).all()

    def test_output_shape(self):
        """Output has same shape as input."""
        apertures = np.ones((5, 16, 16))
        support = apertures_utils.non_redundant_support(apertures)
        assert support.shape == apertures.shape

    def test_boolean_output(self):
        """Output is boolean."""
        apertures = np.ones((2, 8, 8))
        support = apertures_utils.non_redundant_support(apertures)
        assert support.dtype in [np.bool_, bool]


# ============================================================================
# Tests for circular_aperture
# ============================================================================
class TestCircularAperture:
    """Tests for circular aperture generation."""

    def test_basic_transmission(self, npixels, diameter, oversample):
        """Basic circular aperture returns transmission array."""
        trans = apertures_utils.circular_aperture(
            npixels=npixels, diameter=diameter, oversample=oversample
        )
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_array_diameter(self, npixels, diameter, oversample):
        """Increasing array diameter with npixels constant reduces transmission area."""
        trans_nopad = apertures_utils.circular_aperture(
            npixels=npixels, diameter=diameter, oversample=oversample
        )
        trans_pad = apertures_utils.circular_aperture(
            npixels=npixels,
            diameter=diameter,
            oversample=oversample,
            array_diameter=diameter * 2,
        )
        assert trans_pad.shape == (npixels, npixels)
        # Transmission should decrease with larger array diameter due to padding
        assert trans_pad.sum() < trans_nopad.sum()

    def test_with_secondary(self, npixels, diameter, oversample):
        """Secondary obscuration reduces transmission area."""
        trans_no_sec = apertures_utils.circular_aperture(
            npixels=npixels, diameter=diameter, oversample=oversample
        )
        trans_with_sec = apertures_utils.circular_aperture(
            npixels=npixels,
            diameter=diameter,
            oversample=oversample,
            secondary_diameter=diameter / 3,
        )
        # Transmission should decrease with secondary
        assert trans_with_sec.sum() < trans_no_sec.sum()

    def test_with_spiders(self, npixels, diameter, oversample):
        """Spiders reduce transmission area."""
        trans_no_spider = apertures_utils.circular_aperture(
            npixels=npixels, diameter=diameter, oversample=oversample
        )
        trans_with_spider = apertures_utils.circular_aperture(
            npixels=npixels,
            diameter=diameter,
            oversample=oversample,
            spider_width=0.05,
            spider_angles=[0, 90, 180, 270],
        )
        # Transmission should decrease with spiders
        assert trans_with_spider.sum() < trans_no_spider.sum()

    def test_spider_validation(self, npixels, diameter, oversample):
        """Spider width and angles must both be provided or both None."""
        with pytest.raises(ValueError, match="must both be provided"):
            apertures_utils.circular_aperture(
                npixels=npixels,
                diameter=diameter,
                oversample=oversample,
                spider_width=0.05,
            )

    def test_with_zernike_no_support(self, npixels, diameter, oversample):
        """Zernike basis without support flag returns (transmission, basis)."""
        result = apertures_utils.circular_aperture(
            npixels=npixels,
            diameter=diameter,
            oversample=oversample,
            zernike_nolls=[1, 2, 3],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        trans, basis = result
        assert trans.shape == (npixels, npixels)
        assert basis.shape == (3, npixels, npixels)

    def test_with_zernike_and_support(self, npixels, diameter, oversample):
        """Zernike basis with support flag returns (transmission, basis, support)."""
        result = apertures_utils.circular_aperture(
            npixels=npixels,
            diameter=diameter,
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        trans, basis, support = result
        assert trans.shape == (npixels, npixels)
        assert basis.shape == (2, npixels, npixels)
        assert support.shape == (npixels, npixels)


# ============================================================================
# Tests for segmented_aperture
# ============================================================================
class TestSegmentedAperture:
    """Tests for segmented aperture generation."""

    def test_basic_transmission(self, npixels, diameter, oversample, nrings):
        """Basic segmented aperture returns transmission."""
        trans = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=nrings,
            segment_diameter=0.3,
            oversample=oversample,
        )
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    @pytest.mark.parametrize("nrings_excluded", [0, 1, 2])
    def test_ring_exclusion(self, npixels, diameter, oversample, nrings_excluded):
        """Ring exclusion parameter affects aperture."""
        trans1 = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=3,
            segment_diameter=0.3,
            oversample=oversample,
            nrings_excluded=0,
        )
        trans2 = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=3,
            segment_diameter=0.3,
            oversample=oversample,
            nrings_excluded=nrings_excluded,
        )
        # Higher exclusion should reduce transmission
        if nrings_excluded > 0:
            assert trans2.sum() <= trans1.sum()

    def test_with_secondary(self, npixels, diameter, oversample):
        """Secondary obscuration reduces transmission."""
        trans_no_sec = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=2,
            segment_diameter=0.3,
            oversample=oversample,
        )
        trans_with_sec = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=2,
            segment_diameter=0.3,
            oversample=oversample,
            secondary_diameter=0.2,
        )
        # Transmission should not increase with secondary
        assert trans_with_sec.sum() <= trans_no_sec.sum()

    def test_spider_validation(self, npixels, diameter, oversample):
        """Spider parameters must be consistent."""
        with pytest.raises(ValueError, match="must both be provided"):
            apertures_utils.segmented_aperture(
                npixels=npixels,
                diameter=diameter,
                nrings=2,
                segment_diameter=0.3,
                oversample=oversample,
                spider_width=0.05,
            )

    def test_with_zernike_no_support(self, npixels, diameter, oversample):
        """Zernike basis without support returns (transmission, basis)."""
        result = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=2,
            segment_diameter=0.3,
            oversample=oversample,
            zernike_nolls=[1, 2],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        trans, basis = result
        assert trans.shape == (npixels, npixels)
        # Basis should have (n_segments, n_terms, npixels, npixels)
        assert basis.ndim == 4
        assert basis.shape[2:] == (npixels, npixels)

    def test_with_zernike_and_support(self, npixels, diameter, oversample):
        """Zernike with support flag returns (transmission, basis, support)."""
        result = apertures_utils.segmented_aperture(
            npixels=npixels,
            diameter=diameter,
            nrings=2,
            segment_diameter=0.3,
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        trans, basis, support = result
        assert trans.shape == (npixels, npixels)
        assert basis.ndim == 4
        assert basis.shape[2:] == (npixels, npixels)
        assert support.ndim == 3
        assert support.shape[1:] == (npixels, npixels)


# ============================================================================
# Tests for sparse_aperture
# ============================================================================
class TestSparseAperture:
    """Tests for sparse aperture generation."""

    def test_circular_sparse(
        self, npixels, diameter, oversample, hole_diameter, centers
    ):
        """Circular sparse aperture."""
        trans = apertures_utils.sparse_aperture(
            npixels=npixels,
            diameter=diameter,
            centers=centers,
            hole_diameter=hole_diameter,
            shape="circle",
            oversample=oversample,
        )
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_hex_sparse(self, npixels, diameter, oversample, hole_diameter, centers):
        """Hexagonal sparse aperture."""
        trans = apertures_utils.sparse_aperture(
            npixels=npixels,
            diameter=diameter,
            centers=centers,
            hole_diameter=hole_diameter,
            shape="hex",
            oversample=oversample,
        )
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_invalid_shape(self, npixels, diameter, oversample, hole_diameter, centers):
        """Invalid shape raises error."""
        with pytest.raises(ValueError, match="must be either"):
            apertures_utils.sparse_aperture(
                npixels=npixels,
                diameter=diameter,
                centers=centers,
                hole_diameter=hole_diameter,
                shape="triangle",
                oversample=oversample,
            )

    def test_with_zernike_no_support(
        self, npixels, diameter, oversample, hole_diameter, centers
    ):
        """Zernike without support returns (transmission, basis)."""
        result = apertures_utils.sparse_aperture(
            npixels=npixels,
            diameter=diameter,
            centers=centers,
            hole_diameter=hole_diameter,
            shape="circle",
            oversample=oversample,
            zernike_nolls=[1, 2],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        trans, basis = result
        assert trans.shape == (npixels, npixels)
        # Basis should have one entry per aperture
        assert basis.shape == (len(centers), 2, npixels, npixels)

    def test_with_zernike_and_support(
        self, npixels, diameter, oversample, hole_diameter, centers
    ):
        """Zernike with support returns (transmission, basis, support)."""
        result = apertures_utils.sparse_aperture(
            npixels=npixels,
            diameter=diameter,
            centers=centers,
            hole_diameter=hole_diameter,
            shape="circle",
            oversample=oversample,
            zernike_nolls=[1],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        trans, basis, support = result
        assert trans.shape == (npixels, npixels)
        assert basis.shape == (len(centers), 1, npixels, npixels)
        assert support.shape == (len(centers), npixels, npixels)

    def test_hex_sparse_with_zernike(
        self, npixels, diameter, oversample, hole_diameter, centers
    ):
        """Hexagonal sparse aperture with Zernike basis."""
        result = apertures_utils.sparse_aperture(
            npixels=npixels,
            diameter=diameter,
            centers=centers,
            hole_diameter=hole_diameter,
            shape="hex",
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        trans, basis, support = result
        assert trans.shape == (npixels, npixels)
        assert basis.shape == (len(centers), 2, npixels, npixels)
        assert support.shape == (len(centers), npixels, npixels)


# ============================================================================
# Tests for telescope presets
# ============================================================================
class TestHSTLike:
    """Tests for HST-like aperture preset."""

    def test_basic(self, npixels, oversample):
        """HST-like aperture returns transmission."""
        trans = apertures_utils.hst_like(npixels=npixels, oversample=oversample)
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_with_zernike(self, npixels, oversample):
        """HST-like with Zernike basis."""
        result = apertures_utils.hst_like(
            npixels=npixels, oversample=oversample, zernike_nolls=[1, 2, 3]
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        trans, basis = result
        assert trans.shape == (npixels, npixels)
        assert basis.shape == (3, npixels, npixels)

    def test_with_zernike_and_support(self, npixels, oversample):
        """HST-like with Zernike and support."""
        result = apertures_utils.hst_like(
            npixels=npixels,
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestJWSTLike:
    """Tests for JWST-like aperture preset."""

    def test_basic(self, npixels, oversample):
        """JWST-like aperture returns transmission."""
        trans = apertures_utils.jwst_like(npixels=npixels, oversample=oversample)
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_with_zernike(self, npixels, oversample):
        """JWST-like with Zernike basis."""
        result = apertures_utils.jwst_like(
            npixels=npixels, oversample=oversample, zernike_nolls=[1, 2]
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_with_zernike_and_support(self, npixels, oversample):
        """JWST-like with Zernike and support."""
        result = apertures_utils.jwst_like(
            npixels=npixels,
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestEuclidLike:
    """Tests for Euclid-like aperture preset."""

    def test_basic(self, npixels, oversample):
        """Euclid-like aperture returns transmission."""
        trans = apertures_utils.euclid_like(npixels=npixels, oversample=oversample)
        assert trans.shape == (npixels, npixels)
        assert (trans >= 0).all() and (trans <= 1).all()

    def test_with_zernike(self, npixels, oversample):
        """Euclid-like with Zernike basis."""
        result = apertures_utils.euclid_like(
            npixels=npixels, oversample=oversample, zernike_nolls=[1, 2]
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_with_zernike_and_support(self, npixels, oversample):
        """Euclid-like with Zernike and support."""
        result = apertures_utils.euclid_like(
            npixels=npixels,
            oversample=oversample,
            zernike_nolls=[1, 2],
            return_support=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
