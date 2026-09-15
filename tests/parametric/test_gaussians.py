"""Tests for continuous Gaussian parametrisations and detector jitter."""

from io import BytesIO

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def coordinates():
    """Return a rectangular grid in physical ``(x, y)`` coordinate order."""
    return dlu.nd_coords((7, 5), (0.5, 0.5))


def relative_gaussian(coordinates, mean, cov):
    """Independently evaluate a peak-relative two-dimensional Gaussian."""
    mean = np.asarray(mean)
    offset = coordinates - mean[:, None, None]
    exponent = np.einsum(
        "i...,ij,j...->...",
        offset,
        np.linalg.inv(np.asarray(cov)),
        offset,
    )
    return np.exp(-exponent / 2)


@pytest.mark.parametrize(
    ("profile", "mean", "cov"),
    [
        (dl.Gaussian(0.8, mean=[0.5, -0.5]), [0.5, -0.5], np.eye(2) * 0.8**2),
        (
            dl.Gaussian([1.2, 0.6], mean=[0.5, -0.5]),
            [0.5, -0.5],
            np.diag(np.asarray([1.2**2, 0.6**2])),
        ),
        (
            dl.CorrelatedGaussian([1.2, 0.6], 0.35, mean=[0.5, -0.5]),
            [0.5, -0.5],
            np.asarray([[1.2**2, 0.35 * 1.2 * 0.6], [0.35 * 1.2 * 0.6, 0.6**2]]),
        ),
        (
            dl.EllipticalGaussian(1.2, ratio=0.5, angle=0.4, mean=[0.5, -0.5]),
            [0.5, -0.5],
            np.asarray(
                [
                    [
                        1.2**2 * np.cos(0.4) ** 2 + 0.6**2 * np.sin(0.4) ** 2,
                        (1.2**2 - 0.6**2) * np.sin(0.4) * np.cos(0.4),
                    ],
                    [
                        (1.2**2 - 0.6**2) * np.sin(0.4) * np.cos(0.4),
                        1.2**2 * np.sin(0.4) ** 2 + 0.6**2 * np.cos(0.4) ** 2,
                    ],
                ]
            ),
        ),
        (
            dl.CorrelatedGaussian.from_cov(
                [[1.4, 0.25], [0.25, 0.7]],
                mean=[0.5, -0.5],
            ),
            [0.5, -0.5],
            np.asarray([[1.4, 0.25], [0.25, 0.7]]),
        ),
    ],
)
def test_gaussian_profile_contract(profile, mean, cov, coordinates):
    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates), profile
    )
    expected = relative_gaussian(coordinates, mean, cov)

    assert output.shape == coordinates.shape[-2:]
    assert np.allclose(output, expected, rtol=1e-5, atol=1e-6)
    assert np.isclose(output.max(), 1.0)


def test_gaussian_parametrisations_are_equivalent(coordinates):
    mean = [0.2, -0.3]
    cov = np.diag(np.asarray([1.2**2, 0.6**2]))
    profiles = [
        dl.Gaussian([1.2, 0.6], mean=mean),
        dl.CorrelatedGaussian([1.2, 0.6], mean=mean),
        dl.EllipticalGaussian(1.2, ratio=0.5, angle=0.0, mean=mean),
        dl.CorrelatedGaussian.from_cov(cov, mean=mean),
    ]
    outputs = [profile.evaluate(coordinates=coordinates) for profile in profiles]

    for output in outputs:
        assert np.isclose(output.max(), 1.0)
    for output in outputs[1:]:
        assert np.allclose(output, outputs[0], rtol=1e-5, atol=1e-6)


def test_widths_and_ellipse_angle_use_physical_xy_order():
    coordinates = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    x_major = dl.Gaussian([2.0, 0.5]).evaluate(coordinates=coordinates)
    ellipse_x = dl.EllipticalGaussian(2.0, ratio=0.25).evaluate(coordinates=coordinates)
    ellipse_y = dl.EllipticalGaussian(2.0, ratio=0.25, angle=np.pi / 2).evaluate(
        coordinates=coordinates
    )

    assert x_major[0, 0] > x_major[0, 1]
    assert np.allclose(ellipse_x, x_major)
    assert np.allclose(ellipse_y, x_major[..., ::-1], atol=1e-6)


def test_correlation_matches_xy_covariance(coordinates):
    stds, corr = np.asarray([1.3, 0.7]), -0.4
    cov = np.asarray(
        [
            [stds[0] ** 2, corr * stds[0] * stds[1]],
            [corr * stds[0] * stds[1], stds[1] ** 2],
        ]
    )
    correlated = dl.CorrelatedGaussian(stds, corr)
    from_cov = dl.CorrelatedGaussian.from_cov(cov)

    assert np.allclose(correlated.cov, cov)
    assert np.allclose(from_cov.stds, stds)
    assert np.allclose(from_cov.corr, corr)
    assert np.allclose(
        correlated.evaluate(coordinates=coordinates),
        from_cov.evaluate(coordinates=coordinates),
        rtol=1e-5,
        atol=1e-6,
    )


def test_from_cov_preserves_batched_covariances():
    cov = np.asarray(
        [
            [[0.8, 0.1], [0.1, 1.2]],
            [[1.4, -0.2], [-0.2, 0.7]],
        ]
    )
    profile = dl.CorrelatedGaussian.from_cov(cov)

    assert profile.stds.shape == (2, 2)
    assert profile.corr.shape == (2,)
    assert np.allclose(profile.cov, cov, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["diagonal", "from_cov"])
def test_gaussian_profiles_broadcast_leading_axes(kind):
    coordinates = dlu.nd_coords((5, 3), (0.4, 0.6))
    coordinates = np.stack((coordinates, coordinates + 0.15))[:, None]
    means = np.asarray([[[0.0, 0.0], [0.2, -0.1], [-0.3, 0.25]]])

    if kind == "diagonal":
        parameters = np.asarray([[[0.7, 1.0], [1.1, 0.6], [0.8, 1.3]]])
        profile = dl.Gaussian(parameters, mean=means)
        constructor = dl.Gaussian
    else:
        parameters = np.asarray(
            [
                [
                    [[0.7, 0.1], [0.1, 1.0]],
                    [[1.1, -0.2], [-0.2, 0.6]],
                    [[0.8, 0.15], [0.15, 1.3]],
                ]
            ]
        )
        profile = dl.CorrelatedGaussian.from_cov(parameters, mean=means)
        constructor = dl.CorrelatedGaussian.from_cov

    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates), profile
    )
    expected = np.stack(
        [
            np.stack(
                [
                    constructor(parameter, mean=mean).evaluate(coordinates=coordinate)
                    for parameter, mean in zip(parameters[0], means[0], strict=True)
                ]
            )
            for coordinate in coordinates[:, 0]
        ]
    )

    assert output.shape == (2, 3, 3, 5)
    assert np.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_zero_diagonal_width_is_a_nearest_sample_delta():
    coordinates = dlu.nd_coords((5, 5))
    circular = dl.Gaussian(0.0).evaluate(coordinates=coordinates)
    x_delta = dl.Gaussian([0.0, 1.0]).evaluate(coordinates=coordinates)
    even_coordinates = dlu.nd_coords((6, 4), (0.2, 0.3))
    even = dl.Gaussian(0.0).evaluate(coordinates=even_coordinates)

    expected = np.zeros((5, 5)).at[2, 2].set(1.0)
    expected_even = np.zeros((4, 6)).at[1:3, 2:4].set(1.0)
    assert np.array_equal(circular, expected)
    assert np.array_equal(even, expected_even)
    assert np.all(x_delta[:, :2] == 0)
    assert np.all(x_delta[:, 3:] == 0)
    assert np.isclose(x_delta[2, 2], 1.0)


@pytest.mark.parametrize(
    ("profile", "path"),
    [
        (dl.Gaussian([0.8, 1.1], mean=[0.1, -0.2]), "std"),
        (
            dl.CorrelatedGaussian([0.8, 1.1], 0.2, mean=[0.1, -0.2]),
            "corr",
        ),
        (
            dl.EllipticalGaussian(1.1, 0.7, 0.3, mean=[0.1, -0.2]),
            "angle",
        ),
    ],
)
def test_gaussian_profile_gradients(profile, path):
    coordinates = dlu.nd_coords((6, 5), (0.4, 0.3), offsets=(0.12, -0.21))
    function = lambda value: profile.set(path, value).evaluate(coordinates=coordinates)

    gradient = assert_differentiable(function, profile.get(path), rtol=1e-5)
    assert np.any(gradient != 0)


@pytest.mark.parametrize(
    ("value", "profile"),
    [
        (0.7, dl.Gaussian(0.7)),
        ([0.7, 1.1], dl.Gaussian([0.7, 1.1])),
        (
            [[0.7, 0.15], [0.15, 1.1]],
            dl.CorrelatedGaussian.from_cov([[0.7, 0.15], [0.15, 1.1]]),
        ),
    ],
)
def test_jitter_numeric_construction_matches_profiles(value, profile):
    numeric = dl.Jitter(value, kernel_size=(7, 5), oversample=(2, 3))
    parametric = dl.Jitter(
        profile=profile,
        kernel_size=(7, 5),
        oversample=(2, 3),
    )

    assert np.allclose(numeric.kernel(), parametric.kernel(), rtol=1e-5, atol=1e-6)


def test_jitter_accepts_nested_parametric_std(make_psf):
    std = dl.Basis(np.ones(1), [0.7])
    parametric = dl.Jitter(dl.Gaussian(std), kernel_size=5, oversample=2)
    expected = dl.Jitter(0.7, kernel_size=5, oversample=2)

    assert np.allclose(parametric.kernel(make_psf()), expected.kernel())


def test_jitter_profile_kernel_contract():
    profile = dl.CorrelatedGaussian([1.2, 0.5], 0.35, mean=[0.2, -0.1])
    jitter = dl.Jitter(
        profile=profile,
        kernel_size=(7, 5),
        oversample=(2, 3),
    )
    kernel = assert_jittable(jitter.kernel, rtol=1e-5, atol=1e-6)
    coordinates = dlu.nd_coords((7, 5))

    assert kernel.shape == (5, 7)
    assert np.allclose(kernel.sum(), 1.0)
    assert np.sum(kernel * coordinates[0] ** 2) > np.sum(kernel * coordinates[1] ** 2)


def test_jitter_normalises_batched_profiles_independently():
    profile = dl.CorrelatedGaussian(
        stds=[[0.6, 1.0], [1.1, 0.5]],
        corr=[0.2, -0.3],
        mean=[[0.0, 0.0], [0.2, -0.1]],
    )
    jitter = dl.Jitter(profile=profile, kernel_size=(7, 5), oversample=(2, 3))
    kernel = assert_jittable(jitter.kernel, rtol=1e-5, atol=1e-6)
    expected = np.stack(
        [
            dl.Jitter(
                profile=dl.CorrelatedGaussian([0.6, 1.0], 0.2),
                kernel_size=(7, 5),
                oversample=(2, 3),
            ).kernel(),
            dl.Jitter(
                profile=dl.CorrelatedGaussian(
                    [1.1, 0.5],
                    -0.3,
                    mean=[0.2, -0.1],
                ),
                kernel_size=(7, 5),
                oversample=(2, 3),
            ).kernel(),
        ]
    )

    assert kernel.shape == (2, 5, 7)
    assert np.allclose(kernel.sum((-2, -1)), 1.0)
    assert np.allclose(kernel, expected, rtol=1e-5, atol=1e-6)


def test_zero_profile_jitter_is_identity():
    jitter = dl.Jitter(profile=dl.Gaussian(0.0), kernel_size=5, oversample=2)
    expected = np.zeros((5, 5)).at[2, 2].set(1.0)

    assert np.allclose(jitter.kernel(), expected)


def test_narrow_positive_jitter_remains_finite():
    jitter = dl.Jitter(profile=dl.Gaussian(1e-3), kernel_size=5, oversample=2)
    expected = np.zeros((5, 5)).at[2, 2].set(1.0)

    kernel = assert_jittable(jitter.kernel)
    assert np.all(np.isfinite(kernel))
    assert np.allclose(kernel, expected)


def test_jitter_profile_archive_roundtrip():
    original = dl.Jitter(
        profile=dl.CorrelatedGaussian([0.7, 1.1], 0.25, mean=[0.1, -0.2]),
        kernel_size=(7, 5),
        oversample=(2, 3),
    )
    file = BytesIO()
    zdx.save(file, original)
    file.seek(0)

    loaded = zdx.load(file)

    assert eqx.tree_equal(loaded, original, typematch=True)
    assert np.allclose(loaded.kernel(), original.kernel())


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Gaussian(-0.1),
        lambda: dl.Gaussian(np.nan),
        lambda: dl.Gaussian(0.5, mean=np.inf),
        lambda: dl.Gaussian([0.5]),
        lambda: dl.Gaussian([0.5, 0.7, 0.9]),
        lambda: dl.Gaussian(0.5, mean=[0.0]),
        lambda: dl.CorrelatedGaussian(0.5),
        lambda: dl.CorrelatedGaussian([0.5]),
        lambda: dl.CorrelatedGaussian([0.0, 0.5]),
        lambda: dl.CorrelatedGaussian([0.5, -0.1]),
        lambda: dl.CorrelatedGaussian([np.inf, 0.5]),
        lambda: dl.CorrelatedGaussian([0.5, 0.7], np.nan),
        lambda: dl.CorrelatedGaussian([0.5, 0.7], -1.0),
        lambda: dl.CorrelatedGaussian([0.5, 0.7], 1.0),
        lambda: dl.CorrelatedGaussian([[0.5, 0.7], [0.8, 0.9]], [0.1, 0.2, 0.3]),
        lambda: dl.EllipticalGaussian(0.0),
        lambda: dl.EllipticalGaussian(1.0, angle=np.inf),
        lambda: dl.EllipticalGaussian(1.0, ratio=0.0),
        lambda: dl.EllipticalGaussian(1.0, ratio=1.1),
        lambda: dl.CorrelatedGaussian.from_cov([1.0, 1.0]),
        lambda: dl.CorrelatedGaussian.from_cov([[1.0, 0.2], [0.1, 1.0]]),
        lambda: dl.CorrelatedGaussian.from_cov([[1.0, 1.0], [1.0, 1.0]]),
        lambda: dl.CorrelatedGaussian.from_cov([[1.0, np.nan], [np.nan, 1.0]]),
        lambda: dl.CorrelatedGaussian.from_cov([[1.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_gaussian_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_jitter_requires_profile():
    with pytest.raises(TypeError):
        dl.Jitter()


@pytest.mark.parametrize(
    ("constructor", "parameters"),
    [
        (dl.Gaussian, {"std": [0.7, 1.1], "mean": [0.1, -0.2]}),
        (
            dl.CorrelatedGaussian,
            {"stds": [0.7, 1.1], "corr": 0.3, "mean": [0.1, -0.2]},
        ),
        (
            dl.EllipticalGaussian,
            {"major": 1.1, "ratio": 0.7, "angle": 0.3, "mean": [0.1, -0.2]},
        ),
    ],
)
def test_expression_parameters_resolve_like_arrays(
    constructor, parameters, coordinates
):
    expected = constructor(**parameters)
    profile = constructor(
        **{name: zdx.Map(value) for name, value in parameters.items()}
    )

    # Missing coordinates retain the profile; preparing children enables properties.
    deferred = profile.resolve()
    prepared = profile.resolve_children()
    assert isinstance(deferred, constructor)
    if constructor is not dl.Gaussian:
        assert np.allclose(prepared.cov, expected.cov)
    if constructor is dl.EllipticalGaussian:
        assert np.allclose(prepared.minor, expected.minor)

    # The complete expression route produces the same sampled profile under JIT.
    output = assert_jittable(
        lambda value: value.resolve(coordinates=coordinates), profile
    )
    assert np.allclose(output, expected.resolve(coordinates=coordinates), atol=1e-6)
