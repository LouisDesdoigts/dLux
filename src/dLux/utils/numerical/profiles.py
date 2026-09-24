"""Sample spatial densities and displacement profiles into pixel masses."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array, lax
from jax.nn import softmax

from .. import grids
from ..batching import as_axis
from .interpolation import Interp
from .units import Unitful, unit_factor, unit_dimension
from .gaussians import Gaussian, CorrelatedGaussian, log_gaussian

__all__ = ["Profile", "Jitter"]


class Profile(zdx.Module):
    """A spatial density sampled and integrated on a two-dimensional grid.

    The optional ``grid`` defines input sampling for arrays or expressions. Without
    one, expressions calculate directly on the requested output grid.
    ``sample`` returns pixel masses, normalised to unit sum by default. Only the
    final two spatial axes are integrated and normalised; earlier axes are batches.
    Grids supply SI coordinates, including their centre, without recentering.
    Ordinary resolution prepares stored parameters and retains this sampling owner.
    """

    profile: Array | zdx.Expression
    grid: grids.Grid | None = None
    oversample: tuple[int, int] = (1, 1)
    normalise: bool = True

    def __init__(self, profile, grid=None, oversample=1, normalise=True, *, alias=None):
        """Store sampled densities or a coordinate-dependent expression.

        ``grid`` is required for arrays. A Field supplies both its array and grid.
        Expression parameters use the grid's SI coordinate units. ``oversample``
        gives subpixel integration factors in physical (x, y) order. With
        ``normalise=False``, output masses retain the density's scale and include
        the SI pixel area. Density samples must be non-negative with nonzero sum;
        no clipping or runtime value checks are added.
        """
        # Unpack sampled Fields; raw density arrays must carry their own source grid.
        if isinstance(profile, grids.Field):
            if grid is not None:
                raise ValueError("A Field already supplies the profile grid.")
            profile, grid = profile.array, profile.grid
        profile = zdx.as_array(profile, dtype=float)
        if grid is not None and not isinstance(grid, grids.Grid):
            raise TypeError("grid must be a Grid or None.")
        if not isinstance(profile, zdx.Expression):
            if profile is None or profile.ndim < 2:
                raise ValueError("A sampled profile requires final (ny, nx) axes.")
            if grid is None:
                raise ValueError("A sampled profile requires a grid.")
        self.profile = profile
        self.grid = grid
        self.oversample = grids.as_size(oversample, 2, "oversample")
        self.normalise = bool(normalise)
        self.alias = alias

    def sample(self, grid=None, *, wavelengths=None, **context) -> Array:
        """Return spatial masses on a supplied grid, or the stored default grid.

        Stored sampling is interpolated using Interp's cubic spline and zero outside
        its support. Expressions receive local ``coordinates``,
        ``pixel_scale`` and any supplied wavelength or other context. Normalisation
        is independent for every leading batch, including wavelength batches.
        """
        target = self.grid if grid is None else grid
        if target is None:
            raise zdx.Unresolved("Profile sampling requires a grid.")
        if not isinstance(target, grids.Grid):
            raise TypeError("grid must be a Grid.")
        if wavelengths is not None:
            context = dict(context, wavelengths=wavelengths)

        # A stored grid also supports array-producing expressions such as Basis:
        # evaluate there first, then interpolate onto the requested output sampling.
        source = None if self.grid is None else self.grid.resolve(**context)
        if source is not None and not isinstance(self.profile, zdx.Expression):
            source = source.match_shape(self.profile.shape[-2:])
        native = (grid is None or grid is self.grid) and self.oversample == (1, 1)
        target = source if source is not None and grid is None else target
        target = target.resolve(**context)

        # Both source and target sampling must be complete before evaluating a density.
        for sampling in (target, source):
            if sampling is None:
                continue
            missing = sampling.n is None or sampling.d is None
            if missing or isinstance(sampling.d, zdx.Expression):
                raise zdx.Unresolved(
                    "Profile sampling requires resolved grid counts and spacing."
                )
        fine = target.oversample(self.oversample)
        sampling = fine if source is None else source
        local = dict(
            context, coordinates=sampling.coordinates, pixel_scale=sampling.d_si
        )

        # Gaussian log profiles retain narrow and zero-width definitions that
        # would underflow if converted to density samples before normalising.
        if self.normalise:
            density = self._normalise(**local)
        else:
            density = zdx.resolve(self.profile, **local)

        if isinstance(density, zdx.Expression):
            raise zdx.Unresolved("Profile sampling requires resolved density samples.")
        if density.ndim < 2 or density.shape[-2:] != sampling.shape:
            raise ValueError("The profile must match the requested spatial grid.")

        # Resample stored densities when needed, then integrate fine cells into pixels.
        if source is not None and not native:
            density = Interp(density, source, fill=0.0).sample(fine.coordinates)

        masses = grids.downsample(density, self.oversample, mean=False)
        if self.normalise:
            return masses / masses.sum((-2, -1), keepdims=True)
        return masses * fine.cell_volume[..., None, None]

    def _normalise(self, *, axis=(-2, -1), **context):
        """Sample and normalise a spatial profile to unit sum.

        Arrays and Expressions must supply non-negative samples with nonzero sum.
        Log profiles retain narrow Gaussians without exponentiating before normalising.
        Zero-width Gaussian axes select their nearest samples, sharing equally near
        ties. This is a discrete sampling policy, separate from Gaussian evaluation.
        ``axis`` selects the normalisation axes; other axes remain independent batches.
        """
        profile = self.profile

        # Custom Gaussian subclasses retain their own log_profile calculation.
        if type(profile) is Gaussian:
            local = profile.sampling_context(ndim=2, **context)
            mean, std = zdx.resolve((profile.mean, profile.std), **local)
            logarithms = self._discrete_log_gaussian(local["coordinates"], mean, std, 2)
            return softmax(logarithms, axis=axis)

        # Inspect the class: Module attribute raising could otherwise pick up a child
        # Gaussian's method and bypass the containing expression's own calculation.
        if getattr(type(profile), "log_profile", None) is not None:
            return softmax(profile.log_profile(**context), axis=axis)
        values = zdx.resolve(profile, **context)
        if isinstance(values, zdx.Expression):
            raise zdx.Unresolved("Profile normalisation requires resolved samples.")
        values = np.asarray(values)
        return values / values.sum(axis, keepdims=True)

    @staticmethod
    def _discrete_log_gaussian(coordinates, mean, std, ndim):
        """Sample a Gaussian, assigning zero widths to the nearest discrete samples.

        The reduction over the grid is only needed when a width is zero. Under vmap,
        JAX may evaluate both conditional branches. Nearly equal distances share the
        support so floating-point roundoff does not break symmetric ties.
        """
        mean = as_axis(mean, ndim, "mean")
        std = as_axis(std, ndim, "std")
        spatial_shape = (1,) * ndim
        centre = mean.reshape(mean.shape + spatial_shape)
        zero_width = (std == 0).reshape(std.shape + spatial_shape)

        def discrete(_):
            distance = np.abs(coordinates - centre)
            spatial_axes = tuple(range(-ndim, 0))
            minimum = distance.min(spatial_axes, keepdims=True)
            tolerance = 4 * np.finfo(distance.dtype).eps
            nearest = np.isclose(distance, minimum, rtol=tolerance, atol=0.0)
            support = np.where(zero_width, nearest, True).all(-ndim - 1)

            # Evaluate positive-width axes normally; zero-width axes contribute only
            # their discrete support, with no attenuation at the selected samples.
            sampled = np.where(zero_width, centre, coordinates)
            logarithms = log_gaussian(sampled, mean, std, ndim)
            return np.where(support, logarithms, -np.inf)

        func = lambda _: log_gaussian(coordinates, mean, std, ndim)
        return lax.cond(np.any(std == 0), discrete, func, operand=None)


class Jitter(zdx.Expression, Unitful):
    """A continuous displacement profile resolving to an image-blur kernel.

    The profile is sampled about zero displacement, integrated over each image
    pixel, then normalised to unit sum. Physical units use the supplied image
    grid's spacing; its centre never shifts the kernel. Pixel widths refer to
    native image pixels, independently of Imager's optical oversampling and
    before any additional detector binning.

    A scalar or (x, y) profile constructs a Gaussian with those standard
    deviations. A (2, 2) covariance constructs a CorrelatedGaussian. Supply an
    Expression for another profile; it receives local ``coordinates`` and
    ``pixel_scale`` in this Jitter's unit.
    Kernel values are dimensionless masses with final shape (ny, nx).
    """

    profile: zdx.Expression
    unit: str = "pixel"
    kernel_size: tuple[int, int] = (9, 9)
    oversample: tuple[int, int] = (3, 3)

    def __init__(
        self, profile, kernel_size=9, oversample=3, *, unit="pixel", alias=None
    ):
        """Store a displacement profile and fixed kernel sampling.

        Parameters
        ----------
        profile : Expression or ArrayLike
            Relative continuous profile, Gaussian width(s), or covariance.
            Profiles must have finite, non-negative samples and nonzero sum.
        kernel_size : int or pair of ints
            Odd native image-pixel support in physical (x, y) order. Optical
            refinement rounds the fine kernel up to odd counts to keep its centre.
        oversample : int or pair of ints
            Subpixel integration factors, independent of optical PSF sampling.
        unit : str
            ``"pixel"`` or a length/angular unit matching the image grid's
            dimension. No angular-to-length plate scale is inferred.
        alias : mapping or sequence of pairs, optional
            User-defined parameter aliases.
        """
        # Turn width or covariance inputs into a continuous Gaussian definition.
        if not isinstance(profile, zdx.Expression):
            value = zdx.as_array(profile, dtype=float)
            if value.ndim == 0 or value.shape == (2,):
                profile = Gaussian(value)
            elif value.shape == (2, 2):
                profile = CorrelatedGaussian.from_cov(value)
            else:
                raise ValueError("profile must be an Expression, width or covariance.")

        # Store a supported spatial unit and an odd, centred kernel sampling.
        self.profile = profile
        self.unit = "pixel" if unit == "pixel" else self.validate_unit(unit)
        dimension = self.unit
        if self.unit not in (None, "pixel"):
            dimension = unit_dimension(self.unit)
        if dimension not in ("pixel", "length", "angle"):
            raise ValueError("Jitter requires pixel, length or angular units.")
        self.kernel_size = grids.as_size(kernel_size, 2, "kernel_size")
        self.oversample = grids.as_size(oversample, 2, "oversample")
        if any(size % 2 == 0 for size in self.kernel_size):
            raise ValueError("kernel_size must contain odd values.")
        self.alias = alias

    def evaluate(self, *, grid=None, oversample=1, **context) -> Array:
        """Return a unit-sum kernel on the supplied image-pixel sampling.

        Physical profiles require a Grid with matching units. Pixel profiles
        can resolve without one. ``oversample`` is the optical refinement from
        native image pixels to this grid, supplied by Imager. The stored
        ``self.oversample`` instead controls subpixel integration of the profile.
        Leading profile and grid batches broadcast.
        """
        # Displacements use local kernel coordinates, never absolute image centres.
        refinement = grids.as_size(oversample, 2, "oversample")
        spacing = 1 / np.asarray(refinement)
        if self.unit != "pixel":
            if grid is None:
                raise zdx.Unresolved("Physical jitter requires an image grid.")
            grid = grid.resolve(**context)
            dimension = unit_dimension(self.unit)
            if grid.unit is None or unit_dimension(grid.unit) != dimension:
                raise ValueError(
                    "Jitter and image grid units must have the same dimension."
                )
            spacing = grid.d_si / unit_factor(self.unit)

        # Odd fine kernels retain a zero-displacement sample for even refinement.
        sizes = tuple(n * q for n, q in zip(self.kernel_size, refinement))
        sizes = tuple(n + (n % 2 == 0) for n in sizes)
        local_grid = grids.Grid(n=sizes, d=spacing)
        profile = Profile(self.profile, oversample=self.oversample)
        return profile.sample(local_grid, **context)
