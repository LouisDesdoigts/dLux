"""Physical propagation recipes built from Fourier and phase operations.

Each propagator resolves its physical inputs, constructs the required array
operations, and evaluates them. GridSpec describes explicit sampling; FFTSpec
and ResizeSpec request sampling derived from the incident grid.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import jax
import jax.numpy as np
from jax import core
import numpy as onp
import zodiax as zdx

from ..grids import FFTSpec, GridSpec, ResizeSpec
from ..units import _unit_dimension
from .abcd import ABCD
from .fft import FFT, _fft_convolution, _output_fft, _resized_fft, _shifted_fft
from .mft import MFT
from .operations import Chirp, _series

__all__ = [
    "BasePropagator",
    "LCT",
    "Propagator",
    "FFTPropagator",
    "FreeSpace",
    "SpectralPropagator",
    "ThinPower",
]


def _validate_value(value, name, *, positive=False, nonzero=False):
    """Check concrete values at setup, adding no compiled runtime checks.

    Traced inputs retain the documented physical domain as a caller contract.
    """
    value = np.asarray(value)
    if isinstance(value, core.Tracer):
        return value
    concrete = onp.asarray(value)
    invalid = ~onp.isfinite(concrete)
    if positive:
        invalid |= concrete <= 0
    if nonzero:
        invalid |= concrete == 0
    if onp.any(invalid):
        domain = "positive" if positive else "non-zero" if nonzero else "finite"
        raise ValueError(f"{name} must contain finite {domain} values.")
    return value


def _validate_finite(value, name):
    """Check finite concrete setup inputs."""
    return _validate_value(value, name)


def _validate_positive(value, name):
    """Check positive concrete setup inputs."""
    return _validate_value(value, name, positive=True)


def _validate_nonzero(value, name):
    """Check non-zero concrete setup inputs."""
    return _validate_value(value, name, nonzero=True)


def _validate_fft_direction(value, inverse, name):
    """Check that concrete Fourier scales agree with the static FFT direction."""
    value = _validate_nonzero(value, name)
    if not isinstance(value, core.Tracer):
        wrong = onp.asarray(value) >= 0 if inverse else onp.asarray(value) <= 0
        if onp.any(wrong):
            raise ValueError(f"{name} sign must agree with inverse={inverse}.")
    return value


def _validated_grid(grid, name) -> GridSpec:
    """Return a complete regular grid for physical propagation."""
    if not isinstance(grid, GridSpec):
        raise TypeError(f"{name} must be a GridSpec.")
    if grid.n is None or grid.d is None or grid.unit is None:
        raise ValueError(f"{name} must define n, d, and unit.")
    if grid.indexing != "xy":
        raise ValueError(f"{name} must use indexing='xy' for optical propagation.")
    return grid


def _require_grid(grid, dimension, name) -> GridSpec:
    """Return a complete grid with the required physical dimension."""
    grid = _validated_grid(grid, name)
    if _unit_dimension(grid.unit) != dimension:
        raise ValueError(f"{name} must use a {dimension!r} unit.")
    return grid


def _require_fft_spec(spec, name) -> FFTSpec:
    """Return an FFT request with a guarded optional centre."""
    if not isinstance(spec, FFTSpec):
        raise TypeError(f"{name} must be an FFTSpec.")
    center = None if spec.c is None else _validate_finite(spec.c, f"{name}.c")
    return spec.set(c=center)


def _require_propagation_ndim(grid) -> None:
    """Restrict physical propagation to one or two transverse dimensions."""
    if not isinstance(grid, GridSpec):
        raise TypeError("grid_in must be a GridSpec.")
    if grid.ndim not in (1, 2):
        raise ValueError("physical propagation supports one or two transverse axes.")


def _require_unimodular(abcd) -> ABCD:
    """Return an ABCD matrix guarded as a symplectic optical transform."""
    if not isinstance(abcd, ABCD):
        raise TypeError("abcd must be an ABCD.")
    if abcd.matrix.shape[-2:] != (2, 2):
        raise ValueError("ABCD.matrix must end in shape (2, 2).")
    if np.issubdtype(abcd.matrix.dtype, np.complexfloating):
        raise ValueError("ABCD.matrix must be real-valued.")

    if isinstance(abcd.matrix, core.Tracer):
        return abcd
    matrix = onp.asarray(abcd.matrix)
    tolerance = 100 * onp.finfo(matrix.dtype).eps
    determinant = onp.linalg.det(matrix)
    if not onp.all(onp.isfinite(matrix)) or not onp.all(
        abs(determinant - 1) <= tolerance
    ):
        raise ValueError(
            "ABCD must be finite and unimodular for physical LCT propagation."
        )
    return abcd


def _resolved(value, **context):
    """Resolve numerical inputs and identify any missing expression context."""
    value = zdx.resolve(value, **context)
    leaves = jax.tree_util.tree_flatten_with_path(
        value, is_leaf=lambda leaf: isinstance(leaf, zdx.Expression)
    )[0]
    for path, leaf in leaves:
        if isinstance(leaf, zdx.Expression):
            name = jax.tree_util.keystr(path) or "input"
            raise ValueError(
                f"Unresolved propagation input {name}: {type(leaf).__name__}. "
                "Supply its required context before propagation."
            )
    return value


def _fourier_mft(grid_in, grid_out, q) -> MFT:
    """Prepare only the cross-term Fourier kernels and Collins normalisation."""
    q = _validate_nonzero(q, "Fourier scale")
    scale = np.sqrt(grid_in.d_si * grid_out.d_si / np.abs(q)[..., None])
    collins = np.exp(-0.25j * np.pi * np.sign(q))
    weight = scale * collins[..., None]
    weights = tuple(weight[..., i] for i in range(grid_in.ndim))
    return MFT.from_grids(grid_in, grid_out, -2 * np.pi / q, weights)


def _fraunhofer_mft(grid_in, grid_out, wavelength, focal_length, inverse) -> MFT:
    """Prepare a dedicated Fraunhofer transform, without constructing chirps."""
    q = np.asarray(wavelength) * focal_length * (-1 if inverse else 1)
    return _fourier_mft(grid_in, grid_out, q)


def _direct_mft(grid_in, grid_out, wavelength, abcd) -> MFT:
    """Prepare the Collins cross-term and fuse the two quadratic phases."""
    q = np.asarray(wavelength) * abcd.b
    transform = _fourier_mft(grid_in, grid_out, q)
    batch = np.broadcast_shapes(transform.batch_shape, abcd.matrix.shape[:-2])

    # Quadratic phases become row/column factors, not extra field operations.
    before = Chirp.from_axes(grid_in.axes, np.pi * abcd.a / q).broadcast(batch)
    after = Chirp.from_axes(grid_out.axes, np.pi * abcd.d / q).broadcast(batch)
    transform = transform.broadcast(batch).fuse_input(before)
    return transform.fuse_output(after)


def _fourier_grids(grid_in, spec, q):
    """Resolve padded input, native spectrum and requested physical output."""
    working = spec.working_grid(grid_in)
    grid_freq = working.frequency()
    scale = np.abs(q)[..., None]
    native = GridSpec(working.n, scale * grid_freq.d_si, scale * grid_freq.c_si, "m")
    return working, grid_freq, spec.output_grid(native, grid_in.n)


def _output_ramp(working, grid_freq, n_out, inverse) -> Chirp:
    """Sample the physical FFT-origin correction and Collins phase."""
    direction = np.where(inverse, 1.0, -1.0)
    origin = direction[..., None] * 2 * np.pi * FFT.origin(working)
    linear = tuple(origin[..., i] for i in range(working.ndim))
    axes = grid_freq.resize_samples(n_out).axes
    return Chirp.from_axes(axes, linear=linear, constant=direction * np.pi / 4)


def _frequency_shift(grid_freq, grid_out, q):
    """Return the shift from native to requested output frequencies per axis."""
    native = grid_freq.resize_samples(grid_out.n)
    delta = grid_out.c_si / np.abs(q)[..., None] - native.c_si
    return tuple(delta[..., i] for i in range(grid_out.ndim))


def _fraunhofer_fft(grid_in, spec, wavelength, focal_length, inverse):
    """Prepare a chirp-free FFT with physical centring and optional output shift."""
    q = np.asarray(wavelength) * focal_length * (-1 if inverse else 1)
    q = _validate_fft_direction(q, inverse, "wavelength * focal_length")
    working, grid_freq, grid_out = _fourier_grids(grid_in, spec, q)
    batch = np.broadcast_shapes(grid_in.batch_shape, grid_out.batch_shape, np.shape(q))

    # The native path has no input chirp or shift arrays.
    after = _output_ramp(working, grid_freq, grid_out.n, inverse).broadcast(batch)
    if spec.c is None:
        transform = _output_fft(grid_in.n, working.n, after.factors, inverse=inverse)
        return transform, grid_out

    # Shift the requested Fourier window before padding and evaluating the FFT.
    direction = 1 if inverse else -1
    shift = _frequency_shift(grid_freq, grid_out, q)
    linear = tuple(direction * 2 * np.pi * value for value in shift)
    before = Chirp.from_axes(grid_in.axes, linear=linear).broadcast(batch)
    transform = _shifted_fft(working.n, before.factors, after.factors, inverse=inverse)
    return transform, grid_out


def _direct_fft(grid_in, spec, wavelength, abcd):
    """Prepare direct Collins FFT evaluation, fusing phase factors on each side."""
    q = _validate_nonzero(wavelength * abcd.b, "wavelength * ABCD.b")
    working, grid_freq, grid_out = _fourier_grids(grid_in, spec, q)
    batch = np.broadcast_shapes(
        grid_in.batch_shape, grid_out.batch_shape, abcd.matrix.shape[:-2]
    )

    # The directed matrix determines the Fourier sign. A concrete, uniform sign
    # selects FFT/IFFT at preparation; traced or mixed signs stay as array values.
    inverse = q < 0
    if not isinstance(inverse, core.Tracer):
        signs = onp.asarray(inverse)
        if onp.all(signs):
            inverse = True
        elif not onp.any(signs):
            inverse = False
    if not isinstance(inverse, bool):
        inverse = np.broadcast_to(inverse, batch)

    # Combine the input curvature and optional Fourier-window shift.
    direction = -np.sign(q)
    linear = None
    if spec.c is not None:
        shift = _frequency_shift(grid_freq, grid_out, q)
        linear = tuple(direction * 2 * np.pi * value for value in shift)
    before = Chirp.from_axes(grid_in.axes, np.pi * abcd.a / q, linear).broadcast(batch)

    # Combine FFT-origin correction with curvature on the returned output grid.
    ramp = _output_ramp(working, grid_freq, grid_out.n, inverse)
    curvature = Chirp.from_axes(grid_out.axes, np.pi * abcd.d / q)
    after = ramp.then(curvature).broadcast(batch)
    transform = _shifted_fft(working.n, before.factors, after.factors, inverse=inverse)
    return transform, grid_out


def _focal_length_grid(grid, focal_length, name) -> GridSpec:
    """Represent a length or angular focal grid in physical metres."""
    grid = _validated_grid(grid, name)
    if _unit_dimension(grid.unit) == "length":
        return grid
    if _unit_dimension(grid.unit) != "angle":
        raise ValueError(f"{name} must use a length or angular unit.")

    scale = np.asarray(focal_length)[..., None]
    return GridSpec(grid.n, scale * grid.d_si, scale * grid.c_si, "m")


def _focal_fft_spec(spec, focal_length, angular) -> tuple[FFTSpec, str | None]:
    """Return a physical direct-FFT request and optional angular output unit."""
    spec = _require_fft_spec(spec, "grid_out")
    if not angular:
        return spec, None

    c = None
    if spec.c is not None:
        c = np.asarray(focal_length)[..., None] * spec.c
    return spec.set(c=c), "rad"


def _from_focal_length_grid(grid, focal_length) -> GridSpec:
    """Convert a physical focal grid to normalised angular coordinates."""
    scale = np.asarray(focal_length)[..., None]
    return GridSpec(grid.n, grid.d_si / scale, grid.c_si / scale, "rad")


def _prepare_fraunhofer(grid_in, grid_out, wavelength, focal_length, inverse):
    """Prepare a Fourier relay in angular or Cartesian focal coordinates."""
    wavelength = _validate_positive(wavelength, "wavelength")
    angular = focal_length is None
    focal_length = 1.0 if angular else _validate_positive(focal_length, "focal_length")

    # Express the input plane in the physical coordinates used by the kernels.
    dimension_in = "angle" if angular and inverse else "length"
    grid_in = _require_grid(grid_in, dimension_in, "grid_in")
    physical_in = _focal_length_grid(grid_in, focal_length, "grid_in")
    _require_propagation_ndim(physical_in)

    # Explicit output sampling builds only Fourier kernels, without chirps.
    if isinstance(grid_out, GridSpec):
        dimension_out = "angle" if angular and not inverse else "length"
        grid_out = _require_grid(grid_out, dimension_out, "grid_out")
        physical_out = _focal_length_grid(grid_out, focal_length, "grid_out")
        if physical_in.ndim != physical_out.ndim:
            raise ValueError("grid_in and grid_out dimensions must match.")
        transform = _fraunhofer_mft(
            physical_in,
            physical_out,
            wavelength,
            focal_length,
            inverse,
        )
        return transform, grid_out

    # FFT sampling includes padding, cropping and a requested focal-plane centre.
    fft_spec, angular_unit = _focal_fft_spec(
        grid_out, focal_length, angular and not inverse
    )
    transform, grid_out = _fraunhofer_fft(
        physical_in,
        fft_spec,
        wavelength,
        focal_length,
        inverse,
    )
    if angular_unit is not None:
        grid_out = _from_focal_length_grid(grid_out, focal_length)
    return transform, grid_out


def _as_abcd(abcd):
    """Keep ABCD expressions and validate concrete matrices."""
    if isinstance(abcd, zdx.Expression):
        return abcd
    if not isinstance(abcd, ABCD):
        abcd = ABCD(abcd)
    return _require_unimodular(abcd)


def _relay_grid(grid_out, focal_length, abcd, inverse):
    """Validate output units before any numerical sampling takes place."""
    if isinstance(grid_out, GridSpec):
        dimension = (
            "length"
            if abcd is not None or inverse or focal_length is not None
            else "angle"
        )
        _require_grid(grid_out, dimension, "grid_out")


class BasePropagator(zdx.Module):
    """Numerical propagation with explicit incident sampling and wavelength.

    Calling a propagator returns an array. Pass ``grid_out=True`` to also return the
    realised output GridSpec. The flag is static under JIT. Parameters and sampling
    may contain expressions; all required values must resolve before evaluation.

    Numerical parameter batches use ordinary broadcasting. Sampled kernels align
    with the leading field prefix, leaving later axes such as Jones components
    untouched. For an outer wavelength/grid population use wavelength ``(W, 1)``,
    grid leaves ``(B, ndim)``, and fields ``(W, B, ..., ny, nx)``.

    ``prepare`` returns a reusable numerical operation: an MFT for direct
    propagation, or a Sequential of FFT, resize and phase operations. Preparation
    leaves the physical definition unchanged. Prepare again after changing the
    wavelength, sampling or physical parameters; prepare inside the differentiated
    function when derivatives through those quantities are required.

    Subclasses implement ``_prepare(grid_in, wavelength)`` on resolved inputs,
    returning the numerical operation and its realised output grid.
    """

    def __init__(self, *, alias=None):
        """Store optional Zodiax parameter-path aliases."""
        self.alias = alias

    def __call__(self, field, grid_in, wavelength, *, grid_out=False, **context):
        """Propagate a complex array on its declared incident sampling.

        Parameters
        ----------
        field : Array
            Complex samples with final spatial axes in reversed physical order.
        grid_in : GridSpec
            Complete regular incident grid, with declared physical units.
        wavelength : Array or Expression
            Positive wavelength in metres; leading dimensions broadcast with grids
            and physical parameters.
        grid_out : bool
            Return ``(field_out, grid_out)`` when true; otherwise return the array.
            Defaults to false and must be static under JIT.
        **context
            Explicit inputs needed by parameter expressions, including state.

        Returns
        -------
        Array or tuple[Array, GridSpec]
            Propagated samples, optionally with their realised sampling.
        """
        if not isinstance(grid_out, bool):
            raise TypeError("grid_out must be a static bool.")
        operation, output_grid = self.prepare(
            grid_in, wavelength, grid_out=True, **context
        )
        output = operation(field)
        return (output, output_grid) if grid_out else output

    def prepare(self, grid_in, wavelength, *, grid_out=False, **context):
        """Return the sampled operation for repeated array evaluation.

        Parameters
        ----------
        grid_in : GridSpec
            Incident sampling. Parameter batches use numerical broadcasting.
        wavelength : Array or Expression
            Positive wavelength in metres, broadcast with the physical parameters.
        grid_out : bool
            Also return the realised output GridSpec. Defaults to false and must
            be static under JIT.
        **context
            Inputs required to resolve physical parameters and sampling.

        Returns
        -------
        BaseOperation or Sequential or tuple
            Callable sampled operation, or ``(operation, output_grid)`` when
            requested. Call the operation with a field array to reuse its kernels.
            The operation contains no reference back to the physical definition.
        """
        if not isinstance(grid_out, bool):
            raise TypeError("grid_out must be a static bool.")
        definition, grid_in, wavelength = self._resolve(grid_in, wavelength, **context)
        operation, output_grid = definition._prepare(grid_in, wavelength)
        return (operation, output_grid) if grid_out else operation

    def _resolve(self, grid_in, wavelength, **context):
        """Resolve the physical definition in the incident coordinate context."""
        if not isinstance(grid_in, GridSpec):
            raise TypeError("grid_in must be a GridSpec.")
        wavelength = _validate_positive(_resolved(wavelength, **context), "wavelength")
        context = {**context, "grid": grid_in, "wavelengths": wavelength}
        grid_in = _resolved(grid_in, **context).sampled(**context)
        context["grid"] = grid_in
        definition = _resolved(self, **context)
        return definition, grid_in, wavelength

    @abstractmethod
    def _prepare(self, grid_in, wavelength):
        """Return an array operation and output grid from resolved physical inputs."""


class LCT(BasePropagator):
    """General direct Collins propagation onto an explicit Cartesian grid.

    ``abcd`` is a real unimodular matrix or an expression producing one. The direct
    factorisation requires nonzero B. Input and output grids use length units.
    Both quadratic phases are incorporated into separable MFT kernels. The axial
    carrier is omitted and normalisation follows sampled field power.
    """

    abcd: ABCD | zdx.Expression
    grid_out: GridSpec

    def __init__(self, abcd, grid_out, *, alias=None):
        """Construct a direct linear canonical transform.

        Parameters
        ----------
        abcd : ABCD, Array, or Expression
            Matrix ending in (2, 2), with broadcast parameter axes. Expressions may
            describe an ordered ABCDChain. Concrete matrices must be unimodular.
        grid_out : GridSpec
            Complete Cartesian output sampling in declared length units.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.abcd = _as_abcd(abcd)
        self.grid_out = _require_grid(grid_out, "length", "grid_out")
        super().__init__(alias=alias)

    @property
    def matrix(self):
        """Directed ABCD array; expressions must resolve before it is inspected."""
        return _as_abcd(self.abcd).matrix

    def _prepare(self, grid_in, wavelength):
        """Validate the physical domain and evaluate the selected direct route."""
        grid_in = _require_grid(grid_in, "length", "grid_in")
        _require_propagation_ndim(grid_in)
        abcd = _require_unimodular(ABCD(self.matrix))
        _validate_nonzero(abcd.b, "ABCD.b")
        return self._direct(grid_in, wavelength, abcd)

    def _direct(self, grid_in, wavelength, abcd):
        """Build explicit-grid Collins kernels."""
        grid_out = _require_grid(self.grid_out, "length", "grid_out")
        if grid_in.ndim != grid_out.ndim:
            raise ValueError("grid_in and grid_out dimensions must match.")
        return _direct_mft(grid_in, grid_out, wavelength, abcd), grid_out


class Propagator(LCT):
    """Explicit-grid propagation through a conjugate relay or custom ABCD system.

    With no ABCD definition, an optional positive focal length selects Cartesian
    focal sampling; otherwise focal coordinates are angular. ``defocus`` adds a
    longitudinal displacement beyond focus. None omits this term; zero retains its
    derivative. Without focal length, defocus uses the equivalent f=1 metre relay.

    A custom ``abcd`` requires Cartesian sampling and excludes focal_length and
    defocus. A supplied matrix already defines its direction, so ``inverse`` has
    no effect in that case; pass ``abcd.inverse()`` to reverse it. Without a custom
    matrix, ``inverse`` reverses the relay and returns a Cartesian pupil grid.
    Finite windows and resampling need not form an exact numerical inverse.

    Examples
    --------
    ```python
    import jax.numpy as np
    import dLux.utils as dlu

    pupil = dlu.GridSpec(n=(32, 32), diam=1.0, unit="m")
    focus = dlu.GridSpec(n=(24, 24), d=20.0, unit="mas")
    propagator = dlu.Propagator(focus)
    field = np.ones(pupil.shape, dtype=complex)
    output = propagator(field, pupil, 600e-9)
    print(propagator)
    print(output.shape)
    ```
    """

    # Accepted output specification, shared by the class rather than stored per object.
    _sampling: ClassVar[type] = GridSpec
    abcd: ABCD | zdx.Expression | None = None
    grid_out: GridSpec
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False

    def __init__(
        self,
        grid_out,
        focal_length=None,
        defocus=None,
        *,
        abcd=None,
        inverse=False,
        alias=None,
    ):
        """Construct explicit output sampling and a physical propagation definition.

        Parameters
        ----------
        grid_out : GridSpec
            Complete output grid: angular for a forward relay without focal length,
            Cartesian when focal length or custom ABCD is supplied, and Cartesian
            for reverse propagation. Angular units are converted to radians.
        focal_length : Array, Expression, or None
            Positive focal length in metres. None selects angular focal coordinates.
        defocus : Array, Expression, or None
            Signed longitudinal focal displacement in metres. None omits defocus;
            zero keeps the term differentiable at exact focus.
        abcd : ABCD, Array, Expression, or None
            Optional custom matrix or chain, exclusive with focal_length and defocus.
        inverse : bool or None
            Reverse a built-in relay; defaults to false. None also selects forward
            propagation. With a supplied matrix this argument is ignored and stored
            as None: the matrix itself defines the direction.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases. Physical parameter arrays broadcast.
        """
        if not isinstance(grid_out, self._sampling):
            raise TypeError(f"grid_out must be a {self._sampling.__name__}.")
        if abcd is not None and (focal_length is not None or defocus is not None):
            raise ValueError("abcd is exclusive with focal_length and defocus.")
        focal_length = zdx.as_array(focal_length, dtype=float)
        defocus = zdx.as_array(defocus, dtype=float)
        if focal_length is not None and not isinstance(focal_length, zdx.Expression):
            focal_length = _validate_positive(focal_length, "focal_length")
        if defocus is not None and not isinstance(defocus, zdx.Expression):
            defocus = _validate_finite(defocus, "defocus")
        inverse = None if abcd is not None else bool(inverse)
        _relay_grid(grid_out, focal_length, abcd, inverse)

        self.abcd = None if abcd is None else _as_abcd(abcd)
        self.grid_out = grid_out
        self.focal_length = focal_length
        self.defocus = defocus
        self.inverse = inverse
        self.alias = alias

    @property
    def matrix(self):
        """Directed custom or conjugate ABCD array, derived from physical leaves."""
        if self.abcd is not None:
            return _as_abcd(self.abcd).matrix
        f = 1.0 if self.focal_length is None else self.focal_length
        abcd = ABCD.fraunhofer(f)
        if self.defocus is not None:
            abcd = abcd.then(ABCD.free_space(self.defocus))
        return (abcd.inverse() if self.inverse else abcd).matrix

    def _prepare(self, grid_in, wavelength):
        """Use general Collins kernels or the specialised conjugate calculation."""
        if self.abcd is not None:
            if self.focal_length is not None or self.defocus is not None:
                raise ValueError("abcd is exclusive with focal_length and defocus.")
            return super()._prepare(grid_in, wavelength)

        # The conjugate route constructs only the required Fourier kernels.
        operation, grid_out = _prepare_fraunhofer(
            grid_in, self.grid_out, wavelength, self.focal_length, self.inverse
        )
        if self.defocus is None:
            return operation, grid_out

        # Defocus contributes one pupil-side phase, including at zero displacement.
        f = 1.0 if self.focal_length is None else self.focal_length
        z = _validate_finite(self.defocus, "defocus")
        direction = 1 if self.inverse else -1
        pupil = grid_out if self.inverse else grid_in
        quadratic = direction * np.pi * z / (wavelength * f**2)
        batch = np.broadcast_shapes(
            grid_in.batch_shape, grid_out.batch_shape, np.shape(quadratic)
        )
        phase = Chirp.from_axes(pupil.axes, quadratic).broadcast(batch)
        if isinstance(operation, MFT):
            operation = operation.broadcast(batch)
            operation = (
                operation.fuse_output(phase)
                if self.inverse
                else operation.fuse_input(phase)
            )
        elif self.inverse:
            operation = _series(operation, phase)
        else:
            operation = _series(phase, operation)
        return operation, grid_out


class FFTPropagator(Propagator):
    """Conjugate or custom ABCD propagation on native FFT output sampling.

    Physical parameters follow Propagator. FFTSpec determines working dimensions,
    cropping, and optional output centre; output spacing is derived from wavelength,
    input sampling, and the physical transform. A custom ABCD completely determines
    the propagation, including its Fourier sign; ``inverse`` has no effect. This
    also works for traced matrices and batches with different signs of B.
    """

    _sampling: ClassVar[type] = FFTSpec
    abcd: ABCD | zdx.Expression | None = None
    grid_out: FFTSpec
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False

    def __init__(
        self,
        grid_out=None,
        focal_length=None,
        defocus=None,
        *,
        abcd=None,
        inverse=False,
        alias=None,
    ):
        """Construct native Fourier sampling and its physical transformation.

        Parameters
        ----------
        grid_out : FFTSpec, ResizeSpec, or None
            FFT counts or padding/cropping and centre. None uses unchanged counts.
            Centre uses metres for custom/physical output and radians for angular
            focal output. ResizeSpec is promoted to the equivalent FFTSpec.
        focal_length : Array, Expression, or None
            Positive focal length in metres; None gives angular focal coordinates.
        defocus : Array, Expression, or None
            Signed focal displacement in metres; None omits the phase term.
        abcd : ABCD, Array, Expression, or None
            Custom matrix or chain, exclusive with focal_length and defocus.
        inverse : bool or None
            Reverse a built-in relay, returning Cartesian output; defaults to false.
            None also selects forward propagation. Ignored and stored as None when a
            matrix is supplied; its values determine propagation directly.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        if grid_out is None:
            grid_out = FFTSpec()
        elif isinstance(grid_out, ResizeSpec) and not isinstance(grid_out, FFTSpec):
            grid_out = (
                FFTSpec(n=grid_out.n, c=grid_out.c)
                if grid_out.n is not None
                else FFTSpec(pad=grid_out.pad, crop=grid_out.crop, c=grid_out.c)
            )
        super().__init__(
            grid_out, focal_length, defocus, abcd=abcd, inverse=inverse, alias=alias
        )

    def _direct(self, grid_in, wavelength, abcd):
        """Build native Collins FFT phases and sampling."""
        spec = _require_fft_spec(self.grid_out, "grid_out")
        return _direct_fft(grid_in, spec, wavelength, abcd)


class ThinPower(BasePropagator):
    """A thin lens defined by optical power, with unchanged output sampling.

    Power is ``1 / focal_length``. Calling this propagator samples a Chirp using
    the incident coordinates and wavelength. Chirp stores those sampled factors;
    ThinPower stores the physical lens parameter from which they are calculated.
    """

    power: Any

    def __init__(self, power, *, alias=None):
        """Construct a same-grid thin phase.

        Parameters
        ----------
        power : Array or Expression
            Optical power in reciprocal metres; leading dimensions broadcast with
            wavelength and grid parameters.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.power = zdx.as_array(power, dtype=float)
        super().__init__(alias=alias)

    def _prepare(self, grid_in, wavelength):
        """Sample the phase on the incident Cartesian grid."""
        grid_in = _require_grid(grid_in, "length", "grid_in")
        _require_propagation_ndim(grid_in)
        power = _validate_finite(self.power, "power")
        return Chirp.from_axes(grid_in.axes, -np.pi * power / wavelength), grid_in


class FreeSpace(BasePropagator):
    """Native-grid paraxial angular-spectrum propagation.

    This propagator exclusively prepares FFT-to-transfer-to-IFFT evaluation. Its
    output inherits the source spacing and centre after the requested central resize.
    Use :class:`Propagator` with a free-space ABCD matrix for direct propagation
    onto an arbitrary Cartesian output grid.

    Parameters
    ----------
    distance : ArrayLike
        Signed free-space distance in metres.
    resize : ResizeSpec or None
        FFT working and output topology. ``None`` uses a unit
        :class:`~dLux.utils.grids.ResizeSpec`.
    """

    distance: Any
    resize: ResizeSpec

    def __init__(self, distance, resize=None, *, alias=None):
        """Initialise a reusable paraxial ASM propagator.

        Parameters
        ----------
        distance : ArrayLike
            Signed propagation distance in metres.
        resize : ResizeSpec or None
            FFT working and output topology; defaults to no resize.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        resize = ResizeSpec() if resize is None else resize
        if not isinstance(resize, ResizeSpec):
            raise TypeError("resize must be a ResizeSpec or None.")
        if resize.c is not None:
            raise ValueError("ASM resize cannot define an output centre.")
        distance = zdx.as_array(distance, dtype=float)
        if not isinstance(distance, zdx.Expression):
            distance = _validate_finite(distance, "distance")
        self.distance = distance
        self.resize = resize
        super().__init__(alias=alias)

    @property
    def abcd(self) -> ABCD:
        """Return the equivalent paraxial free-space matrix."""
        distance = _validate_finite(zdx.resolve(self.distance), "distance")
        return ABCD.free_space(distance)

    def _prepare(self, grid_in, wavelength):
        """Prepare native FFT-convolution ASM for one source context."""
        wavelength = _validate_positive(wavelength, "wavelength")
        grid_in = _require_grid(grid_in, "length", "grid_in")
        _require_propagation_ndim(grid_in)
        distance = _validate_finite(self.distance, "distance")

        # Resolve padding without discarding any input samples.
        n_work, n_out = self.resize.counts(grid_in.n)
        if any(work < size for work, size in zip(n_work, grid_in.n)):
            raise ValueError("ASM working counts may not crop the input grid.")
        working = grid_in.resize_samples(n_work)

        # Sample the paraxial transfer in the raw FFT frequency order.
        frequencies = tuple(
            np.fft.fftfreq(size, working.d_si[..., i, None])
            for i, size in enumerate(n_work)
        )
        transfer = Chirp.from_axes(frequencies, -np.pi * wavelength * distance)
        transform = _fft_convolution(grid_in.n, n_out, transfer.factors)
        return transform, working.resize_samples(n_out)


class SpectralPropagator(LCT):
    """The spectral two-transform factorisation of an ABCD LCT.

    An explicit frequency ``GridSpec`` builds MFT-to-MFT evaluation. A native
    frequency ``FFTSpec`` builds FFT-to-MFT evaluation and may pad but not crop the
    intermediate spectrum. Both produce an explicit physical output ``GridSpec``;
    FFT-to-IFFT remains exclusively an ASM route. Use an explicit frequency grid
    when deliberate spectral band-limiting is required.

    This is a numerical factorisation, not a different physical propagation law.
    A free-space matrix describes the same paraxial operator as FreeSpace.
    A defocused Fourier-relay matrix describes Fresnel propagation. Its
    A=-defocus/f term vanishes at exact focus: use the direct focal transform there.
    Different finite intermediate grids need not produce identical discretised fields.

    Parameters
    ----------
    abcd : ABCD, ArrayLike, or Expression
        Unimodular paraxial ray-transfer matrix. The spectral form also requires
        non-zero ``A``.
    grid_freq : GridSpec or FFTSpec
        Explicit or native intermediate spatial-frequency sampling.
    grid_out : GridSpec
        Explicit physical output sampling.
    """

    abcd: Any
    grid_freq: GridSpec | FFTSpec
    grid_out: GridSpec

    def __init__(self, abcd, grid_freq, grid_out, *, alias=None):
        """Initialise a reusable spectral LCT.

        Parameters
        ----------
        abcd : ABCD, ArrayLike, or Expression
            Unimodular matrix object or array ending in shape ``(2, 2)``, or an
            expression producing one. Parameter batches broadcast; A must be nonzero.
        grid_freq : GridSpec or FFTSpec
            Explicit or native intermediate frequency sampling.
        grid_out : GridSpec
            Explicit physical output sampling.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        if not isinstance(grid_freq, (GridSpec, FFTSpec)):
            raise TypeError("grid_freq must be a GridSpec or FFTSpec.")
        if not isinstance(grid_out, GridSpec):
            raise TypeError("grid_out must be a GridSpec.")
        self.grid_freq = grid_freq
        super().__init__(abcd, grid_out, alias=alias)

    def _prepare(self, grid_in, wavelength):
        """Prepare the spectral operation for one source context."""
        wavelength = _validate_positive(wavelength, "wavelength")
        grid_in = _require_grid(grid_in, "length", "grid_in")
        grid_out = _require_grid(self.grid_out, "length", "grid_out")
        _require_propagation_ndim(grid_in)
        abcd = _require_unimodular(ABCD(self.matrix))
        _validate_nonzero(abcd.a, "ABCD.a")
        if grid_in.ndim != grid_out.ndim:
            raise ValueError("grid_in and grid_out dimensions must match.")

        # Explicit frequency sampling evaluates two requested MFTs
        if isinstance(self.grid_freq, GridSpec):
            grid_freq = _require_grid(
                self.grid_freq,
                "spatial_frequency",
                "grid_freq",
            )
            if grid_in.ndim != grid_freq.ndim:
                raise ValueError("grid_in and grid_freq dimensions must match.")
            inverse = self._inverse(grid_in, grid_freq, wavelength)
            weight = tuple(grid_in.d_si[..., i] for i in range(grid_in.ndim))
            forward = MFT.from_grids(grid_in, grid_freq, -2 * np.pi, weight)
            batch = np.broadcast_shapes(inverse.batch_shape, grid_in.batch_shape)
            forward = forward.broadcast(batch)
            return _series(forward, inverse), grid_out

        # Native frequency sampling accelerates only the first transform
        spec = _require_fft_spec(self.grid_freq, "grid_freq")
        working = spec.working_grid(grid_in)
        native = working.frequency()
        grid_freq = spec.output_grid(native, grid_in.n)
        if grid_freq.n != working.n:
            raise ValueError(
                "A spectral FFTSpec may pad but may not crop the spectrum; "
                "use an explicit frequency GridSpec to band-limit it."
            )

        # Fuse physical FFT-origin correction into the inverse Fourier kernel.
        inverse = self._inverse(grid_in, grid_freq, wavelength, working.n)
        origin = -2 * np.pi * FFT.origin(working)
        linear = tuple(origin[..., i] for i in range(grid_in.ndim))
        ramp = Chirp.from_axes(native.axes, linear=linear)
        batch = np.broadcast_shapes(inverse.batch_shape, grid_in.batch_shape)
        inverse = inverse.fuse_input(ramp.broadcast(batch))

        # A shifted spectrum needs only an input ramp, applied before padding.
        forward = _resized_fft(grid_in.n, working.n)
        if spec.c is not None:
            shift = -2 * np.pi * (grid_freq.c_si - native.c_si)
            linear = tuple(shift[..., i] for i in range(grid_in.ndim))
            ramp = Chirp.from_axes(grid_in.axes, linear=linear)
            forward = _series(ramp.broadcast(inverse.batch_shape), forward)
        return _series(forward, inverse), grid_out

    def _inverse(self, grid_in, grid_freq, wavelength, fft_counts=None) -> MFT:
        """Prepare inverse quadrature with spectral and output chirps fused in."""
        abcd, grid_out = ABCD(self.matrix), self.grid_out
        q = wavelength * abcd.b

        # Continue the positive-B branch at zero. In two transverse dimensions
        # both one-sided limits give the signed 1/A magnification prefactor.
        direction = np.where(q < 0, -1.0, 1.0)
        branch = direction - np.sign(direction / abcd.a)
        phase = np.exp(-0.25j * np.pi * branch)
        scale = phase / np.sqrt(np.abs(abcd.a))

        # Convert continuous inverse quadrature to the sampled-field convention.
        weight = grid_freq.d_si * np.sqrt(grid_out.d_si / grid_in.d_si)
        weight = weight * scale[..., None]
        if fft_counts is not None:
            weight = weight * grid_in.d_si * np.sqrt(np.asarray(fft_counts))
        weights = tuple(weight[..., i] for i in range(grid_in.ndim))
        inverse = MFT.from_grids(grid_freq, grid_out, 2 * np.pi / abcd.a, weights)

        # Fuse transfer and output curvature into the two matrix dimensions.
        transfer = Chirp.from_axes(grid_freq.axes, -np.pi * q / abcd.a)
        curvature = Chirp.from_axes(
            grid_out.axes, np.pi * abcd.c / (wavelength * abcd.a)
        )
        batch = np.broadcast_shapes(
            inverse.batch_shape, grid_in.batch_shape, np.shape(wavelength)
        )
        inverse = inverse.broadcast(batch)
        inverse = inverse.fuse_input(transfer.broadcast(batch))
        return inverse.fuse_output(curvature.broadcast(batch))
