"""Physical propagation recipes built from Fourier and phase operations.

Each propagator resolves its physical inputs, constructs the required array
operations, and evaluates them. Grid describes explicit sampling; FFTSpec
and ResizeSpec request sampling derived from the incident grid.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import jax.numpy as np
import jax.tree as jtu
from jax.tree_util import keystr
from jax import Array, core
import zodiax as zdx

from ..grids import FFTSpec, Grid, ResizeSpec
from ..numerical.units import unit_dimension
from .abcd import ABCD, _validate_value
from .fft import FFT
from .asm import ASM
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


class BasePropagator(zdx.Module):
    """Numerical propagation with explicit incident sampling and wavelength.

    Calling a propagator returns an array. Pass ``return_grid=True`` to also return the
    realised output Grid. The flag is static under JIT. Parameters and sampling
    may contain expressions; all required values must resolve before evaluation.

    Numerical parameters, sampled kernels and fields use ordinary leading batch
    broadcasting. Final array axes are spatial; no batch axis has a physical role
    inferred from its shape. Use explicit singleton axes or map independent cases.

    ``prepare`` returns a reusable numerical operation: an MFT for direct
    propagation, or a Series of FFT, resize and phase operations. Preparation
    leaves the physical definition unchanged. Prepare again after changing the
    wavelength, sampling or physical parameters; prepare inside the differentiated
    function when derivatives through those quantities are required.

    Subclasses implement ``_prepare(grid_in, wavelength)`` on resolved inputs,
    returning the numerical operation and its realised output grid.
    """

    def __init__(self, *, alias=None):
        """Store optional Zodiax parameter-path aliases."""
        self.alias = alias

    def __call__(self, field, grid_in, wavelength, *, return_grid=False, **context):
        """Propagate a complex array on its declared incident sampling.

        Parameters
        ----------
        field : Array
            Complex samples with final spatial axes in reversed physical order.
        grid_in : Grid
            Complete two-dimensional incident grid, with declared physical units.
        wavelength : Array or Expression
            Positive wavelength in metres; leading dimensions broadcast with grids
            and physical parameters.
        return_grid : bool
            Return ``(field_out, grid_out)`` when true; otherwise return the array.
            Defaults to false and must be static under JIT.
        **context
            Explicit inputs needed by parameter expressions, including state.

        Returns
        -------
        Array or tuple[Array, Grid]
            Propagated samples, optionally with their realised sampling.
        """
        if not isinstance(return_grid, bool):
            raise TypeError("return_grid must be a static bool.")
        operation, output_grid = self.prepare(
            grid_in, wavelength, return_grid=True, **context
        )
        output = operation(field)
        return (output, output_grid) if return_grid else output

    def prepare(self, grid_in, wavelength, *, return_grid=False, **context):
        """Return the sampled operation for repeated array evaluation.

        Parameters
        ----------
        grid_in : Grid
            Two-dimensional incident sampling. Parameter batches use numerical
            broadcasting.
        wavelength : Array or Expression
            Positive wavelength in metres, broadcast with the physical parameters.
        return_grid : bool
            Also return the realised output Grid. Defaults to false and must
            be static under JIT.
        **context
            Inputs required to resolve physical parameters and sampling.

        Returns
        -------
        BaseLayer or tuple[BaseLayer, Grid]
            Callable sampled operation, or ``(operation, output_grid)`` when
            requested. Call the operation with a field array to reuse its kernels.
            The operation contains no reference back to the physical definition.
        """
        if not isinstance(return_grid, bool):
            raise TypeError("return_grid must be a static bool.")

        # Physical definitions consume the incident sampling and wavelength.
        if not isinstance(grid_in, Grid):
            raise TypeError("grid_in must be a Grid.")
        wavelength = self._resolve_input(wavelength, **context)
        wavelength = _validate_value(wavelength, "wavelength", positive=True)
        context = dict(
            context, grid=grid_in, wavelength=wavelength, wavelengths=wavelength
        )
        grid_in = self._check_grid(self._resolve_input(grid_in, **context), "grid_in")
        context["grid"] = grid_in
        definition = self._resolve_parameters(**context)

        operation, output_grid = definition._prepare(grid_in, wavelength)
        return (operation, output_grid) if return_grid else operation

    def _resolve_parameters(self, **context):
        """Resolve the physical parameters needed to prepare this operation."""
        return self._resolve_input(self, **context)

    @abstractmethod
    def _prepare(self, grid_in, wavelength):
        """Return an array operation and output grid from resolved physical inputs."""

    @staticmethod
    def _check_grid(grid, name, *, dimension=None) -> Grid:
        """Check complete sampling and an optional physical unit dimension."""
        if not isinstance(grid, Grid):
            raise TypeError(f"{name} must be a Grid.")
        if grid.n is None or grid.d is None or grid.unit is None:
            raise ValueError(f"{name} must define n, d, and unit.")
        if dimension is not None and unit_dimension(grid.unit) != dimension:
            raise ValueError(f"{name} must use a {dimension!r} unit.")
        return grid

    @staticmethod
    def _check_fft_spec(spec, name) -> FFTSpec:
        """Return an FFT request with a guarded optional centre."""
        if not isinstance(spec, FFTSpec):
            raise TypeError(f"{name} must be an FFTSpec.")
        center = None if spec.c is None else _validate_value(spec.c, f"{name}.c")
        return spec.set(c=center)

    @staticmethod
    def _resolve_input(value, **context):
        """Resolve numerical inputs and identify any missing expression context."""
        value = zdx.resolve(value, **context)
        leaves = jtu.flatten_with_path(
            value, is_leaf=lambda leaf: isinstance(leaf, zdx.Expression)
        )[0]
        for path, leaf in leaves:
            if isinstance(leaf, zdx.Expression):
                name = keystr(path) or "input"
                raise ValueError(
                    f"Unresolved propagation input {name}: {type(leaf).__name__}. "
                    "Supply its required context before propagation."
                )
        return value


class LCT(BasePropagator):
    """General direct Collins propagation onto an explicit Cartesian grid.

    ``abcd`` is a real unimodular matrix or an expression producing one. The direct
    factorisation requires nonzero B. Input and output grids use length units.
    Both quadratic phases are incorporated into separable MFT kernels. The axial
    carrier is omitted and normalisation follows sampled field power.
    """

    abcd: Array | zdx.Expression
    grid_out: Grid

    def __init__(self, abcd, grid_out, *, alias=None):
        """Construct a direct linear canonical transform.

        Parameters
        ----------
        abcd : ABCD, Array, or Expression
            Matrix ending in (2, 2), with broadcast parameter axes. Expressions may
            describe an ordered ABCDChain. Concrete matrices must be unimodular.
        grid_out : Grid
            Complete Cartesian output sampling in declared length units.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.abcd = abcd
        if not isinstance(abcd, zdx.Expression):
            self.abcd = ABCD.validate(abcd, unimodular=True)
        if not isinstance(grid_out, Grid):
            raise TypeError("grid_out must be a Grid.")
        self.grid_out = grid_out
        super().__init__(alias=alias)

    @property
    def matrix(self):
        """Directed ABCD array; expressions must resolve before it is inspected."""
        return ABCD.validate(self._resolve_input(self.abcd), unimodular=True)

    def _prepare(self, grid_in, wavelength):
        """Validate the physical domain and evaluate the selected direct route."""
        grid_in = self._check_grid(grid_in, "grid_in", dimension="length")
        abcd = self.matrix
        _validate_value(abcd[..., 0, 1], "ABCD.b", nonzero=True)
        return self._direct(grid_in, wavelength, abcd)

    def _direct(self, grid_in, wavelength, abcd):
        """Build explicit-grid Collins kernels, fusing the two quadratic phases."""
        grid_out = self._check_grid(self.grid_out, "grid_out", dimension="length")
        a, b, d = abcd[..., 0, 0], abcd[..., 0, 1], abcd[..., 1, 1]
        q = np.asarray(wavelength) * b
        transform = self._fourier_mft(grid_in, grid_out, q)

        # Quadratic phases become row/column factors, not extra field operations.
        before = Chirp.from_axes(grid_in.axes, np.pi * a / q)
        after = Chirp.from_axes(grid_out.axes, np.pi * d / q)
        transform = transform.fuse_input(before).fuse_output(after)
        return transform, grid_out

    @staticmethod
    def _fourier_mft(grid_in, grid_out, q) -> MFT:
        """Prepare only the cross-term Fourier kernels and Collins normalisation."""
        scale = np.sqrt(grid_in.d_si * grid_out.d_si / np.abs(q)[..., None])
        collins = np.exp(-0.25j * np.pi * np.sign(q))
        weight = scale * collins[..., None]
        weights = tuple(weight[..., i] for i in range(grid_in.ndim))
        return MFT.from_grids(grid_in, grid_out, -2 * np.pi / q, weights)


class Propagator(LCT):
    """Explicit-grid propagation through a conjugate relay or custom ABCD system.

    With no ABCD definition, an optional positive focal length selects Cartesian
    focal sampling; otherwise focal coordinates are angular. ``defocus`` adds a
    longitudinal displacement beyond focus. None omits this term; zero retains its
    derivative. Without focal length, defocus uses the equivalent f=1 metre relay.

    A custom ``abcd`` requires Cartesian sampling and excludes focal_length and
    defocus. A supplied matrix already defines its direction, so ``inverse`` has
    no effect in that case. Supply ``np.linalg.inv(abcd.resolve())`` to invert its
    directed map; ``abcd.inverse()`` instead describes physical optical reversal.
    Without a custom matrix, ``inverse`` reverses the relay and returns a
    Cartesian pupil grid.
    Finite windows and resampling need not form an exact numerical inverse.

    Examples
    --------
    ```python
    import jax.numpy as np
    import dLux.utils as dlu

    pupil = dlu.Grid(n=(32, 32), diam=1.0, unit="m")
    focus = dlu.Grid(n=(24, 24), d=20.0, unit="mas")
    propagator = dlu.Propagator(focus)
    field = np.ones(pupil.shape, dtype=complex)
    output = propagator(field, pupil, 600e-9)
    print(propagator)
    print(output.shape)
    ```
    """

    # Accepted output specification, shared by the class rather than stored per object.
    _sampling: ClassVar[type] = Grid
    abcd: Array | zdx.Expression | None = None
    grid_out: Grid | None = None
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
        """Construct explicit output sampling and a physical propagation definition.

        Parameters
        ----------
        grid_out : Grid or None
            Output grid, which may be partial until preparation. Angular for a
            forward relay without focal length,
            Cartesian when focal length or custom ABCD is supplied, and Cartesian
            for reverse propagation. Angular units are converted to radians.
            None leaves the output sampling unspecified until an immutable update;
            preparing or applying propagation requires a grid.
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
        # Select a built-in relay or an explicit matrix, leaving expressions deferred.
        if grid_out is not None and not isinstance(grid_out, self._sampling):
            raise TypeError(f"grid_out must be a {self._sampling.__name__}.")
        if abcd is not None and (focal_length is not None or defocus is not None):
            raise ValueError("abcd is exclusive with focal_length and defocus.")
        focal_length = zdx.as_array(focal_length, dtype=float)
        defocus = zdx.as_array(defocus, dtype=float)
        if focal_length is not None and not isinstance(focal_length, zdx.Expression):
            focal_length = _validate_value(focal_length, "focal_length", positive=True)
        if defocus is not None and not isinstance(defocus, zdx.Expression):
            defocus = _validate_value(defocus, "defocus")

        # An explicit ABCD matrix already determines the propagation direction.
        inverse = None if abcd is not None else bool(inverse)

        self.abcd = abcd
        if abcd is not None and not isinstance(abcd, zdx.Expression):
            self.abcd = ABCD.validate(abcd, unimodular=True)
        self.grid_out = grid_out
        self.focal_length = focal_length
        self.defocus = defocus
        self.inverse = inverse
        self.alias = alias

    def _resolve_parameters(self, **context):
        """Resolve relay leaves without evaluating an Optic's local effects.

        Subclasses can carry transmission, OPD or other definitions evaluated in
        their own coordinate frame. Preparing propagation needs only this relay.
        Populate first so links to parameters elsewhere on the owner still work.
        """
        self = self.populate()
        parameters = dict(
            grid_out=self.grid_out,
            focal_length=self.focal_length,
            defocus=self.defocus,
            abcd=self.abcd,
        )
        return self.set(**self._resolve_input(parameters, **context))

    @property
    def matrix(self):
        """Directed custom or conjugate ABCD array, derived from physical leaves."""
        if self.abcd is not None:
            return ABCD.validate(self._resolve_input(self.abcd), unimodular=True)
        f = 1.0 if self.focal_length is None else self.focal_length
        abcd = ABCD(fraunhofer=f)
        if self.defocus is not None:
            abcd = abcd.then(ABCD(free_space=self.defocus))
        matrix = abcd.resolve()
        return np.linalg.inv(matrix) if self.inverse else matrix

    def _prepare(self, grid_in, wavelength):
        """Use general Collins kernels or the specialised conjugate calculation."""
        if self.grid_out is None:
            raise ValueError("Propagation requires grid_out.")
        if self.abcd is not None:
            if self.focal_length is not None or self.defocus is not None:
                raise ValueError("abcd is exclusive with focal_length and defocus.")
            return super()._prepare(grid_in, wavelength)

        # The conjugate route constructs only the required Fourier kernels.
        operation, grid_out = self._prepare_fraunhofer(grid_in, wavelength)
        if self.defocus is None:
            return operation, grid_out

        # Defocus contributes one pupil-side phase, including at zero displacement.
        f = 1.0 if self.focal_length is None else self.focal_length
        z = _validate_value(self.defocus, "defocus")
        direction = 1 if self.inverse else -1
        pupil = grid_out if self.inverse else grid_in
        quadratic = direction * np.pi * z / (wavelength * f**2)
        phase = Chirp.from_axes(pupil.axes, quadratic)

        # Fuse defocus into MFT kernels; otherwise retain it as a separate series step.
        if isinstance(operation, MFT):
            if self.inverse:
                operation = operation.fuse_output(phase)
            else:
                operation = operation.fuse_input(phase)
        elif self.inverse:
            operation = _series(operation, phase)
        else:
            operation = _series(phase, operation)
        return operation, grid_out

    def _prepare_fraunhofer(self, grid_in, wavelength):
        """Prepare relay coordinates, then use the subclass's sampling route."""
        angular = self.focal_length is None
        focal_length = 1.0
        if not angular:
            focal_length = _validate_value(
                self.focal_length, "focal_length", positive=True
            )

        # Both MFT and FFT relays use physical coordinates for their kernels.
        dimension = "angle" if angular and self.inverse else "length"
        grid_in = self._check_grid(grid_in, "grid_in", dimension=dimension)
        physical_in = self._focal_length_grid(grid_in, focal_length)
        return self._relay(physical_in, wavelength, focal_length, angular)

    def _relay(self, grid_in, wavelength, focal_length, angular):
        """Build an explicit-grid relay, preserving the declared output sampling."""
        dimension = "angle" if angular and not self.inverse else "length"
        grid_out = self._check_grid(self.grid_out, "grid_out", dimension=dimension)
        physical_out = self._focal_length_grid(grid_out, focal_length)
        q = wavelength * focal_length * (-1 if self.inverse else 1)
        return self._fourier_mft(grid_in, physical_out, q), grid_out

    @staticmethod
    def _focal_length_grid(grid, focal_length) -> Grid:
        """Convert an already-checked angular focal grid to physical metres."""
        if unit_dimension(grid.unit) == "length":
            return grid
        scale = np.asarray(focal_length)[..., None]
        return Grid(grid.n, scale * grid.d_si, scale * grid.c_si, "m")


class FFTPropagator(Propagator):
    """Conjugate or custom ABCD propagation on native FFT output sampling.

    Physical parameters follow Propagator. FFTSpec determines working dimensions,
    cropping, and optional output centre; output spacing is derived from wavelength,
    input sampling, and the physical transform. A custom ABCD completely determines
    the propagation, including its Fourier sign; ``inverse`` has no effect. This
    also works for traced matrices and batches with different signs of B.
    """

    _sampling: ClassVar[type] = FFTSpec
    abcd: Array | zdx.Expression | None = None
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
            focal output. ResizeSpec supplies counts only and is promoted to an
            FFTSpec with its native centre.
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
            grid_out = FFTSpec(
                n=grid_out.n,
                pad=grid_out.pad,
                crop=grid_out.crop,
                alias=grid_out.alias,
            )
        super().__init__(
            grid_out, focal_length, defocus, abcd=abcd, inverse=inverse, alias=alias
        )

    def _direct(self, grid_in, wavelength, abcd):
        """Prepare direct Collins FFT evaluation, fusing phase factors on each side."""
        spec = self._check_fft_spec(self.grid_out, "grid_out")
        a, b, d = abcd[..., 0, 0], abcd[..., 0, 1], abcd[..., 1, 1]
        q = wavelength * b
        working, grid_freq, grid_out = self._fourier_grids(grid_in, spec, q)

        # The directed matrix determines the Fourier sign. A concrete, uniform sign
        # selects FFT/IFFT at preparation; traced or mixed signs stay as array values.
        inverse = q < 0
        if not isinstance(inverse, core.Tracer):
            all_inverse = np.all(inverse)
            any_inverse = np.any(inverse)
            if not isinstance(all_inverse, core.Tracer) and all_inverse:
                inverse = True
            elif not isinstance(any_inverse, core.Tracer) and not any_inverse:
                inverse = False

        # Combine the input curvature and optional Fourier-window shift.
        direction = -np.sign(q)
        linear = None
        if spec.c is not None:
            shift = self._frequency_shift(grid_freq, grid_out, q)
            linear = tuple(direction * 2 * np.pi * value for value in shift)
        before = Chirp.from_axes(grid_in.axes, np.pi * a / q, linear)

        # Combine FFT-origin correction with curvature on the returned output grid.
        ramp = self._output_ramp(working, grid_freq, grid_out.n, inverse)
        curvature = Chirp.from_axes(grid_out.axes, np.pi * d / q)
        after = ramp.then(curvature)
        transform = FFT.prepare(
            grid_in.n,
            working.n,
            input=before.factors,
            output=after.factors,
            inverse=inverse,
        )
        return transform, grid_out

    def _relay(self, grid_in, wavelength, focal_length, angular):
        """Build native Fourier sampling and return the requested output units."""
        spec = self._check_fft_spec(self.grid_out, "grid_out")
        angular_output = angular and not self.inverse
        if angular_output and spec.c is not None:
            center = np.asarray(focal_length)[..., None] * spec.c
            spec = spec.set(c=center)

        q = np.asarray(wavelength) * focal_length * (-1 if self.inverse else 1)
        working, grid_freq, grid_out = self._fourier_grids(grid_in, spec, q)
        after = self._output_ramp(working, grid_freq, grid_out.n, self.inverse)

        # The native route omits input phases; a shifted window adds one linear ramp.
        before = None
        if spec.c is not None:
            direction = 1 if self.inverse else -1
            shift = self._frequency_shift(grid_freq, grid_out, q)
            linear = tuple(direction * 2 * np.pi * value for value in shift)
            before = Chirp.from_axes(grid_in.axes, linear=linear).factors
        transform = FFT.prepare(
            grid_in.n,
            working.n,
            input=before,
            output=after.factors,
            inverse=self.inverse,
        )

        # FFT kernels use metres; the returned sampling follows the relay's units.
        if angular_output:
            scale = np.asarray(focal_length)[..., None]
            grid_out = Grid(
                grid_out.n, grid_out.d_si / scale, grid_out.c_si / scale, "rad"
            )
        return transform, grid_out

    @staticmethod
    def _fourier_grids(grid_in, spec, q):
        """Resolve padded input, native spectrum and requested physical output."""
        working = spec.working_grid(grid_in)
        grid_freq = working.frequency()
        scale = np.abs(q)[..., None]
        native = Grid(working.n, scale * grid_freq.d_si, scale * grid_freq.c_si, "m")
        return working, grid_freq, spec.output_grid(native, grid_in.n)

    @staticmethod
    def _output_ramp(working, grid_freq, n_out, inverse) -> Chirp:
        """Sample the physical FFT-origin correction and Collins phase."""
        direction = np.where(inverse, 1.0, -1.0)
        origin = direction[..., None] * 2 * np.pi * FFT.origin(working)
        linear = tuple(origin[..., i] for i in range(working.ndim))
        axes = grid_freq.resize_samples(n_out).axes
        return Chirp.from_axes(axes, linear=linear, constant=direction * np.pi / 4)

    @staticmethod
    def _frequency_shift(grid_freq, grid_out, q):
        """Return the shift from native to requested output frequencies per axis."""
        native = grid_freq.resize_samples(grid_out.n)
        delta = grid_out.c_si / np.abs(q)[..., None] - native.c_si
        return tuple(delta[..., i] for i in range(grid_out.ndim))


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
        grid_in = self._check_grid(grid_in, "grid_in", dimension="length")
        power = _validate_value(self.power, "power")
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
        if isinstance(resize, FFTSpec) and resize.c is not None:
            raise ValueError("ASM resize cannot define an output centre.")
        distance = zdx.as_array(distance, dtype=float)
        if not isinstance(distance, zdx.Expression):
            distance = _validate_value(distance, "distance")
        self.distance = distance
        self.resize = resize
        super().__init__(alias=alias)

    @property
    def abcd(self) -> ABCD:
        """Return the equivalent paraxial free-space matrix."""
        distance = _validate_value(zdx.resolve(self.distance), "distance")
        return ABCD(free_space=distance)

    def _prepare(self, grid_in, wavelength):
        """Prepare native FFT-convolution ASM for one source context."""
        grid_in = self._check_grid(grid_in, "grid_in", dimension="length")
        distance = _validate_value(self.distance, "distance")

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
        transform = ASM(transfer.factors, n_in=grid_in.n, n_out=n_out)
        return transform, working.resize_samples(n_out)


class SpectralPropagator(LCT):
    """The spectral two-transform factorisation of an ABCD LCT.

    An explicit frequency ``Grid`` builds MFT-to-MFT evaluation. A native
    frequency ``FFTSpec`` builds FFT-to-MFT evaluation and may pad but not crop the
    intermediate spectrum. Both produce an explicit physical output ``Grid``;
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
    grid_freq : Grid or FFTSpec
        Explicit or native intermediate spatial-frequency sampling.
    grid_out : Grid
        Explicit physical output sampling.
    """

    abcd: Any
    grid_freq: Grid | FFTSpec
    grid_out: Grid

    def __init__(self, abcd, grid_freq, grid_out, *, alias=None):
        """Initialise a reusable spectral LCT.

        Parameters
        ----------
        abcd : ABCD, ArrayLike, or Expression
            Unimodular matrix object or array ending in shape ``(2, 2)``, or an
            expression producing one. Parameter batches broadcast; A must be nonzero.
        grid_freq : Grid or FFTSpec
            Explicit or native intermediate frequency sampling.
        grid_out : Grid
            Explicit physical output sampling.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        if not isinstance(grid_freq, (Grid, FFTSpec)):
            raise TypeError("grid_freq must be a Grid or FFTSpec.")
        if not isinstance(grid_out, Grid):
            raise TypeError("grid_out must be a Grid.")
        self.grid_freq = grid_freq
        super().__init__(abcd, grid_out, alias=alias)

    def _prepare(self, grid_in, wavelength):
        """Prepare the spectral operation for one source context."""
        grid_in = self._check_grid(grid_in, "grid_in", dimension="length")
        grid_out = self._check_grid(self.grid_out, "grid_out", dimension="length")
        abcd = self.matrix
        _validate_value(abcd[..., 0, 0], "ABCD.a", nonzero=True)

        # Explicit frequency sampling evaluates two requested MFTs
        if isinstance(self.grid_freq, Grid):
            grid_freq = self._check_grid(
                self.grid_freq, "grid_freq", dimension="spatial_frequency"
            )
            inverse = self._inverse(grid_in, grid_freq, wavelength, abcd)
            weight = tuple(grid_in.d_si[..., i] for i in range(grid_in.ndim))
            forward = MFT.from_grids(grid_in, grid_freq, -2 * np.pi, weight)
            return _series(forward, inverse), grid_out

        # Native frequency sampling accelerates only the first transform
        spec = self._check_fft_spec(self.grid_freq, "grid_freq")
        working = spec.working_grid(grid_in)
        native = working.frequency()
        grid_freq = spec.output_grid(native, grid_in.n)
        if grid_freq.n != working.n:
            raise ValueError(
                "A spectral FFTSpec may pad but may not crop the spectrum; "
                "use an explicit frequency Grid to band-limit it."
            )

        # Fuse physical FFT-origin correction into the inverse Fourier kernel.
        inverse = self._inverse(grid_in, grid_freq, wavelength, abcd, working.n)
        origin = -2 * np.pi * FFT.origin(working)
        linear = tuple(origin[..., i] for i in range(grid_in.ndim))
        ramp = Chirp.from_axes(native.axes, linear=linear)
        inverse = inverse.fuse_input(ramp)

        # A shifted spectrum needs only an input ramp, applied before padding.
        forward = FFT.prepare(grid_in.n, working.n)
        if spec.c is not None:
            shift = -2 * np.pi * (grid_freq.c_si - native.c_si)
            linear = tuple(shift[..., i] for i in range(grid_in.ndim))
            ramp = Chirp.from_axes(grid_in.axes, linear=linear)
            forward = _series(ramp, forward)
        return _series(forward, inverse), grid_out

    def _inverse(self, grid_in, grid_freq, wavelength, abcd, fft_counts=None) -> MFT:
        """Build inverse quadrature from the already checked physical matrix."""
        grid_out = self.grid_out
        a, b, c = abcd[..., 0, 0], abcd[..., 0, 1], abcd[..., 1, 0]
        q = wavelength * b

        # Continue the positive-B branch at zero. In two transverse dimensions
        # both one-sided limits give the signed 1/A magnification prefactor.
        direction = np.where(q < 0, -1.0, 1.0)
        branch = direction - np.sign(direction / a)
        phase = np.exp(-0.25j * np.pi * branch)
        scale = phase / np.sqrt(np.abs(a))

        # Convert continuous inverse quadrature to the sampled-field convention.
        weight = grid_freq.d_si * np.sqrt(grid_out.d_si / grid_in.d_si)
        weight = weight * scale[..., None]
        if fft_counts is not None:
            weight = weight * grid_in.d_si * np.sqrt(np.asarray(fft_counts))
        weights = tuple(weight[..., i] for i in range(grid_in.ndim))
        inverse = MFT.from_grids(grid_freq, grid_out, 2 * np.pi / a, weights)

        # Fuse transfer and output curvature into the two matrix dimensions.
        transfer = Chirp.from_axes(grid_freq.axes, -np.pi * q / a)
        curvature = Chirp.from_axes(grid_out.axes, np.pi * c / (wavelength * a))
        inverse = inverse.fuse_input(transfer)
        return inverse.fuse_output(curvature)
