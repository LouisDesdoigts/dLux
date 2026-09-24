"""Parametric Jones matrices with shape ``(..., 2, 2)``.

The final two axes are output and input polarisation components. Leading axes
broadcast, typically over spatial coordinates. Angles are in radians, measured
counter-clockwise from the horizontal x-axis towards the vertical y-axis.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict

import jax.numpy as np
import zodiax as zdx
from jax import Array

from ..sequential import Series, list2dictionary

__all__ = ["BaseJones", "Jones", "JonesMatrix", "JonesChain"]


# Convert flattened coherency matrices to Stokes [I, Q, U, V]. The V sign retains
# the convention used by dLux's circular polarisers and Jones wavefront fields.
_STOKES = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1j, 1j, 0]])
_COHERENCY = np.conj(_STOKES.T) / 2


class BaseJones(zdx.Expression):
    """A Jones response resolving to an array of shape ``(..., 2, 2)``.

    Subclasses implement ``evaluate(**context)``. The final two axes describe
    output and input polarisation; leading sample axes broadcast. Calling without
    an input resolves the response; calling with a matrix left-multiplies it.
    Optic owns applying the response to a Wavefront.
    """

    @abstractmethod
    def evaluate(self, **context) -> Array:
        """Resolve operands and return the definition's (..., 2, 2) matrix.

        Raise zdx.Unresolved when a required context value is unavailable, allowing
        a surrounding numerical expression to remain partially resolved.
        """

    @classmethod
    def from_matrix(cls, matrix) -> JonesMatrix:
        """Wrap a (..., 2, 2) array or expression in a JonesMatrix."""
        return JonesMatrix(matrix)

    def apply(self, matrix, **context) -> Array:
        """Left-multiply a Jones response of shape (..., 2, 2).

        Leading axes broadcast normally. This operates on arrays; a core Optic
        owns promotion and axis handling for Wavefront objects.
        """
        self, matrix = zdx.populate((self, matrix))
        response = self.evaluate(**context)
        matrix = zdx.resolve(matrix, **context)
        return response @ matrix

    def __call__(self, matrix=None, **context) -> Array | BaseJones:
        """Resolve this response, or left-multiply an explicit (..., 2, 2) input."""
        if matrix is None:
            return self.resolve(**context)
        return self.apply(matrix, **context)

    def rotate(self, angle, **context) -> JonesMatrix:
        """Resolve and rotate a uniform response by a scalar angle in radians.

        Return a JonesMatrix containing the rotated array, including for zero
        rotation. None is not an angle. Spatial matrix fields require an explicit
        coordinate transform and resampling, so this operation is uniform-only.
        """
        if angle is None:
            raise ValueError("angle cannot be None; use 0 for no rotation.")
        self, angle = zdx.populate((self, angle))
        matrix = self.evaluate(**context)
        angle = zdx.resolve(angle, **context)
        if matrix.shape != (2, 2) or np.ndim(angle) != 0:
            raise ValueError(
                "rotate requires a uniform (2, 2) matrix and scalar angle."
            )
        return self.from_matrix(Jones.rotate_matrix(matrix, angle))

    def then(self, other, **context) -> JonesMatrix:
        """Resolve both definitions and return their combined JonesMatrix.

        ``other`` may be an array or Jones expression. The result applies
        other after self, multiplying matrices at corresponding spatial samples
        with ordinary broadcasting. Supply context needed by either definition.
        The returned object stores matrix entries, rather than the original
        parameters; compose inside a differentiated calculation to retain their
        derivatives.
        """
        # Populate together so parameters may be shared across the two operands.
        first, second = zdx.populate((self, other))
        first = first.evaluate(**context)
        second = zdx.resolve(second, **context)
        matrix = second @ first
        return self.from_matrix(matrix)

    def mueller(self, **context) -> Array:
        """Return the real Mueller matrix with shape (..., 4, 4)."""
        matrix = self.populate().evaluate(**context)
        product = np.einsum("...ik,...jl->...ijkl", matrix, np.conj(matrix))
        product = product.reshape(matrix.shape[:-2] + (4, 4))
        return (_STOKES @ product @ _COHERENCY).real

    def to_stokes(self, stokes=None, **context) -> Array:
        """Return output Stokes [I, Q, U, V] with shape (..., 4).

        Input stokes has shape (..., 4) with broadcast leading axes. None denotes
        unpolarised unit intensity, [1, 0, 0, 0].
        """
        self, stokes = zdx.populate((self, stokes))
        if stokes is None:
            stokes = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            stokes = zdx.resolve(stokes, **context)
        return np.einsum("...ab,...b->...a", self.mueller(**context), stokes)


class Jones(BaseJones):
    """An ideal linear polariser or retarder with one physical axis angle.

    ``retardance=None`` selects a polariser; a numerical retardance selects a
    retarder. Zero retardance gives identity, not a polariser. Scalar parameters
    describe uniform elements; arrays broadcast to shape ``(..., 2, 2)`` and may
    describe spatial variation. Expressions resolve on the supplied local context.

    Lowercase helpers such as ``linear_polariser`` return arrays. Capitalised
    factories such as ``LinearPolariser`` return Jones definitions, or JonesMatrix
    objects when ``matrix=True`` requests immediate evaluation. Circular polarisers
    cannot be represented by one linear element; their factories return JonesMatrix.

    Examples
    --------
    ```python
    import dLux as dl
    import dLux.utils as dlu

    plate = dlu.Jones.HalfWavePlate(angle=0.2)
    print(plate)                     # Jones(angle=f32[], retardance=f32[])
    optic = dl.Optic(jones=plate)
    matrix = dlu.Jones.half_wave_plate(angle=0.2)
    ```
    """

    angle: Array | zdx.Expression
    retardance: Array | zdx.Expression | None = None

    def __init__(self, angle=0.0, retardance=None, *, alias=None):
        """Store the axis angle and optional retardance without making a matrix.

        Parameters
        ----------
        angle : float, Array, or Expression, optional
            Transmission-axis angle for a polariser, or fast-axis angle for a
            retarder, in radians counter-clockwise from horizontal. Default zero.
            Scalars and spatial arrays follow ordinary broadcasting. Explicit
            None is invalid; use zero for the horizontal axis.
        retardance : float, Array, Expression, or None, optional
            None selects a linear polariser. Otherwise this is the phase delay of
            the second local axis relative to the first, in radians. Zero gives
            identity. Values broadcast with angle; neither parameter moves pixels.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        """
        if angle is None:
            raise ValueError("angle cannot be None; use 0 for the horizontal axis.")
        self.angle = zdx.as_array(angle, dtype=float)
        self.retardance = zdx.as_array(retardance, dtype=float)
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Resolve the physical inputs and express the element in global axes."""
        # Resolve both operands together so they may share linked values.
        angle, retardance = zdx.resolve((self.angle, self.retardance), **context)

        # A polariser blocks the second local axis; a retarder delays it.
        if retardance is None:
            second_axis = 0.0
        else:
            second_axis = np.exp(1j * retardance)
        cosine, sine = np.cos(angle), np.sin(angle)

        # Rotate the two local polarisation axes into the global frame.
        xx = cosine**2 + second_axis * sine**2
        yy = sine**2 + second_axis * cosine**2
        xy = (1 - second_axis) * cosine * sine
        return np.stack((np.stack((xx, xy), -1), np.stack((xy, yy), -1)), -2)

    def rotate(self, angle, **context) -> Jones:
        """Rotate a uniform element's axis, keeping its retardance and class.

        Resolve scalar parameters in the supplied context, then add angle in
        radians to the existing axis angle. A half-wave plate remains half-wave.
        Spatial definitions require coordinate rotation and resampling instead.
        None is invalid; use zero for no rotation.
        """
        if angle is None:
            raise ValueError("angle cannot be None; use 0 for no rotation.")
        axis, retardance, angle = zdx.resolve(
            (self.angle, self.retardance, angle), **context
        )

        # All parameters must be ready before we can establish spatial uniformity.
        for value in (axis, retardance, angle):
            if isinstance(value, zdx.Expression):
                raise zdx.Unresolved("Supply the context needed to rotate this Jones.")
            if value is not None and np.ndim(value) != 0:
                raise ValueError(
                    "rotate requires a uniform Jones definition and scalar angle."
                )
        return self.set(angle=axis + angle, retardance=retardance)

    @classmethod
    def LinearPolariser(
        cls, angle=0.0, *, matrix=False, **context
    ) -> Jones | JonesMatrix:
        """Construct a linear polariser at angle, in radians from horizontal.

        Angle may be scalar, spatial, or an expression. By default it remains a
        parameter. With matrix=True, supply any context needed to evaluate it now;
        the returned JonesMatrix stores the resulting array.
        """
        polariser = cls(angle=angle)
        if matrix:
            return cls.from_matrix(polariser.evaluate(**context))
        return polariser

    @classmethod
    def HorizontalPolariser(cls, *, matrix=False) -> Jones | JonesMatrix:
        """Construct an angle-zero polariser; matrix=True materialises its array."""
        return cls.LinearPolariser(angle=0.0, matrix=matrix)

    @classmethod
    def VerticalPolariser(cls, *, matrix=False) -> Jones | JonesMatrix:
        """Construct an angle-pi/2 polariser; matrix=True materialises its array."""
        return cls.LinearPolariser(angle=np.pi / 2, matrix=matrix)

    @classmethod
    def Retarder(
        cls, retardance, angle=0.0, *, matrix=False, **context
    ) -> Jones | JonesMatrix:
        """Construct a retarder with phase delay and fast-axis angle in radians.

        Retardance is required, including zero for identity. Both parameters may
        be arrays or expressions with broadcast shapes. With matrix=True, supply
        context to evaluate them now and store the result in a JonesMatrix.
        """
        if retardance is None:
            raise ValueError(
                "Retarder requires a numerical retardance, including zero."
            )
        plate = cls(angle=angle, retardance=retardance)
        if matrix:
            return cls.from_matrix(plate.evaluate(**context))
        return plate

    @classmethod
    def HalfWavePlate(
        cls, angle=0.0, *, matrix=False, **context
    ) -> Jones | JonesMatrix:
        """Construct a pi-retardance plate; scalar or spatial angle is in radians.

        matrix=True evaluates now using context and returns a JonesMatrix.
        """
        return cls.Retarder(np.pi, angle, matrix=matrix, **context)

    @classmethod
    def QuarterWavePlate(
        cls, angle=0.0, *, matrix=False, **context
    ) -> Jones | JonesMatrix:
        """Construct a pi/2-retardance plate; scalar or spatial angle is in radians.

        matrix=True evaluates now using context and returns a JonesMatrix.
        """
        return cls.Retarder(np.pi / 2, angle, matrix=matrix, **context)

    @classmethod
    def Identity(cls, *, matrix=False) -> Jones | JonesMatrix:
        """Construct a zero-retardance element, optionally as a JonesMatrix."""
        return cls.Retarder(0.0, matrix=matrix)

    @classmethod
    def RHCPolariser(cls) -> JonesMatrix:
        """Construct a right-circular polariser as a general JonesMatrix."""
        return cls.from_matrix(cls.rhc_polariser())

    @classmethod
    def LHCPolariser(cls) -> JonesMatrix:
        """Construct a left-circular polariser as a general JonesMatrix."""
        return cls.from_matrix(cls.lhc_polariser())

    @classmethod
    def linear_polariser(cls, angle=0.0, **context) -> Array:
        """Return a polariser array; angle in radians may vary spatially.

        Expressions must be ready in the supplied context. The array has shape
        (..., 2, 2) with the angle's leading dimensions.
        """
        return cls.LinearPolariser(angle).evaluate(**context)

    @classmethod
    def horizontal_polariser(cls) -> Array:
        """Return the uniform horizontal-polariser array, shape (2, 2)."""
        return cls.HorizontalPolariser().evaluate()

    @classmethod
    def vertical_polariser(cls) -> Array:
        """Return the uniform vertical-polariser array, shape (2, 2)."""
        return cls.VerticalPolariser().evaluate()

    @classmethod
    def retarder(cls, retardance, angle=0.0, **context) -> Array:
        """Return a retarder array; phase delay and angle in radians broadcast.

        Expressions must be ready in context. The result has shape (..., 2, 2).
        """
        return cls.Retarder(retardance, angle).evaluate(**context)

    @classmethod
    def half_wave_plate(cls, angle=0.0, **context) -> Array:
        """Return a half-wave array; scalar or spatial angle is in radians."""
        return cls.HalfWavePlate(angle).evaluate(**context)

    @classmethod
    def quarter_wave_plate(cls, angle=0.0, **context) -> Array:
        """Return a quarter-wave array; scalar or spatial angle is in radians."""
        return cls.QuarterWavePlate(angle).evaluate(**context)

    @classmethod
    def identity(cls) -> Array:
        """Return the uniform identity array, shape (2, 2)."""
        return cls.Identity().evaluate()

    @staticmethod
    def rhc_polariser() -> Array:
        """Return the right-circular projector in dLux's Stokes V convention."""
        return np.array([[0.5, -0.5j], [0.5j, 0.5]])

    @staticmethod
    def lhc_polariser() -> Array:
        """Return the left-circular projector in dLux's Stokes V convention."""
        return np.array([[0.5, 0.5j], [-0.5j, 0.5]])

    @staticmethod
    def rotate_matrix(matrix, angle) -> Array:
        """Rotate polarisation axes at fixed samples and return an array.

        Matrix has shape (..., 2, 2); angle in radians broadcasts over leading
        dimensions. None is invalid; use zero for no rotation. This operation
        neither rotates nor interpolates a spatial pattern. Use instance rotate
        for a whole uniform element; spatial pattern rotation needs resampling.
        """
        if angle is None:
            raise ValueError("angle cannot be None; use 0 for no rotation.")
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.stack(
            (np.stack((cosine, sine), -1), np.stack((-sine, cosine), -1)), -2
        )
        return np.swapaxes(rotation, -1, -2) @ matrix @ rotation


class JonesMatrix(BaseJones):
    """A general uniform or spatial Jones matrix with shape (..., 2, 2).

    Store a uniform or spatial matrix directly, or an expression generating one.
    There is no separate orientation parameter. Named physical definitions live
    on Jones; this class also represents the materialised result of composition.

    Examples
    --------
    ```python
    import dLux.utils as dlu

    polariser = dlu.Jones.LinearPolariser(angle=0.2, matrix=True)
    print(polariser)                 # JonesMatrix(matrix=f32[2,2])
    stokes = polariser.to_stokes()
    ```
    """

    matrix: Array | zdx.Expression

    def __init__(self, matrix, *, alias=None):
        """Store a matrix or expression without adding physical parameters.

        Parameters
        ----------
        matrix : Array or Expression
            Real or complex matrix with shape (..., 2, 2), or an expression
            or Jones object producing one. The final axes are output and input
            polarisation.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        """
        self.matrix = zdx.as_array(matrix)
        if not isinstance(self.matrix, zdx.Expression):
            if self.matrix.shape[-2:] != (2, 2):
                raise ValueError("Jones matrices must have shape (..., 2, 2).")
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Resolve the matrix expression using the supplied context."""
        return zdx.resolve(self.matrix, **context)


class JonesChain(Series, BaseJones):
    """Named Jones responses composed in application order.

    The chain stores named ``layers`` and resolves to their combined matrix.
    Calling it with a matrix applies the chain to that input. An empty chain is
    identity. Stored responses may be expressions
    or already-resolved arrays with shape (..., 2, 2), broadcasting leading axes.

    Examples
    --------
    ```python
    import dLux as dl

    chain = dl.JonesChain({"polariser": dl.Jones.LinearPolariser(),
                           "plate": dl.Jones.HalfWavePlate(angle=0.2)})
    print(chain.resolve())
    ```
    """

    # Inherited fields
    layers: OrderedDict

    def __init__(self, elements=(), *, alias=None):
        """Store a mapping or sequence of optionally named Jones responses.

        Parameters
        ----------
        elements : mapping or sequence, optional
            Arrays or Jones expressions, optionally named with (name, value) pairs.
            The first listed response acts first. The empty sequence is identity.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        """
        entries = list(elements.items()) if isinstance(elements, dict) else elements
        elements = list2dictionary(entries, ordered=True)
        layers = {
            name: value if isinstance(value, BaseJones) else JonesMatrix(value)
            for name, value in elements.items()
        }
        super().__init__(layers, alias=alias)

    def evaluate(self, **context) -> Array:
        """Compose the responses from identity."""
        return self.apply(**context)

    def apply(self, matrix=None, **context) -> Array:
        """Compose the responses, optionally acting on an initial input matrix."""
        self, matrix = self._prepare(matrix, **context)
        return Series.apply(self, matrix, **context)

    def debug(self, matrix=None, **context):
        """Return the composed response and its named intermediate matrices."""
        self, matrix = self._prepare(matrix, **context)
        return Series.debug(self, matrix, **context)

    def _prepare(self, matrix, **context):
        """Keep partially resolved responses callable for ordinary sequence execution.

        A previous partial resolve may have replaced ready Jones definitions with
        arrays. Wrapping those as JonesMatrix restores their multiplication call;
        the shared Series loop can then handle every step in the same way.
        """
        self, matrix = zdx.populate((self, matrix))
        matrix = np.eye(2) if matrix is None else zdx.resolve(matrix, **context)
        layers = self.layers.copy()
        for name, element in self.steps.items():
            layers[name] = (
                element if isinstance(element, BaseJones) else JonesMatrix(element)
            )
        return self.set(layers=layers), matrix
