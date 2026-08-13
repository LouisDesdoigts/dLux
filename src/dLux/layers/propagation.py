"""Physical and ABCD-based wavefront propagation layers."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import BaseGridSpec, GridSpec, ResizeSpec
from .optical import OpticalLayer

__all__ = [
    "ABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDFraunhofer",
    "Propagator",
    "FocalPropagator",
    "ABCDPropagator",
    "FreeSpace",
    "Fraunhofer",
    "Fresnel",
]


def _propagation_inputs(wf):
    """Broadcast wavelength and coordinate axes over non-spatial field dimensions."""
    # Broadcast wavelength over intrinsic non-spatial field axes
    wavelength = np.asarray(wf.wavelength)
    extra = wf.phasor.ndim - wavelength.ndim - 2
    wavelength = wavelength.reshape(wavelength.shape + (1,) * extra)

    # Promote coordinate axes over Jones dimensions when required
    x, y = wf.axes
    if wf.is_polarised:
        x, y = x[..., None, None, :], y[..., None, None, :]
    return wavelength, x, y


def _propagate_mft(wf, grid, ABCD=None, **kwargs):
    """Propagate every field to an explicit output grid."""
    # Resolve input and requested output coordinate axes
    wavelength, x, y = _propagation_inputs(wf)
    axes_out = grid.axes

    # Define propagation of one monochromatic field
    def propagate(field, lam, x, y):
        if ABCD is None:
            return dlu.MFT(field, lam, (x, y), axes_out, **kwargs)
        return dlu.ABCD_MFT(field, lam, (x, y), axes_out, ABCD)

    # Vectorise propagation over leading field dimensions
    signature = "(n,m),(),(m),(n)->(p,q)"
    propagate = np.vectorize(propagate, signature=signature)
    phasor = propagate(wf.phasor, wavelength, x, y)

    # Update the propagated field and requested grid
    return wf.set(phasor=phasor, grid=grid)


def _propagate_fft(
    wf, grid, unit=None, ABCD=None, focal_length=None, inverse=False, **kwargs
):
    """Propagate every field at native FFT sampling."""
    # Resolve the output units, centre, and padding
    if unit is None:
        unit = "m" if inverse else "rad"
        if focal_length is not None:
            unit = wf.grid.unit

    scale = dlu.unit_factor(unit)
    center = dlu.as_axis(grid.c, 2, "c")
    center = None if center is None else center * scale
    padding = grid._padding

    # Configure the FFT propagation function and inputs
    if ABCD is None:
        fn = dlu.FFT
        inputs = {"focal_length": focal_length, "inverse": inverse, **kwargs}
    else:
        fn = dlu.ABCD_FFT
        inputs = {"ABCD": ABCD}

    # Define propagation over one monochromatic field
    def propagate(field, wavelength, x, y):
        field, axes = fn(
            field,
            wavelength,
            (x, y),
            output_center=center,
            **padding,
            **inputs,
        )
        return field, *axes

    # Vectorise propagation over the leading field axes
    signature = "(n,m),(),(m),(n)->(p,q),(q),(p)"
    propagate = np.vectorize(propagate, signature=signature)
    wavelength, x, y = _propagation_inputs(wf)
    field, x, y = propagate(wf.phasor, wavelength, x, y)

    # Remove the polarisation axes from the output coordinates
    if wf.is_polarised:
        x, y = x[..., 0, 0, :], y[..., 0, 0, :]

    # Crop the propagated field and coordinate axes
    field = grid.crop_array(field)
    x, y = grid.crop_axes((x, y))

    # Construct the realised output grid
    grid = GridSpec.from_axes((x, y), unit)

    # Update the propagated wavefront
    return wf.set(phasor=field, grid=grid)


def _validate_grid(grid, name, ndim=2, angular=None):
    """Validate a complete propagation grid and its coordinate unit."""
    if grid.n is None or grid.d is None or grid.unit is None:
        raise ValueError(f"The {name} GridSpec requires n, d, and unit.")
    if grid.ndim != ndim:
        raise ValueError(f"The {name} GridSpec must have {ndim} dimensions.")
    try:
        dlu.canonical_unit(grid.unit, dimension="angle")
        is_angular = True
    except ValueError:
        is_angular = False
    if angular is not None and is_angular != angular:
        unit_type = "angular" if angular else "physical"
        raise ValueError(f"The {name} GridSpec must use {unit_type} units.")
    return is_angular


def _validate_method(method, grid, types):
    """Validate a propagation method and its required specification type."""
    method = str(method).lower()
    if method not in types:
        methods = "', '".join(types)
        raise ValueError(f"method must be '{methods}'.")
    if not isinstance(grid, types[method]):
        raise TypeError(
            f"{method.upper()} propagation requires a {types[method].__name__}."
        )
    return method


class ABCDElement(Base):
    """Base contract for a paraxial element represented by an ABCD matrix.

    Matrices act on physical ray coordinates and angles and compose algebraically in
    `ABCDPropagator`. Inverting a matrix is distinct from reverse propagation through
    an arbitrary optical system.
    """


class ABCDFreeSpace(ABCDElement):
    """Represent paraxial free-space propagation over a distance in metres.

    Positive and negative distances describe opposite directions within the ABCD
    system definition.
    """

    distance: float

    def __init__(self, distance):
        """Initialise free-space ABCD translation.

        Parameters
        ----------
        distance : float or Array, metres
            Signed propagation distance.
        """
        self.distance = dlu.to_value(distance)

    @property
    def abcd(self):
        """Return the ``(..., 2, 2)`` free-space ABCD matrix in metres."""
        return dlu.abcd_free_space(self.distance)


class ABCDLens(ABCDElement):
    """Represent an ideal paraxial thin lens with focal length in metres.

    The element changes angular slope while leaving the transverse coordinate
    continuous at the lens plane.
    """

    focal_length: float

    def __init__(self, focal_length):
        """Initialise a thin-lens ABCD element.

        Parameters
        ----------
        focal_length : float or Array, metres
            Signed lens focal length.
        """
        self.focal_length = dlu.to_value(focal_length)

    @property
    def abcd(self):
        """Return the ``(..., 2, 2)`` thin-lens ABCD matrix.

        Focal length is interpreted in metres and leading parameter axes are retained.
        """
        return dlu.abcd_lens(self.focal_length)


class ABCDMirror(ABCDElement):
    """Represent an ideal paraxial spherical mirror by its radius in metres.

    The radius sign follows the package ABCD convention and determines the surface's
    focusing or defocusing power.
    """

    radius: float

    def __init__(self, radius):
        """Initialise a spherical-mirror ABCD element.

        Parameters
        ----------
        radius : float or Array, metres
            Signed mirror radius of curvature.
        """
        self.radius = dlu.to_value(radius)

    @property
    def abcd(self):
        """Return the ``(..., 2, 2)`` curved-mirror ABCD matrix.

        Radius of curvature is interpreted in metres and leading axes are retained.
        """
        return dlu.abcd_mirror(self.radius)


class ABCDFraunhofer(ABCDElement):
    """Represent a focal Fourier transform with focal length in metres.

    This supplies the canonical mapping between conjugate pupil and focal coordinates
    inside `ABCDPropagator`.
    """

    focal_length: float

    def __init__(self, focal_length):
        """Initialise an ABCD Fraunhofer element.

        Parameters
        ----------
        focal_length : float or Array, metres
            Signed focal length defining the conjugate planes.
        """
        self.focal_length = dlu.to_value(focal_length)

    @property
    def abcd(self):
        """Return the ``(..., 2, 2)`` far-field ABCD matrix.

        Focal length is interpreted in metres and leading parameter axes are retained.
        """
        return dlu.abcd_fraunhofer(self.focal_length)


class Propagator(OpticalLayer):
    """Base contract for propagation to an explicit output sampling specification.

    Concrete propagators transform the complex field and its grid metadata together.
    Output units determine whether sampling is angular or physical; physical focal
    grids require the relevant focal length.
    """

    grid: BaseGridSpec

    def __init__(self, grid):
        """Initialise a propagation output specification.

        Parameters
        ----------
        grid : GridSpec or ResizeSpec
            Physical output sampling or FFT resize specification.
        """
        if not isinstance(grid, (GridSpec, ResizeSpec)):
            raise TypeError("grid must be a GridSpec or ResizeSpec.")
        self.grid = grid.broadcast(2)

    def apply(self, wavefront):
        """Propagate the complete wavefront without layer-level axis mapping.

        Propagators natively handle wavelength and sampling leading axes, so this
        method passes the complete `Wavefront` to `apply_mono` and returns the
        propagated immutable object.
        """
        return self.apply_mono(wavefront)

    def _validate(self, wavefront):
        """Validate the input coordinate specification."""
        _validate_grid(wavefront.grid, "input", angular=False)


class FocalPropagator(Propagator):
    """Base contract for reversible propagation between pupil and focal planes.

    The `inverse` leaf selects the physical direction implemented by the concrete
    Fourier algorithm. It does not invert arbitrary intervening optical layers.
    """

    grid: BaseGridSpec
    focal_length: Array | None
    inverse: bool

    def __init__(self, grid, focal_length=None, inverse=False):
        """Initialise a focal-plane propagator contract.

        Parameters
        ----------
        grid : GridSpec or ResizeSpec
            Output sampling appropriate to the concrete numerical method.
        focal_length : float, Array, or None, metres
            Optional focal length; omission uses angular focal coordinates.
        inverse : bool
            Propagate in the explicitly supported reverse direction.
        """
        super().__init__(grid)
        self.focal_length = dlu.to_value(focal_length, optional=True)
        self.inverse = bool(inverse)

    def _validate(self, wavefront):
        """Validate the input and explicitly requested output coordinates."""
        if isinstance(self.grid, ResizeSpec):
            angular = self.inverse and self.focal_length is None
            _validate_grid(wavefront.grid, "input", angular=angular)
            return
        if self.inverse:
            angular = self.focal_length is None
            _validate_grid(wavefront.grid, "input", angular=angular)
            _validate_grid(self.grid, "output", wavefront.grid.ndim, angular=False)
            return
        super()._validate(wavefront)
        angular = _validate_grid(self.grid, "output", wavefront.grid.ndim)
        if self.focal_length is None and not angular:
            raise ValueError(
                "Propagation without a focal length requires angular output units."
            )
        if self.focal_length is not None and angular:
            raise ValueError(
                "Propagation with a focal length requires physical output units."
            )


class Fraunhofer(FocalPropagator):
    """Propagate between conjugate planes using an MFT or FFT.

    Parameters
    ----------
    grid : GridSpec or ResizeSpec
        Explicit MFT output grid or FFT resizing specification.
    focal_length : float or None
        Focal length in meters. Omit for angular focal-plane coordinates.
    method : {"mft", "fft"}
        Numerical propagation method.
    inverse : bool
        Propagate from the focal plane back to a physical pupil plane.

    Examples
    --------
    Propagate with explicit MFT sampling, in reverse, or at native FFT sampling:

    ```python
    import dLux as dl

    # Construct the pupil and focal-plane grids
    pupil_grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    focal_grid = dl.GridSpec(n=64, d=10, unit="mas")

    # Construct the forward and inverse MFT propagators
    forward = dl.Fraunhofer(focal_grid)
    inverse = dl.Fraunhofer(pupil_grid, inverse=True)

    # Construct an apertured pupil wavefront
    pupil = dl.SimpleCircular(diameter=0.9)(pupil_grid)
    wavefront = dl.Wavefront(wavelength=650e-9, grid=pupil_grid)
    wavefront = pupil(wavefront)

    # Propagate to the focal plane and back to the pupil
    focal_wavefront = forward(wavefront)
    pupil_wavefront = inverse(focal_wavefront)

    # Alternatively, use native FFT sampling with two-times padding
    pad_spec = dl.ResizeSpec(pad=2)
    fft = dl.Fraunhofer(pad_spec, method="fft")
    fft_wavefront = fft(wavefront)
    ```
    """

    grid: BaseGridSpec
    focal_length: Array | None
    inverse: bool
    method: str

    def __init__(self, grid, focal_length=None, method="mft", inverse=False):
        """Initialise Fraunhofer propagation.

        Parameters
        ----------
        grid : GridSpec or ResizeSpec
            `GridSpec` for MFT output or `ResizeSpec` for FFT output.
        focal_length : float, Array, or None, metres
            Optional focal length for physical rather than angular output sampling.
        method : str
            ``"mft"`` or ``"fft"``.
        inverse : bool
            Use the corresponding reverse Fourier propagation.
        """
        method = _validate_method(method, grid, {"mft": GridSpec, "fft": ResizeSpec})
        super().__init__(grid, focal_length, inverse)
        self.method = method

    def apply_mono(self, wavefront):
        """Propagate a wavefront between conjugate planes.

        MFT propagation evaluates the configured explicit focal grid. FFT propagation
        uses the configured resize factors and its sampling-derived output grid.
        Forward propagation produces angular coordinates without ``focal_length`` and
        physical coordinates in metres when it is supplied; inverse propagation
        returns to the configured physical pupil grid.
        """
        self._validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront,
                self.grid,
                focal_length=self.focal_length,
                inverse=self.inverse,
            )
        return _propagate_mft(
            wavefront, self.grid, focal_length=self.focal_length, inverse=self.inverse
        )


class Fresnel(FocalPropagator):
    """Propagate between defocused focal planes using an FFT, MFT, or LCT.

    Parameters
    ----------
    grid : GridSpec or ResizeSpec
        Explicit MFT/LCT output grid or FFT resizing specification.
    defocus : float
        Longitudinal defocus distance in meters.
    focal_length : float or None
        Focal length in meters. Omit for angular focal-plane coordinates.
    method : {"fft", "mft", "lct"}
        Numerical propagation method.
    inverse : bool
        Reverse MFT or LCT propagation direction. Inverse FFT propagation is not
        currently supported.

    Examples
    --------
    Propagate to symmetric defocus planes with explicit or native FFT sampling:

    ```python
    import dLux as dl

    # Construct the pupil and defocused focal-plane grids
    pupil_grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    focal_grid = dl.GridSpec(n=64, d=10, unit="mas")

    # Construct propagators to either side of focus
    positive_defocus = dl.Fresnel(focal_grid, defocus=1e-3)
    negative_defocus = dl.Fresnel(focal_grid, defocus=-1e-3)

    # Construct an apertured pupil wavefront
    pupil = dl.SimpleCircular(diameter=0.9)(pupil_grid)
    wavefront = dl.Wavefront(wavelength=650e-9, grid=pupil_grid)
    wavefront = pupil(wavefront)

    # Propagate to the two defocused focal planes
    positive_wavefront = positive_defocus(wavefront)
    negative_wavefront = negative_defocus(wavefront)

    # Alternatively, use native FFT sampling with two-times padding
    pad_spec = dl.ResizeSpec(pad=2)
    fft = dl.Fresnel(pad_spec, defocus=1e-3, method="fft")
    fft_wavefront = fft(wavefront)
    ```
    """

    grid: BaseGridSpec
    focal_length: Array | None
    inverse: bool
    defocus: Array
    method: str

    def __init__(
        self,
        grid,
        defocus=0.0,
        focal_length=None,
        method="lct",
        inverse=False,
    ):
        """Initialise defocused focal-plane propagation.

        Parameters
        ----------
        grid : GridSpec or ResizeSpec
            Output sampling appropriate to the selected method.
        defocus : float or Array, metres
            Signed axial displacement from the focal plane.
        focal_length : float, Array, or None, metres
            Optional focal length for physical rather than angular coordinates.
        method : str
            ``"mft"``, ``"fft"``, or ``"lct"`` where supported by the grid.
        inverse : bool
            Use the explicitly supported reverse propagation.
        """
        types = {"fft": ResizeSpec, "mft": GridSpec, "lct": GridSpec}
        method = _validate_method(method, grid, types)
        if inverse and method == "fft":
            raise ValueError(
                "Inverse Fresnel propagation is not supported with method='fft'."
            )
        super().__init__(grid, focal_length, inverse)
        self.method = method
        self.defocus = dlu.to_value(defocus)

    def apply_mono(self, wavefront):
        """Propagate a wavefront between defocused focal planes.

        The configured propagation distance and focal length are in metres. MFT uses
        an explicit physical output grid, while FFT uses resize-derived sampling. The
        reverse route applies the physically supported inverse Fresnel propagation.
        """
        self._validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront,
                self.grid,
                focal_length=self.focal_length,
                defocus=self.defocus,
            )
        return _propagate_mft(
            wavefront,
            self.grid,
            focal_length=self.focal_length,
            defocus=self.defocus,
            inverse=self.inverse,
        )


class ABCDPropagator(Propagator):
    """Propagate through an ordered ABCD system using an LCT or FFT.

    Parameters
    ----------
    ABCDs : sequence or dict
        Ordered ``ABCDElement`` objects composing the optical system.
    grid : GridSpec or ResizeSpec
        Explicit LCT output grid or FFT resizing specification.
    method : {"lct", "fft"}
        Numerical propagation method.

    Examples
    --------
    Compose and propagate through a slightly defocused optical relay:

    ```python
    import dLux as dl

    # Construct the pupil and focal-plane sampling grids
    pupil_grid = dl.GridSpec(n=128, diam=0.01, unit="m")
    focal_grid = dl.GridSpec(n=128, diam=0.01, unit="m")

    # Construct a slightly defocused optical relay
    propagator = dl.ABCDPropagator(
        ABCDs=[
            ("space_in", dl.ABCDFreeSpace(distance=1.0)),
            ("lens", dl.ABCDLens(focal_length=0.5)),
            ("space_out", dl.ABCDFreeSpace(distance=1.0)),
            ("defocus", dl.ABCDFreeSpace(distance=0.01)),
        ],
        grid=focal_grid,
    )

    # Construct an apertured pupil wavefront
    pupil = dl.SimpleCircular(diameter=0.008)(pupil_grid)
    wavefront = dl.Wavefront(wavelength=650e-9, grid=pupil_grid)
    wavefront = pupil(wavefront)

    # Propagate through the composed optical train
    wavefront = propagator(wavefront)

    # Access the combined ABCD matrix
    abcd = propagator.abcd
    ```
    """

    grid: BaseGridSpec
    ABCDs: dict
    method: str

    def __init__(self, ABCDs, grid, method="lct"):
        """Initialise propagation through an ABCD optical train.

        Parameters
        ----------
        ABCDs : sequence of BaseABCD
            Ordered paraxial optical elements.
        grid : GridSpec or ResizeSpec
            Output sampling required by the selected method.
        method : str
            ``"lct"`` for general transforms or ``"fft"`` where supported.
        """
        super().__init__(grid)
        method = _validate_method(
            method, self.grid, {"lct": GridSpec, "fft": ResizeSpec}
        )

        elements = list(ABCDs.items()) if isinstance(ABCDs, dict) else ABCDs
        self.ABCDs = dlu.list2dictionary(
            elements, ordered=True, allowed_types=(ABCDElement,)
        )
        if not self.ABCDs:
            raise ValueError("ABCDs must contain at least one element.")
        self.method = method

    def __getattr__(self, key):
        """Raise grid parameters, named elements, and element parameters."""
        return dlu.resolve_attr(self, key, self.grid, self.ABCDs)

    @property
    def abcd(self) -> Array:
        """Return the ordered product of all configured ABCD elements.

        The result has shape ``(..., 2, 2)`` and preserves broadcast leading axes.
        """
        return dlu.compose_abcd([element.abcd for element in self.ABCDs.values()])

    def _validate(self, wavefront):
        """Validate physical ABCD input and output coordinates."""
        Propagator._validate(self, wavefront)
        if isinstance(self.grid, ResizeSpec):
            return
        _validate_grid(self.grid, "output", wavefront.grid.ndim, angular=False)

    def apply_mono(self, wavefront):
        """Propagate a wavefront through the composed ABCD system.

        LCT mode evaluates the configured physical output grid. FFT mode uses a
        resize specification and is valid only for ABCD systems supported by that
        numerical route. The returned wavefront stores the realised output sampling.
        """
        self._validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront, self.grid, unit=wavefront.grid.unit, ABCD=self.abcd
            )
        return _propagate_mft(wavefront, self.grid, ABCD=self.abcd)


class FreeSpace(Propagator):
    """Paraxial angular-spectrum propagation over a free-space distance.

    Parameters
    ----------
    distance : float
        Signed propagation distance in meters.
    grid : ResizeSpec or None
        Optional padding, cropping, or output-size specification.
    crop : bool
        Crop the propagated array according to ``grid``.

    Examples
    --------
    Propagate forwards and backwards with padded angular-spectrum calculations:

    ```python
    import dLux as dl

    # Construct padded forward and reverse free-space propagators
    resize = dl.ResizeSpec(pad=4, crop=4)
    forward = dl.FreeSpace(distance=100.0, grid=resize)
    reverse = dl.FreeSpace(distance=-100.0, grid=resize)

    # Construct an apertured wavefront
    grid = dl.GridSpec(n=128, diam=0.01, unit="m")
    pupil = dl.SimpleCircular(diameter=0.008)(grid)
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = pupil(wavefront)

    # Propagate forwards and then in the reverse axial direction
    wavefront = forward(wavefront)
    wavefront = reverse(wavefront)
    ```
    """

    grid: BaseGridSpec
    distance: Array
    crop: bool

    def __init__(self, distance, grid=None, crop=True):
        """Initialise angular-spectrum free-space propagation.

        Parameters
        ----------
        distance : float or Array, metres
            Signed propagation distance.
        grid : ResizeSpec or None
            Optional padding and cropping specification.
        crop : bool
            Crop back to the pre-padding spatial size when true.
        """
        if grid is None:
            grid = ResizeSpec()
        if not isinstance(grid, ResizeSpec):
            raise TypeError("FreeSpace grid must be a ResizeSpec.")
        super().__init__(grid)
        self.distance = dlu.to_value(distance)
        self.crop = bool(crop)

    def apply_mono(self, wavefront):
        """Propagate a wavefront over the configured free-space distance.

        Distance is measured in metres; negative values naturally propagate in the
        reverse axial direction. Padding occurs before angular-spectrum propagation
        and optional cropping afterward. The returned grid records the realised size.
        """
        self._validate(wavefront)

        # Define and vectorise monochromatic angular-spectrum propagation
        wavelength, x, y = _propagation_inputs(wavefront)
        prop_fn = lambda field, lam, x, y: dlu.ASM(
            field, lam, (x, y), self.distance, crop=False, **self.grid._padding
        )
        propagate = np.vectorize(prop_fn, signature="(n,m),(),(m),(n)->(p,q)")
        field = propagate(wavefront.phasor, wavelength, x, y)

        # Apply optional output cropping and update the realised grid
        field = self.grid.crop_array(field) if self.crop else field
        grid = wavefront.grid.resize(field.shape[-2:][::-1])
        return wavefront.set(phasor=field, grid=grid)
