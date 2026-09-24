"""Image formation, coronagraphy and sparse interferometric imaging."""

from __future__ import annotations

from collections.abc import Mapping
import zodiax as zdx

import jax.numpy as np
import jax.random as jr
from jax import Array

import dLux.utils as dlu

from ..fields import Image, Intensity, Wavefront, PSF
from ..utils.sequential import Sequential
from ..layers.optical import Optic, OpticalLayer, PropagatorLayer
from ..prebuilt.definitions import Def
from ..layers.sparse import SparseOptic
from ..layers.coronagraphy import SoummerFPMLayer
from .sources import Source, ResolvedSource, Scene
from ..utils.photometry import BaseSpectrum, Spectrum

from .detector import Detector, DetectorSystem
from .optical import OpticalSystem
from .modelling import (
    render_sources,
    source_samples,
    source_profile,
    spectral_weights,
    integrate_psf,
)

__all__ = [
    "Imager",
    "Coronagraph",
    "Interferometer",
]


def _normalise_grid(grid, name="plane grid") -> dlu.Grid:
    """Validate and return a complete two-dimensional plane grid."""
    if not isinstance(grid, dlu.Grid):
        raise TypeError(f"{name} must be a Grid.")
    if grid.n is None or grid.d is None or grid.unit is None:
        raise ValueError(f"{name} must define n, d, and unit.")
    if dlu.unit_dimension(grid.unit) not in ("length", "angle"):
        raise ValueError(f"{name} must use physical length or angular units.")
    return grid


def _normalise_input_grid(grid) -> dlu.Grid:
    """Validate a complete two-dimensional physical system-input grid."""
    grid = _normalise_grid(grid, "input grid")
    if dlu.unit_dimension(grid.unit) != "length":
        raise ValueError("input grid must use physical length units.")
    return grid


def _sequence_layers(layers, layer_type=dlu.BaseLayer):
    """Validate nested collections without admitting complete optical systems."""
    entries = list(layers.items()) if isinstance(layers, dict) else layers
    layers = dlu.list2dictionary(entries, True)
    for layer in layers.values():
        if isinstance(layer, (Imager, OpticalSystem)):
            raise TypeError("Complete optical systems cannot be used as layers.")
        if isinstance(layer, Sequential):
            if layer_type is OpticalLayer and isinstance(layer, DetectorSystem):
                raise TypeError("Detector sequences cannot be used as optical layers.")
            _sequence_layers(layer.layers, layer_type)
        elif not isinstance(layer, layer_type):
            raise TypeError(f"Expected {layer_type}, got {type(layer)}.")
    return layers


def _optical_leaves(component):
    """Yield optical leaves after rejecting nested complete systems."""
    layers = _sequence_layers([component], OpticalLayer)
    for layer in layers.values():
        if isinstance(layer, Sequential):
            for child in layer.layers.values():
                yield from _optical_leaves(child)
        else:
            yield layer


def _plane_slot(component):
    """Reject known propagation and aperture-axis changes within local slots."""
    if component is not None:
        for layer in _optical_leaves(component):
            if isinstance(layer, SparseOptic) or (
                isinstance(layer, PropagatorLayer)
                and (not isinstance(layer, Optic) or layer.grid_out is not None)
            ):
                raise TypeError("Plane slots require grid-preserving optical layers.")
    return component


def _pupil_slot(component):
    """Allow one sparse pupil, followed by operations on its local fields."""
    if component is not None:
        for index, layer in enumerate(_optical_leaves(component)):
            if index == 0 and isinstance(layer, SparseOptic):
                if layer.grid_out is not None:
                    raise TypeError("Imager owns propagation after its sparse pupil.")
                continue
            _plane_slot(layer)
    return component


def _apply_plane(component, wavefront, **context):
    """Apply a local component and verify its resulting sampling."""
    if component is None:
        return wavefront
    _plane_slot(component)
    output = component(wavefront, **context)
    if not isinstance(output, Wavefront):
        raise TypeError("An optical plane must return a Wavefront.")
    before = wavefront.phasor.shape[: wavefront.batch_ndim]
    after = output.phasor.shape[: output.batch_ndim]
    if after != before:
        raise ValueError(
            "Optical planes must preserve wavelength and source axes. "
            "Use vmap outside Imager for independent optical models."
        )
    output.grid.check_sampling(
        wavefront.grid, "Optical plane transformations must preserve the grid."
    )
    return output.set(grid=wavefront.grid)


def _focal_propagator(grid, propagator):
    """Validate an optional custom MFT against the declared image grid."""
    grid = _normalise_grid(grid, "focal grid")
    if not isinstance(propagator, (dlu.Propagator, dlu.SpectralPropagator)):
        raise TypeError(
            "A custom imaging relay must be a Propagator or SpectralPropagator."
        )
    if isinstance(propagator, dlu.Propagator) and propagator.inverse:
        raise TypeError("The focal propagator must be forward.")
    output_grid = _normalise_grid(propagator.grid_out, "propagator grid_out")
    grid.check_sampling(
        output_grid, "propagator grid_out must match the declared focal grid."
    )
    return propagator


def _relay_parameters(optic):
    """Read one relay owner, or explicit user aliases on a pupil sequence."""
    defaults = dict(focal_length=None, abcd=None, defocus=None, inverse=False)
    owner, aliases = optic, {}
    if isinstance(optic, Sequential):
        # A sequence's relay belongs to one optical definition. Default-valued
        # optics do not introduce another physical relay or require extra aliases.
        owners, active_names = [], set()
        for layer in _optical_leaves(optic):
            if not isinstance(layer, PropagatorLayer):
                continue
            active = set()
            for name in defaults:
                value = getattr(layer, name)
                if (name == "inverse" and value) or (
                    name != "inverse" and value is not None
                ):
                    active.add(name)
            if active:
                owners.append(layer)
                active_names.update(active)
        aliases = dict(optic.alias or ())
        if len(owners) > 1 and not active_names.issubset(aliases):
            raise ValueError(
                "The pupil sequence has multiple relay definitions. Keep physical "
                "propagation parameters on one Optic, or provide explicit aliases "
                "for its active relay parameters."
            )
        owner = owners[0] if len(owners) == 1 else None

    # Read relay parameters from the chosen owner, respecting explicit aliases.
    parameters = {}
    for name, default in defaults.items():
        if name in aliases:
            parameters[name] = getattr(optic, name)
        else:
            parameters[name] = (
                default if owner is None else getattr(owner, name, default)
            )
    return parameters


def _imaging_propagator(grid, optic):
    """Derive a forward relay from output sampling and flat optical parameters."""
    parameters = _relay_parameters(optic)
    abcd, defocus, inverse = (
        parameters["abcd"],
        parameters["defocus"],
        parameters["inverse"],
    )

    # Require a forward relay and a physical focal scale for length-valued image grids.
    if inverse:
        raise ValueError("An imaging relay must be forward.")
    focal_length = None
    if dlu.unit_dimension(grid.unit) == "length":
        if optic is None:
            raise ValueError("A Cartesian image grid requires a pupil focal_length.")
        focal_length = parameters["focal_length"]
        if focal_length is None and abcd is None:
            raise ValueError("A Cartesian image grid requires Optic(focal_length=...).")
    return dlu.Propagator(
        grid,
        focal_length=focal_length,
        defocus=defocus,
        abcd=abcd,
        inverse=inverse,
    )


def _channels(system, defaults):
    """Pair source/filter/bandpass keys, using stored defaults for missing keys."""
    fields = ("source", "filter", "bandpass")
    names = []
    for field in fields:
        value = getattr(system, field)
        if isinstance(value, Mapping):
            for name in value:
                if not isinstance(name, str):
                    raise TypeError("Output channel names must be strings.")
                if name not in names:
                    names.append(name)

    # Select raw definitions before populating each channel's shared owner.
    channels = {}
    for name in names:
        selected = {}
        for field in fields:
            value = getattr(system, field)
            if isinstance(value, Mapping):
                default = getattr(defaults, field)
                if isinstance(default, Mapping):
                    default = default.get(name)
                value = value.get(name, default)
            selected[field] = value
        channels[name] = system.set(**selected)
    return channels


def _filter_input(value):
    """Store a numerical response or named independent response definitions."""
    if isinstance(value, Mapping):
        return {name: _filter_input(child) for name, child in value.items()}
    if value is None or isinstance(value, dlu.Filter):
        return value
    if isinstance(value, zdx.Module) and not isinstance(value, zdx.Expression):
        raise TypeError("filter must be a Filter or numerical response.")
    return dlu.Filter(value)


def _bandpass_input(value):
    """Store wavelength sampling without evaluating expressions."""
    if isinstance(value, Mapping):
        return {name: _bandpass_input(child) for name, child in value.items()}
    if isinstance(value, dlu.UnitSpec):
        return value
    if isinstance(value, zdx.Module) and not isinstance(value, zdx.Expression):
        raise TypeError("bandpass must be a wavelength UnitSpec, array or Expression.")
    return zdx.as_array(value, dtype=float)


class Imager(zdx.Module):
    """Compose optical point responses, source photometry and detection.

    ``wavefront`` propagates unweighted fields; ``psf`` samples their intensity at
    optical wavelengths without source brightness. ``photometry`` independently
    calculates component-by-wavelength photon weights. ``intensity`` combines
    those stages and source morphology on the optical samples. ``image`` and ``model``
    then run the detector; ``simulate`` additionally draws photon and read noise.

    BandPass sets optical sampling; native spectral curves set photometric detail.
    Fine curves are integrated into weights around the optical nodes. Source
    lists and Scenes add incoherently; named model-input dictionaries select independent
    outputs. Entrance normalisation is conditional on the aperture; later optical,
    filter and detector losses retain their scale.
    """

    pupil_grid: dlu.Grid
    pupil: OpticalLayer | Sequential | None = None
    image_grid: dlu.Grid
    source: Source | Scene | list | dict | None = None
    filter: dlu.Filter | dict | None = None
    bandpass: dlu.UnitSpec | Array | zdx.Expression | dict | None = None
    detector: Detector | None = None
    jitter: Array | zdx.Expression | None = None
    oversample: tuple[int, int] = (1, 1)

    def __init__(
        self,
        pupil_grid,
        image_grid,
        pupil=None,
        *,
        source=None,
        filter=None,
        bandpass=None,
        detector=None,
        jitter=None,
        oversample=1,
        alias=None,
    ):
        """Construct an imaging model with optional stored modelling inputs.

        Parameters
        ----------
        pupil_grid : Grid
            Two-dimensional physical pupil sampling.
        image_grid : Grid
            Native image-pixel sampling for every wavelength. Angular units select an
            angular MFT; length units use the pupil's focal_length. A sequence
            with multiple focal lengths needs an explicit user-defined alias.
        pupil : optical layer, Sequential, or None
            Grid-preserving pupil operations, or a SparseOptic followed by local
            optics. For a sparse pupil, pupil_grid samples one aperture patch;
            the system coherently combines patches on the common image grid.
            None transmits the incident field.
        source : Source or Scene or None
            Stored source; methods may supply a replacement.
        filter : Filter or None
            Additional absolute spectral transmission; no renormalisation follows.
        bandpass : BandPass, wavelength array, or None
            BandPass defines optical samples and photometric bin edges. Arrays
            support wavefront/PSF calculations only. None uses the spectrum's stored
            sampling, which must be a BandPass for intensity or image calculations.
        detector : Detector or None
            Detector sampling, spectral QE, deterministic layers and read noise.
        jitter : Array, Expression, or None
            Centred, unit-sum image-blur kernel. Expressions resolve on the image
            grid after spectral integration, before detector layers and integration.
            Pixel widths and array-kernel offsets refer to native image pixels.
        oversample : int or pair of ints
            Positive optical subpixel factors in physical (x, y) order. Propagation
            uses the finer grid; detected images integrate back to pixels.
            Detector.downsample specifies additional detector binning.
        alias : sequence or None
            User-defined parameter aliases.
        """
        # Check model input types before constructing pupil or detector factories.
        for name, value, allowed in (
            ("filter", filter, (dlu.Filter, zdx.Expression)),
            ("bandpass", bandpass, (dlu.UnitSpec, zdx.Expression)),
        ):
            if isinstance(value, zdx.Module) and not isinstance(value, allowed):
                raise TypeError(
                    f"{name} received an incompatible {type(value).__name__}."
                )

        # Materialise factories on their owner grid and retain the modelling inputs.
        self.pupil_grid = _normalise_input_grid(pupil_grid)
        if isinstance(pupil, Def):
            pupil = pupil(grid=self.pupil_grid)
        self.pupil = _pupil_slot(pupil)
        self.image_grid = _normalise_grid(image_grid, "image grid")
        self.source = source
        self.filter = _filter_input(filter)
        self.bandpass = _bandpass_input(bandpass)
        self.detector = detector() if isinstance(detector, Def) else detector
        self.jitter = zdx.as_array(jitter, dtype=float)
        self.oversample = dlu.as_size(oversample, 2, "oversample")
        self.alias = alias
        self._check_inputs()
        _ = self.propagator  # Validate the derived relay after storing its inputs.

    def _check_inputs(self):
        """Check known types and shapes without resolving deferred expressions."""
        if self.source is not None:
            self._check_source(self.source)
        if self.detector is not None and not isinstance(self.detector, Detector):
            raise TypeError("detector must be a Detector.")

        # Validate each stored filter and bandpass, including named output collections.
        filters = (
            self.filter.values() if isinstance(self.filter, Mapping) else (self.filter,)
        )
        for response in filters:
            if response is not None and not isinstance(response, dlu.Filter):
                raise TypeError("filter must be a Filter or a spectral response array.")
        bandpasses = (
            self.bandpass.values()
            if isinstance(self.bandpass, Mapping)
            else (self.bandpass,)
        )
        for bandpass in bandpasses:
            if isinstance(bandpass, zdx.Module) and not isinstance(
                bandpass, (dlu.UnitSpec, zdx.Expression)
            ):
                raise TypeError(
                    "bandpass must be a wavelength UnitSpec, array or Expression."
                )
            if bandpass is not None and not isinstance(bandpass, zdx.Expression):
                if bandpass.ndim > 1:
                    raise ValueError(
                        "bandpass must describe one-dimensional wavelength sampling."
                    )
                if isinstance(bandpass, Array) and bandpass.size == 0:
                    raise ValueError("bandpass must contain at least one wavelength.")
                if isinstance(bandpass, dlu.UnitSpec) and bandpass.unit is not None:
                    dlu.canonical_unit(
                        bandpass.unit, dimension="length", name="bandpass unit"
                    )

        # Require a spatial jitter kernel and one common input/output grid per Imager.
        jitter = self.jitter
        if jitter is not None and not isinstance(jitter, zdx.Expression):
            if jitter.ndim != 2 or 0 in jitter.shape:
                raise ValueError("jitter must be a non-empty two-dimensional kernel.")
        for name, grid in (("pupil", self.pupil_grid), ("output", self.image_grid)):
            for value in (grid.d, grid.c):
                if value is not None and not isinstance(value, zdx.Expression):
                    if value.ndim != 1:
                        raise ValueError(
                            f"Imager requires one common {name} grid for all samples. "
                            "Use vmap outside Imager for independent grids."
                        )

    @property
    def grid_in(self):
        """Physical entrance sampling, stored as pupil_grid."""
        return self.pupil_grid

    @property
    def propagation_grid(self):
        """Fine image sampling, with the native field of view and centre."""
        return self.image_grid.oversample(self.oversample)

    @property
    def propagator(self):
        """Derived MFT relay; only the image grid and optical parameters are stored."""
        return _imaging_propagator(self.propagation_grid, self.pupil)

    @property
    def _sparse_pupil(self):
        """The leading sparse optic, or None for a full pupil grid."""
        if self.pupil is None:
            return None
        first = next(_optical_leaves(self.pupil), None)
        return first if isinstance(first, SparseOptic) else None

    def apply_pupil(self, wavefront, offset=None, *, unit_input=False, **context):
        """Apply pupil optics and source tilt, returning a reusable pupil field.

        Sparse pupils retain their aperture axis here; interference happens during
        propagate_image. unit_input distributes unit total power across patches.
        A subclass can pass this immutable field to multiple downstream branches.
        """
        _pupil_slot(self.pupil)
        self = self.populate()
        expected = wavefront.batch_shape
        populations = wavefront.batch_ndim - wavefront.wavelength.ndim
        sparse = self._sparse_pupil
        if sparse is not None:
            # Source flux describes the whole pupil, not each copied patch.
            if unit_input:
                wavefront = wavefront.set(
                    phasor=wavefront.phasor / np.sqrt(sparse.n_apertures)
                )
            wavefront = self.pupil(wavefront, **context)
            expected = (
                expected[:populations] + (sparse.n_apertures,) + expected[populations:]
            )
            if wavefront.phasor.shape[: wavefront.batch_ndim] != expected:
                raise ValueError(
                    "A sparse pupil must add exactly one aperture axis between "
                    "the source and wavelength axes."
                )
            if offset is not None:
                # The final population axis is aperture, even when its size equals
                # the source or wavelength count. Include each centre's piston.
                if offset.ndim > 1:
                    axis = -2 - wavefront.wavelength.ndim
                    offset = np.expand_dims(offset, axis)
                wavefront = wavefront.tilt(offset)
        else:
            if offset is not None:
                wavefront = wavefront.tilt(offset)
            wavefront = _apply_plane(self.pupil, wavefront, **context)

        return wavefront

    def propagate_image(self, wavefront, **context):
        """Propagate a prepared pupil field, then coherently combine sparse patches."""
        self = self.populate()
        expected = wavefront.batch_shape
        wavefront = wavefront.propagate(self.propagator, **context)
        if wavefront.batch_shape != expected:
            raise ValueError(
                "An imaging relay must preserve wavelength, source and aperture axes. "
                "Use vmap outside Imager for independent propagation parameters."
            )
        if self._sparse_pupil is not None:
            wavefront = wavefront.coherent_sum()
        return wavefront

    def _propagate(self, wavefront, offset=None, *, unit_input=False, **context):
        """Compose the pupil and image stages; subclasses may branch between them."""
        wavefront = self.apply_pupil(
            wavefront, offset, unit_input=unit_input, **context
        )
        return self.propagate_image(wavefront, **context)

    def apply(self, wavefront, **context):
        """Apply pupil optics and the MFT to an existing unweighted Wavefront.

        A sparse pupil copies the supplied incident patch to every aperture. Use
        source modelling for plane-wave illumination with global source offsets.
        """
        system = self.populate()
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront instance.")
        grid = system.pupil_grid.resolve(**context)
        wavefront.grid.check_sampling(
            grid, "wavefront grid must match the OpticalSystem input grid."
        )
        return system._propagate(wavefront, **context)

    def initialise_wavefront(self, wavelength, offset=None, **context):
        """Construct unit incident power on the pupil grid, optionally tilted."""
        grid = self.populate().pupil_grid.resolve(**context)
        wavefront = Wavefront(wavelength, grid)
        if offset is not None:
            offset = np.asarray(offset)
            if offset.shape != (2,):
                raise ValueError("offset must have shape (2,).")
            wavefront = wavefront.tilt(offset)
        return wavefront

    def propagate_mono(self, wavelength, offset=None, stokes=None, **context):
        """Return an unweighted monochromatic PSF on the fine optical grid."""
        wavelength = np.asarray(wavelength)
        if wavelength.ndim != 0:
            raise ValueError("wavelength must be scalar.")
        return self._point_wavefront(wavelength, offset, **context).to_psf(stokes)

    def propagate(self, wavelengths, offset=None, weights=None, stokes=None, **context):
        """Return integrated point-source Intensity on the fine grid before detection.

        Explicit weights retain their total; omission uses equal unit-sum weights.
        Source modelling, filters and detector operations use ``intensity`` instead.
        """
        wavelengths = np.atleast_1d(np.asarray(wavelengths))
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty one-dimensional array.")
        if weights is None:
            weights = np.ones_like(wavelengths) / wavelengths.size
        else:
            weights = np.atleast_1d(np.asarray(weights))
        if weights.shape != wavelengths.shape or np.iscomplexobj(weights):
            raise ValueError("weights must be real and match wavelengths.shape.")
        psf = self._point_wavefront(wavelengths, offset, **context).to_psf(stokes)
        if psf.grid.d.ndim != 1 or (psf.grid.c is not None and psf.grid.c.ndim != 1):
            raise ValueError(
                "Point-response integration requires one common output grid."
            )
        data = (psf.data * weights[..., None, None]).sum(-3)
        return Intensity(data, psf.grid)

    def _point_wavefront(self, wavelength, offset=None, **context):
        """Propagate unit incident power, retaining sparse-aperture phase offsets."""
        system = self.populate()
        wavefront = system.initialise_wavefront(wavelength, **context)
        offset = None if offset is None else np.asarray(offset)
        wavefront = system._propagate(wavefront, offset, unit_input=True, **context)
        grid = system.propagation_grid.resolve(**context)
        wavefront.grid.check_sampling(grid, "Output grid changed.")
        return wavefront.set(grid=grid)

    def __call__(self, source=None, **kwargs):
        """Run the stored model, with optional model overrides, and return an Image."""
        return self.model(source, **kwargs)

    def wavefront(
        self, source=None, *, bandpass=None, pupil=None, optics=None, **kwargs
    ):
        """Return unweighted complex fields at the optical wavelength samples.

        A Source contributes its positions only. Lists and Scenes retain their
        structure; named source/filter/bandpass dictionaries select independent
        outputs. Other keyword inputs follow ``model``'s override contract.
        """
        return self._run(
            "wavefront", source, bandpass=bandpass, pupil=pupil, optics=optics, **kwargs
        )

    def psf(
        self,
        source=None,
        *,
        bandpass=None,
        pupil=None,
        optics=None,
        stokes=None,
        **kwargs,
    ):
        """Return unweighted optical responses shaped (source, wavelength, y, x).

        The BandPass centres set optical wavelengths. Spectra, fluxes, filters,
        resolved profiles, jitter and detector
        response do not change this result. Sampling remains optically oversampled.
        """
        return self._run(
            "psf",
            source,
            bandpass=bandpass,
            pupil=pupil,
            optics=optics,
            stokes=stokes,
            **kwargs,
        )

    def photometry(self, source=None, *, bandpass=None, filter=None, **kwargs):
        """Return Weights without propagating an optical field.

        Each Weights value has shape (source, wavelength). Incident brightness,
        filter transmission and detector QE enter once here. Lists and Scenes keep
        their structure, so their different spectra remain independently inspectable.
        """
        return self._run(
            "photometry", source, bandpass=bandpass, filter=filter, **kwargs
        )

    def intensity(
        self, source=None, *, bandpass=None, filter=None, optics=None, **kwargs
    ):
        """Return fine expected photoelectrons before detector layers or readout.

        Source contributions add incoherently. Resolved morphology, optional
        complex field resampling and image jitter are included, but pixel
        integration, detector layers, gain and noise belong to ``image``.
        """
        return self._run(
            "intensity",
            source,
            bandpass=bandpass,
            filter=filter,
            optics=optics,
            **kwargs,
        )

    def image(self, source=None, *, bandpass=None, filter=None, **kwargs):
        """Return the expected detector Image from the same staged calculation."""
        return self._run("image", source, bandpass=bandpass, filter=filter, **kwargs)

    def simulate(self, key, source=None, *, n_frames=1, **kwargs):
        """Draw photon counts, read noise and gain after summing source intensities."""
        return self._run("image", source, key=key, n_frames=n_frames, **kwargs)

    def model(
        self,
        source=None,
        *,
        bandpass=None,
        filter=None,
        pupil=None,
        optics=None,
        pupil_grid=None,
        image_grid=None,
        detector=None,
        jitter=None,
        oversample=None,
        stokes=None,
        **context,
    ):
        """Calculate an Image using optional replacements for stored model inputs.

        None retains the stored choice. A source list or Scene adds intensities;
        plain dictionaries in source, filter or bandpass instead name independent
        outputs. Matching names pair, scalar inputs are shared, and missing entries
        use stored defaults. There is no Cartesian product between dictionaries.
        Overrides apply to an immutable copy before shared links are populated.
        """
        return self._run(
            "image",
            source,
            bandpass=bandpass,
            filter=filter,
            pupil=pupil,
            optics=optics,
            pupil_grid=pupil_grid,
            image_grid=image_grid,
            detector=detector,
            jitter=jitter,
            oversample=oversample,
            stokes=stokes,
            **context,
        )

    def _run(
        self,
        stage,
        source=None,
        *,
        spectrum=None,
        bandpass=None,
        filter=None,
        pupil=None,
        optics=None,
        pupil_grid=None,
        image_grid=None,
        detector=None,
        jitter=None,
        oversample=None,
        stokes=None,
        key=None,
        n_frames=1,
        **context,
    ):
        """Choose model inputs and named outputs before running one fixed stage."""

        # Collect non-None call-time overrides without modifying the stored model.
        if pupil is not None and optics is not None:
            raise ValueError("Supply pupil or optics, not both.")
        pupil = optics if optics is not None else pupil
        changes = {}
        for name, value in (
            ("source", source),
            ("filter", filter),
            ("bandpass", bandpass),
            ("detector", detector),
            ("jitter", jitter),
            ("pupil", pupil),
        ):
            if value is not None:
                changes[name] = value
        if pupil_grid is not None:
            changes["pupil_grid"] = _normalise_input_grid(pupil_grid)
        if image_grid is not None:
            changes["image_grid"] = _normalise_grid(image_grid, "image grid")
        if oversample is not None:
            changes["oversample"] = dlu.as_size(oversample, 2)
        if filter is not None:
            changes["filter"] = _filter_input(filter)
        if bandpass is not None:
            changes["bandpass"] = _bandpass_input(bandpass)
        if pupil is not None:
            changes["pupil"] = _pupil_slot(pupil)
        if jitter is not None:
            changes["jitter"] = zdx.as_array(jitter, dtype=float)
        system = self.set(**changes) if changes else self

        # Match source/filter/bandpass collections by key, not by Cartesian product.
        channels = _channels(system, self)
        if not channels:
            return system._single(
                stage,
                spectrum=spectrum,
                stokes=stokes,
                key=key,
                n_frames=n_frames,
                **context,
            )

        # Give each named output its own simulation key and selected model inputs.
        outputs = {}
        keys = [None] * len(channels) if key is None else jr.split(key, len(channels))
        for (name, channel), channel_key in zip(channels.items(), keys):
            try:
                outputs[name] = channel._single(
                    stage,
                    spectrum=spectrum,
                    stokes=stokes,
                    key=channel_key,
                    n_frames=n_frames,
                    **context,
                )
            except (TypeError, ValueError) as error:
                raise type(error)(f"Channel {name!r}: {error}") from error
        return outputs

    def _single(
        self, stage, *, spectrum=None, stokes=None, key=None, n_frames=1, **context
    ):
        """Compose the requested stages for one populated output channel."""

        # Prepare one source with any spectrum override, then populate shared links.
        source = Source() if self.source is None else self.source
        if spectrum is not None:
            if isinstance(source, (Scene, list, tuple)):
                raise TypeError("Set spectra on individual sources in a collection.")
            if isinstance(spectrum, zdx.Module) and not isinstance(
                spectrum, (BaseSpectrum, zdx.Expression)
            ):
                raise TypeError("spectrum must be a Spectrum or numerical definition.")
            spectrum = (
                spectrum if isinstance(spectrum, BaseSpectrum) else Spectrum(spectrum)
            )
            source = source.set(spectrum=spectrum)
        self = self.set(source=source)
        self._check_inputs()
        self = self.populate()

        # Return the requested stage; apply blur only after summing intensity.
        result = render_sources(self, stage, self.source, stokes=stokes, **context)
        if stage in ("wavefront", "psf", "photometry"):
            return result
        result = self._blur(result, **context)
        if stage == "intensity":
            return result
        return self._image(
            intensity=result,
            detector=self.detector,
            key=key,
            n_frames=n_frames,
            **context,
        )

    def _render_source(self, stage, source, offset=None, stokes=None, **context):
        """Compose optical responses and photometry for one source definition."""
        qe = None if self.detector is None else self.detector.qe
        if stage == "photometry":
            return self._photometry(
                source=source,
                bandpass=self.bandpass,
                filter=self.filter,
                qe=qe,
                **context,
            )

        # Optical-only requests use wavelength points, without evaluating spectra.
        wavelengths, positions = source_samples(
            source, self.bandpass, offset, **context
        )
        weights = None
        if stage not in ("wavefront", "psf"):
            weights = self._photometry(
                source=source,
                bandpass=self.bandpass,
                filter=self.filter,
                qe=qe,
                **context,
            )
        wavefront = self._wavefront(
            wavelengths=wavelengths, positions=positions, **context
        )
        if stage == "wavefront":
            return wavefront
        psf = self._psf(wavefront=wavefront, positions=positions, stokes=stokes)
        if stage == "psf":
            return psf

        # Sample source morphology before detector resampling and photon weighting.
        profile = source_profile(
            source, wavefront.grid, self.propagator, wavelengths=wavelengths, **context
        )
        return self._intensity(
            psf=psf,
            weights=weights,
            profile=profile,
            wavefront=wavefront,
            detector=self.detector,
            oversample=self.oversample,
            stokes=stokes,
            **context,
        )

    def _psf(self, wavefront, positions, stokes=None):
        """Convert optical fields to unweighted point responses."""
        return wavefront.to_psf(stokes).set(positions=positions)

    def _photometry(self, source, bandpass, filter=None, qe=None, **context):
        """Integrate fine spectral curves onto the optical BandPass bins."""
        return spectral_weights(source, bandpass, filter, qe, **context)

    def _intensity(
        self,
        psf,
        weights,
        *,
        profile=None,
        wavefront=None,
        detector=None,
        oversample=1,
        stokes=None,
        **context,
    ):
        """Combine supplied optical responses, photon counts and source morphology."""
        if profile is not None and profile.ndim == 3:
            if profile.shape[0] != psf.data.shape[-3]:
                raise ValueError(
                    "A chromatic profile must match the PSF wavelength axis."
                )

        # Morphology broadens intensity. Only reconstruct an effective complex field
        # when the detector explicitly requests complex resampling.
        resample = detector is not None and detector.resample is not None
        if profile is not None and not resample:
            psf = psf.convolve(profile)
        elif resample:
            if profile is not None:
                if wavefront.jones is True:
                    raise ValueError(
                        "Complex detector resampling of a resolved Jones field "
                        "requires a polarisation reconstruction model. "
                        "Pupil-plane Jones fields support resolved intensity "
                        "convolution without Detector.resample."
                    )
                amplitude = np.abs(wavefront.phasor)
                broadened = PSF(amplitude**2, wavefront.grid).convolve(profile).data
                # This is an effective field: retain the point-field phase and use
                # zero phase where an initially dark pixel acquires intensity.
                phase = np.where(
                    amplitude > 0,
                    wavefront.phasor / np.where(amplitude > 0, amplitude, 1),
                    1,
                )
                positive = broadened > 0
                amplitude = np.where(
                    positive, np.sqrt(np.where(positive, broadened, 1)), 0
                )
                wavefront = wavefront.set(phasor=amplitude * phase)
            wavefront = detector.sample(wavefront, oversample=oversample, **context)
            psf = self._psf(wavefront, psf.positions, stokes)

        # Apply source photons and spectral losses after the spatial operations.
        return integrate_psf(psf, weights)

    def _check_source(self, source):
        """Check ordinary source structure without resolving local definitions."""
        if isinstance(source, (list, tuple)):
            if not source:
                raise ValueError("A source collection cannot be empty.")
            for child in source:
                self._check_source(child)
        elif isinstance(source, Mapping):
            if not source:
                raise ValueError("Named source outputs cannot be empty.")
            for name, child in source.items():
                if not isinstance(name, str):
                    raise TypeError("Output channel names must be strings.")
                self._check_source(child)
        elif isinstance(source, Scene):
            if not isinstance(source.sources, Mapping) or not source.sources:
                raise ValueError("Scene sources must be a non-empty named mapping.")
            for name, child in source.sources.items():
                if not isinstance(name, str):
                    raise TypeError("Scene source names must be strings.")
                self._check_source(child)
        elif isinstance(source, Source):
            if not isinstance(source.spectrum, BaseSpectrum):
                raise TypeError("Source.spectrum must be a Spectrum or Blackbody.")
        else:
            raise TypeError("source must be a Source, Scene or source collection.")

    def _wavefront(self, wavelengths, positions, **context):
        """Propagate the supplied optical wavelengths and source positions."""
        wavefront = self.initialise_wavefront(wavelengths, **context)
        if wavefront.phasor.shape[: wavefront.batch_ndim] != (wavelengths.size,):
            raise ValueError(
                "Imager requires one pupil grid; use vmap for independent grids."
            )
        shape = (positions.shape[0],) + wavefront.phasor.shape
        wavefront = wavefront.set(phasor=np.broadcast_to(wavefront.phasor, shape))
        # Each source position is shared across the trailing wavelength axis.
        offset = positions[:, None, :]
        wavefront = self._propagate(wavefront, offset, unit_input=True, **context)
        expected = (positions.shape[0], wavelengths.size)
        if wavefront.phasor.shape[: wavefront.batch_ndim] != expected:
            raise ValueError(
                "Imager must retain exactly its wavelength and source axes. "
                "Use vmap outside Imager for independent optical models."
            )
        grid = self.image_grid.resolve(**context).oversample(self.oversample)
        if grid.d.ndim != 1 or (grid.c is not None and grid.c.ndim != 1):
            raise ValueError("Imager requires one common output grid for all samples.")
        return wavefront.set(grid=grid)

    def _image(self, intensity, detector=None, *, key=None, n_frames=1, **context):
        """Apply detector operations once, after every source has been summed."""
        if detector is None and key is None:
            return Image(intensity.downsample(self.oversample, mean=False))
        detector = Detector() if detector is None else detector
        return detector.image(
            intensity, oversample=self.oversample, key=key, n_frames=n_frames, **context
        )

    def _blur(self, intensity, **context):
        """Apply image jitter once on the fine grid after incoherent summation."""
        if self.jitter is not None:
            if isinstance(self.jitter, zdx.Expression):
                local = dict(context, grid=intensity.grid, oversample=self.oversample)
                kernel = zdx.resolve(self.jitter, **local)
                if kernel.ndim != 2 or 0 in kernel.shape:
                    raise ValueError(
                        "jitter must resolve to a non-empty two-dimensional kernel."
                    )
            else:
                kernel = self.jitter

                # Keep jitter offsets in native pixels on the oversampled grid.
                ox, oy = self.oversample
                if (ox, oy) != (1, 1):
                    ny, nx = kernel.shape[-2:]
                    shape = kernel.shape[:-2] + ((ny - 1) * oy + 1, (nx - 1) * ox + 1)
                    kernel = (
                        np.zeros(shape, dtype=kernel.dtype)
                        .at[..., ::oy, ::ox]
                        .set(kernel)
                    )
            return intensity.convolve(kernel)
        return intensity


class Coronagraph(Imager):
    """Imager with a compact Soummer mask and a Lyot-plane optic.

    The focal mask is the identity outside ``focal_grid``. Spatially resolved
    source kernels are unsupported; explicit point-source components are allowed.
    All photometric, detector and image methods are inherited from Imager.
    """

    # Inherited fields
    pupil_grid: dlu.Grid
    pupil: OpticalLayer | dlu.Sequential | None = None
    image_grid: dlu.Grid
    source: Source | Scene | list | dict | None = None
    filter: dlu.Filter | dict | None = None
    bandpass: dlu.UnitSpec | Array | zdx.Expression | dict | None = None
    detector: Detector | None = None
    jitter: Array | zdx.Expression | None = None
    oversample: tuple[int, int] = (1, 1)

    focal: OpticalLayer | Sequential
    lyot: OpticalLayer | Sequential | None = None
    focal_grid: dlu.Grid

    def __init__(
        self,
        pupil_grid,
        focal_grid,
        image_grid,
        pupil=None,
        focal=None,
        lyot=None,
        *,
        source=None,
        filter=None,
        bandpass=None,
        detector=None,
        jitter=None,
        oversample=1,
        alias=None,
    ):
        """Construct the pupil, compact focal mask, Lyot and science planes.

        ``pupil`` and ``lyot`` preserve pupil sampling. ``focal``
        preserves its compact focal grid. Both propagators must use explicit MFT
        grids. Remaining inputs have the same meaning as in Imager.
        """
        super().__init__(
            pupil_grid,
            image_grid,
            pupil,
            source=source,
            filter=filter,
            bandpass=bandpass,
            detector=detector,
            jitter=jitter,
            oversample=oversample,
            alias=alias,
        )
        # Soummer's correction is evaluated on one full Lyot-plane grid.
        if self._sparse_pupil is not None:
            raise TypeError(
                "Coronagraph requires a full pupil grid for the Lyot plane."
            )
        self.focal = _plane_slot(focal)
        if focal is None:
            raise TypeError("focal must be an optical layer or Sequential.")
        self.lyot = _plane_slot(lyot)
        self.focal_grid = _normalise_grid(focal_grid, "focal grid")
        SoummerFPMLayer(self.focal, self.focal_propagator)

    @property
    def focal_propagator(self):
        """MFT to the compact mask, using the pupil's effective focal length."""
        return _imaging_propagator(self.focal_grid, self.pupil)

    def propagate_focal(self, wavefront, **context):
        """Return the compact focal field before applying the focal mask."""
        self = self.populate()
        return wavefront.propagate(self.focal_propagator, **context)

    def apply_focal(self, wavefront, **context):
        """Apply focal-plane optics while retaining the compact focal sampling."""
        self = self.populate()
        return _apply_plane(self.focal, wavefront, **context)

    def propagate_lyot(self, pupil, focal, masked, **context):
        """Return to the pupil using the compact focal-field difference."""
        self = self.populate()
        mask = SoummerFPMLayer(self.focal, self.focal_propagator)
        return mask.return_to_pupil(pupil, focal, masked, **context)

    def apply_lyot(self, wavefront, **context):
        """Apply Lyot-plane optics to the returned pupil field."""
        self = self.populate()
        return _apply_plane(self.lyot, wavefront, **context)

    def _propagate(self, wavefront, offset=None, *, unit_input=False, **context):
        """Compose pupil, focal, Lyot and image stages with accessible branch points."""
        self = self.populate()
        pupil = self.apply_pupil(wavefront, offset, unit_input=unit_input, **context)
        focal = self.propagate_focal(pupil, **context)
        masked = self.apply_focal(focal, **context)
        lyot = self.propagate_lyot(pupil, focal, masked, **context)
        lyot = self.apply_lyot(lyot, **context)
        return self.propagate_image(lyot, **context)

    def _check_source(self, source):
        """Resolved-source convolution assumes an isoplanatic optical response."""
        super()._check_source(source)
        if isinstance(source, ResolvedSource):
            raise ValueError(
                "Coronagraph supports point sources, not resolved source kernels."
            )


class Interferometer(Imager):
    """An Imager with a sparse pupil and optional optics at the focal plane.

    ``grid_in`` samples one local aperture patch. The first pupil optic places
    copies at its aperture centres, where source tilt includes the relative piston.
    The MFT combines these fields coherently; inherited Imager methods then handle
    source photometry, resolved profiles and detector operations.
    """

    # Inherited fields
    pupil_grid: dlu.Grid
    pupil: OpticalLayer | dlu.Sequential | None = None
    image_grid: dlu.Grid
    source: Source | Scene | list | dict | None = None
    filter: dlu.Filter | dict | None = None
    bandpass: dlu.UnitSpec | Array | zdx.Expression | dict | None = None
    detector: Detector | None = None
    jitter: Array | zdx.Expression | None = None
    oversample: tuple[int, int] = (1, 1)

    focal_plane: OpticalLayer | Sequential | None = None
    relay: dlu.Propagator | dlu.SpectralPropagator | None = None

    def __init__(
        self,
        grid_in,
        grid_out,
        aperture,
        focal_plane=None,
        *,
        propagator=None,
        source=None,
        filter=None,
        bandpass=None,
        detector=None,
        jitter=None,
        oversample=1,
        alias=None,
    ):
        """Store sparse optics with the same modelling inputs as Imager.

        ``propagator`` optionally supplies a custom MFT relay; its output sampling
        follows grid_out and the Imager's oversampling factor. None derives the
        usual imaging MFT from the output grid and pupil focal length.
        """
        self.relay = (
            None if propagator is None else _focal_propagator(grid_out, propagator)
        )
        super().__init__(
            grid_in,
            grid_out,
            aperture,
            source=source,
            filter=filter,
            bandpass=bandpass,
            detector=detector,
            jitter=jitter,
            oversample=oversample,
            alias=alias,
        )
        if self._sparse_pupil is None:
            raise TypeError("aperture must begin with a SparseOptic.")
        self.focal_plane = _plane_slot(focal_plane)

    @property
    def aperture(self):
        """The sparse pupil, stored using Imager's common pupil leaf."""
        return self.pupil

    @property
    def grid_out(self):
        """Native output pixels, before optical oversampling."""
        return self.image_grid

    @property
    def propagator(self):
        """Return the MFT relay on the current fine image grid."""
        if self.relay is None:
            return super().propagator
        return self.relay.set(grid_out=self.propagation_grid)

    def propagate_image(self, wavefront, **context):
        """Coherently combine sparse fields, then apply the focal-plane optics."""
        wavefront = super().propagate_image(wavefront, **context)
        return _apply_plane(self.focal_plane, wavefront, **context)
