"""Recursive optical paths sharing one source-weighted entrance field."""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as np
import zodiax as zdx

import dLux.utils as dlu

from ..fields import Intensity, Wavefront
from .detector import DetectorSystem
from .optical import OpticalSystem
from .sources import ResolvedSource, Source

__all__ = ["FrameLag", "StateTransition", "OpticalBranch", "OpticalTree"]


def _named_branches(branches):
    """Store branch names in propagation order with structural path-safe keys."""
    if not isinstance(branches, Mapping) or not branches:
        raise ValueError("branches must be a non-empty named mapping.")
    named = OrderedDict(branches)
    for name, branch in named.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("branch names must be non-empty strings without dots.")
        if not isinstance(branch, OpticalBranch):
            raise TypeError("branches must contain OpticalBranch values.")
    return named


def _select_wavelengths(field: Wavefront, indices: tuple[int, ...]) -> Wavefront:
    """Select the final wavelength batch axis, retaining source and Jones axes."""
    if indices is None:
        raise ValueError("OpticalTree branch routes must be prepared before modelling.")
    field = OpticalSystem._common_grid(field)
    axis = field.batch_ndim - 1
    jones = field.jones
    if jones is not None and jones is not True:
        jones = np.take(field._uniform_jones(), np.asarray(indices), axis=axis)
    return field.set(
        phasor=np.take(field.phasor, np.asarray(indices), axis=axis),
        wavelength=np.take(field.wavelength, np.asarray(indices)),
        jones=jones,
    )


class FrameLag(zdx.Module):
    """Delay an array-valued command by a fixed number of model frames.

    The caller keeps the history in its State and calls ``step`` once before
    each optical evaluation. Commands and history have the same dtype and units.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    lag = dl.FrameLag(1)
    command = np.ones((4, 1))
    history = lag.initial_history(command)
    applied, history = lag.step(history, command)  # applied is zero
    applied, history = lag.step(history, command)  # applied is command
    ```
    """

    frames: int = eqx.field(static=True)

    def __init__(self, frames: int):
        """Store the non-negative number of frames between command and effect.

        Parameters
        ----------
        frames : int
            Zero applies the current command in the current frame.
        """
        if isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 0:
            raise ValueError("frames must be a non-negative integer.")
        self.frames = int(frames)

    def initial_history(self, command):
        """Return a zero-filled history with shape ``(frames, *command.shape)``."""
        command = np.asarray(command)
        return np.zeros((self.frames, *command.shape), dtype=command.dtype)

    def step(self, history, command):
        """Return the applied command and updated history for one model frame."""
        command = np.asarray(command)
        history = np.asarray(history)
        if history.shape != (self.frames, *command.shape):
            raise ValueError("history must have shape (frames, *command.shape).")
        if history.dtype != command.dtype:
            raise ValueError("history and command must have the same dtype.")
        if self.frames == 0:
            return command, history
        applied = history[0]
        updated = np.concatenate((history[1:], command[None]), axis=0)
        return applied, updated


class StateTransition(zdx.Module):
    """Update a simulator State before one optical evaluation.

    Subclasses store their parameters as Zodiax fields and implement ``apply``.
    They return the updated State without advancing its time or index; OpticalTree
    performs that advance after computing the rates for the current tick.
    """

    @abstractmethod
    def apply(self, state: zdx.State, inputs, dt, key) -> zdx.State:
        """Return a State updated from the current commands and random key."""

    def __call__(self, state: zdx.State, inputs, dt, key) -> zdx.State:
        return self.apply(state, inputs, dt, key)


class OpticalBranch(zdx.Module):
    """One transmitted optical path ending at a detector or another split.

    ``transmission`` is a wavelength-dependent power fraction, evaluated at the
    source's optical wavelength centres. A parent branch applies its optics once
    before supplying the resulting immutable field to every child.

    Examples
    --------
    ```python
    import dLux as dl

    band = dl.BandPass(range=(1500, 1800), n=3, unit="nm")
    branch = dl.OpticalBranch(
        dl.OpticalSystem([dl.Propagator(dl.Grid(8, d=0.1, unit="arcsec"))]),
        transmission=dl.Filter(0.5, band),
    )
    ```
    """

    optics: OpticalSystem
    transmission: dlu.Filter
    children: OrderedDict[str, OpticalBranch]
    detector: DetectorSystem | None = None
    indices: tuple[int, ...] | None = None

    def __init__(
        self,
        optics: OpticalSystem,
        transmission=1.0,
        *,
        children=None,
        detector=None,
        alias=None,
    ):
        """Store one path and optional downstream branches.

        Parameters
        ----------
        optics : OpticalSystem
            Optical sequence applied after this branch's transmission.
        transmission : Filter or numerical value
            Non-negative power fraction as a function of wavelength. A numerical
            value is wrapped in Filter. Sibling fractions define the split.
        children : mapping or None
            Named OpticalBranch children. None makes this a terminal branch.
        detector : DetectorSystem or None
            Optional deterministic response at a terminal branch, after spectral
            integration. None leaves the rate unchanged.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases.
        """
        if not isinstance(optics, OpticalSystem):
            raise TypeError("branch optics must be an OpticalSystem.")
        if detector is not None and not isinstance(detector, DetectorSystem):
            raise TypeError("branch detector must be a DetectorSystem or None.")
        if children is not None and detector is not None:
            raise ValueError("a branch with children cannot also have a detector.")
        self.optics = optics
        self.transmission = (
            transmission
            if isinstance(transmission, dlu.Filter)
            else dlu.Filter(transmission)
        )
        self.children = OrderedDict() if children is None else _named_branches(children)
        self.detector = detector
        self.indices = None
        self.alias = alias

    def _prepare(self, populated, wavelengths):
        """Cache selection from the parent wavelength vector in a copied branch."""
        response = populated.transmission.sample(wavelengths)
        if response.shape != wavelengths.shape:
            raise ValueError("branch transmission must have one value per wavelength.")
        if not bool(np.all(np.isfinite(response) & (response >= 0))):
            raise ValueError("branch transmission must be finite and non-negative.")
        indices = tuple(
            index for index, value in enumerate(jax.device_get(response)) if value != 0
        )
        if not indices:
            raise ValueError("a prepared branch must transmit at least one wavelength.")
        selected = np.take(wavelengths, np.asarray(indices))
        children = OrderedDict(
            (
                name,
                child._prepare(populated.children[name], selected),
            )
            for name, child in self.children.items()
        )
        return self.set(indices=indices, children=children)

    def _model(self, field, path, state):
        """Evaluate this branch and return terminal rates keyed by path."""
        field = _select_wavelengths(field, self.indices)
        response = self.transmission.sample(field.wavelength, state=state)
        if response.shape != field.wavelength.shape:
            raise ValueError("branch transmission must have one value per wavelength.")
        trailing = field.phasor.ndim - field.batch_ndim
        scale_shape = (1,) * (field.batch_ndim - 1) + response.shape + (1,) * trailing
        field = field.set(phasor=field.phasor * np.sqrt(response).reshape(scale_shape))
        output = self.optics.apply(field, state=state)
        if not isinstance(output, Wavefront) or output.batch_shape != field.batch_shape:
            raise ValueError("branch optics must preserve source and wavelength axes.")

        if self.children:
            rates = {}
            for name, child in self.children.items():
                rates.update(child._model(output, f"{path}.{name}", state))
            return rates

        output = OpticalSystem._common_grid(output)
        intensity = output.intensity
        axes = tuple(range(intensity.ndim - 2))
        rate = Intensity(intensity.sum(axis=axes), output.grid)
        if self.detector is not None:
            rate = self.detector.apply(rate, state=state)
        return {path: rate}


class OpticalTree(zdx.Module):
    """A source-weighted optical system with recursive detector branches.

    Source flux is applied once in the shared optical system. Every branch then
    scales the complex field by the square root of its power transmission, runs its
    own OpticalSystem, and either splits again or returns an Intensity rate.
    ``prepare()`` is required before model or step; it returns a new tree and leaves
    the original, unprepared object unchanged.

    Examples
    --------
    ```python
    import dLux as dl

    pupil = dl.Grid(8, diam=1, unit="m")
    image = dl.Grid(8, d=0.1, unit="arcsec")
    wfs_focal = dl.Grid(16, d=0.05, unit="arcsec")
    band = dl.BandPass(range=(500, 700), n=2, unit="nm")
    source = dl.PointSource(spectrum=dl.Spectrum(1, band, flux=1000))
    # Quarter-wave phase dot at 600 nm, one lambda/D in diameter.
    phase_dot = dl.Optic(opd=150e-9 * dl.Circle(600e-9)(wfs_focal))
    wfs = dl.SoummerFPMLayer(phase_dot, dl.Propagator(wfs_focal))
    tree = dl.OpticalTree(
        source,
        dl.OpticalSystem([dl.Optic(transmission=dl.Circle(1))], pupil),
        {
            "science": dl.OpticalBranch(
                dl.OpticalSystem([dl.Propagator(image)]), transmission=0.5
            ),
            "wfs": dl.OpticalBranch(dl.OpticalSystem([wfs]), transmission=0.5),
        },
    ).prepare()
    rates = tree.model()  # rates["science"] and rates["wfs"]
    ```
    """

    source: Source
    optics: OpticalSystem
    branches: OrderedDict[str, OpticalBranch]
    transition: StateTransition | None = None
    prepared: bool = False

    def __init__(
        self,
        source: Source,
        optics: OpticalSystem,
        branches,
        transition: StateTransition | None = None,
        *,
        alias=None,
    ):
        """Store the source, common optics, named branches and optional dynamics.

        Parameters
        ----------
        source : Source or PointSource
            Source with a fixed BandPass and flux interpreted by the caller as a
            rate per second for streaming use. Scene and ResolvedSource are not
            supported by this first tree interface.
        optics : OpticalSystem
            Common entrance optics with an input grid.
        branches : mapping
            Non-empty named OpticalBranch roots.
        transition : StateTransition or None
            Optional state update applied before optical evaluation in step.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases.
        """
        if not isinstance(source, Source) or isinstance(source, ResolvedSource):
            raise TypeError("source must be a Source or PointSource.")
        if not isinstance(optics, OpticalSystem):
            raise TypeError("optics must be an OpticalSystem.")
        if transition is not None and not isinstance(transition, StateTransition):
            raise TypeError("transition must be a StateTransition or None.")
        self.source = source
        self.optics = optics
        self.branches = _named_branches(branches)
        self.transition = transition
        self.prepared = False
        self.alias = alias

    def prepare(self) -> OpticalTree:
        """Return a copy with fixed wavelength routes for every branch."""
        bandpass = self.source.spectrum.bandpass
        if not isinstance(bandpass, dlu.BandPass):
            raise TypeError("OpticalTree requires a Source with a fixed BandPass.")
        populated = self.populate()
        wavelengths = bandpass.wavelengths
        branches = OrderedDict(
            (
                name,
                branch._prepare(populated.branches[name], wavelengths),
            )
            for name, branch in self.branches.items()
        )
        return self.set(branches=branches, prepared=True)

    def model(self, state: zdx.State | None = None) -> dict[str, Intensity]:
        """Return expected rates from the current state at every detector leaf."""
        if not self.prepared:
            raise ValueError("OpticalTree must be prepared before model or step.")
        tree = self.populate()
        field = tree.optics.model(tree.source, return_wf=True, state=state)
        rates = {}
        for name, branch in tree.branches.items():
            rates.update(branch._model(field, name, state))
        return rates

    def step(self, state: zdx.State, inputs, dt, key):
        """Update state, evaluate current rates, then advance time and index."""
        if not self.prepared:
            raise ValueError("OpticalTree must be prepared before model or step.")
        if not isinstance(state, zdx.State):
            raise TypeError("step requires a zodiax.State.")
        tree = self.populate()
        updated = (
            state
            if tree.transition is None
            else tree.transition(state, inputs, dt, key)
        )
        if not isinstance(updated, zdx.State):
            raise TypeError("transition must return a zodiax.State.")
        rates = tree.model(updated)
        return updated.step(dt), rates
