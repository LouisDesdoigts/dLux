"""Standalone sequences of optical layers and numerical propagators."""

from __future__ import annotations

from collections import OrderedDict

import jax.numpy as np
import zodiax as zdx

import dLux.utils as dlu

from ..fields import Wavefront, PSF
from .sources import Source, PointSource, ResolvedSource
from .modelling import (
    render_sources,
    source_samples,
    source_profile,
    spectral_weights,
    integrate_psf,
)
from ..layers.optical import OpticalLayer
from ..utils.sequential import Sequential

__all__ = ["OpticalSystem"]


class OpticalSystem(Sequential):
    """Named optical layers with optional entrance sampling.

    ``OpticalSystem(layers, grid)`` supports direct wavelength propagation and
    source modelling. Both return arrays by default; ``return_psf=True`` retains
    sampling and ``return_wf=True`` returns the complex field. Calling the system
    on an existing Wavefront still applies its layers in order.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    # A named layer sequence retains the established propagation interface.
    grid = dl.Grid(8, diam=1.0, unit="m")
    system = dl.OpticalSystem({
        "pupil": dl.Optic(transmission=dl.Circle(1.0), normalise=True),
        "focus": dl.Propagator(dl.Grid(8, d=0.1, unit="arcsec")),
    }, grid)
    mono = system.propagate_mono(600e-9)
    psf = system.propagate(np.array([550e-9, 650e-9]), return_psf=True)
    image = system.model(dl.PointSource(flux=1000, spectrum=dl.Spectrum(
        bandpass=dl.BandPass(range=(550, 650), n=2, unit="nm"))))
    print(psf)
    ```
    """

    # Inherited fields
    layers: OrderedDict

    grid: dlu.Grid | None = None

    def __init__(self, layers=(), grid=None, *, alias=None):
        """Store layers and entrance grid; grid is optional for Wavefront input.

        Parameters
        ----------
        layers : mapping or sequence
            Optical layers, numerical propagators or nested named sequences in
            propagation order.
        grid : Grid or None
            Entrance sampling for propagate/model. May be omitted when applying the
            system to an existing Wavefront.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        super().__init__(layers, (OpticalLayer, dlu.BasePropagator), alias=alias)
        if grid is not None and not isinstance(grid, dlu.Grid):
            raise TypeError("grid must be a Grid.")
        self.grid = grid

    @staticmethod
    def _apply_step(step, wavefront, **context):
        """Adapt numerical propagation, including within nested sequences."""
        if isinstance(step, Sequential):
            for child in step.populate().steps.values():
                wavefront = OpticalSystem._apply_step(child, wavefront, **context)
            return wavefront
        if isinstance(step, OpticalLayer):
            return step(wavefront, **context)
        return wavefront.propagate(step, **context)

    def apply(self, wavefront, **context):
        """Apply the ordered layers to an existing Wavefront."""
        if not isinstance(wavefront, Wavefront):
            raise TypeError("OpticalSystem requires a Wavefront.")
        for step in self.populate().steps.values():
            wavefront = self._apply_step(step, wavefront, **context)
        return wavefront

    def debug(self, wavefront, **context):
        """Return named intermediate fields with the same numerical adaptation."""
        if not isinstance(wavefront, Wavefront):
            raise TypeError("OpticalSystem requires a Wavefront.")
        outputs = {"input": wavefront}
        for name, step in self.populate().steps.items():
            wavefront = self._apply_step(step, wavefront, **context)
            outputs[name] = wavefront
        return wavefront, outputs

    def insert_layer(self, layer, index, layer_type=(OpticalLayer, dlu.BasePropagator)):
        """Insert a compatible optical layer at the requested position."""
        return super().insert_layer(layer, index, layer_type)

    @property
    def propagator(self):
        """Return the last relay, used for angular profiles on a length-valued image."""

        def relays(sequence):
            for step in sequence.steps.values():
                if isinstance(step, Sequential):
                    yield from relays(step)
                elif (
                    isinstance(step, dlu.BasePropagator)
                    and getattr(step, "grid_out", None) is not None
                ):
                    yield step

        return next(reversed(list(relays(self))), None)

    def initialise_wavefront(self, wavelength, offset=None, **context):
        """Create unit incident power on the stored entrance grid."""
        if self.grid is None:
            raise ValueError(
                "Supply OpticalSystem(grid=...) to initialise a Wavefront."
            )
        grid = self.grid.resolve(wavelength=wavelength, **context)
        field = Wavefront(wavelength, grid)
        return field if offset is None else field.tilt(np.asarray(offset))

    def propagate_mono(
        self, wavelength, offset=None, return_wf=False, *, return_psf=False, **context
    ):
        """Propagate one wavelength in metres; return intensity samples by default."""
        if np.ndim(wavelength) != 0:
            raise ValueError("wavelength must be scalar; use propagate for a vector.")
        if return_wf and return_psf:
            raise ValueError("Select return_wf or return_psf, not both.")
        system = self.populate()
        field = system.initialise_wavefront(wavelength, offset, **context)
        field = system.apply(field, **context)
        if return_wf:
            return field
        psf = field.to_psf()
        return psf if return_psf else psf.data

    def propagate(
        self,
        wavelengths,
        offset=None,
        weights=None,
        return_wf=False,
        return_psf=False,
        *,
        return_all=False,
        **context,
    ):
        """Incoherently sum wavelength responses with optional integrated weights.

        Default weights sum to one. Supplied weights retain their total. A returned
        Wavefront has amplitude scaled by sqrt(weights), as in the released API.
        Wavelength samples must share one output grid to form a summed PSF.
        """
        if sum((return_wf, return_psf, return_all)) > 1:
            raise ValueError("Select one of return_wf, return_psf or return_all.")
        wavelengths = np.atleast_1d(np.asarray(wavelengths))
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty vector.")
        weights = (
            np.ones_like(wavelengths) / wavelengths.size
            if weights is None
            else np.asarray(weights)
        )
        if weights.shape != wavelengths.shape or np.iscomplexobj(weights):
            raise ValueError("weights must be real and match wavelengths.shape.")

        system = self.populate()
        field = system.initialise_wavefront(wavelengths, offset, **context)
        field = system.apply(field, **context)
        # Integrated intensity is linear in the weights, including at zero weight.
        # Only an explicitly requested complex field needs amplitude scaling.
        if return_wf:
            return field * np.sqrt(weights)[..., None, None]
        field = self._common_grid(field)
        data = (field.intensity * weights[..., None, None]).sum(axis=-3)
        psf = PSF(data, field.grid)
        if return_all:
            field = field * np.sqrt(weights)[..., None, None]
            return {"Wavefront": field, "PSF": psf}
        return psf if return_psf else psf.data

    @staticmethod
    def _common_grid(field):
        """Remove repeated sampling metadata before adding incoherent intensities.

        Monochromatic layers map over fields and repeat their grid metadata. The
        samples must still describe one physical grid before spectral summation.
        """
        grid = field.grid
        d = grid.d.reshape(-1, 2)[0]
        c = None if grid.c is None else grid.c.reshape(-1, 2)[0]
        common = grid.set(d=d, c=c)
        message = "Spectral integration requires one common output grid."
        field.grid.check_sampling(common, message)
        return field.set(grid=common)

    def _wavefront(self, wavelengths, positions, **context):
        """Supply source-by-wavelength fields to the shared modelling calculation."""
        field = self.initialise_wavefront(wavelengths, **context)
        field = field.set(
            phasor=np.broadcast_to(field.phasor, (len(positions),) + field.phasor.shape)
        )
        field = self.apply(field.tilt(positions[:, None, :]), **context)
        if field.batch_shape != (len(positions), len(wavelengths)):
            raise ValueError(
                "Source modelling requires source and wavelength axes; "
                "combine aperture fields in the optical sequence."
            )
        field = self._common_grid(field)
        return field

    def _psf(self, wavefront, positions, stokes=None):
        """Convert optical fields using the shared unweighted PSF convention."""
        return wavefront.to_psf(stokes).set(positions=positions)

    def _photometry(self, source, bandpass, filter=None, qe=None, **context):
        """Use bin integration, retaining released point-weight input separately."""
        sampling = source.spectrum.bandpass if bandpass is None else bandpass
        if isinstance(sampling, dlu.BandPass):
            return spectral_weights(source, sampling, filter, qe, **context)
        from ..compat.modelling import spectral_weights as released_weights

        return released_weights(source, bandpass, filter, qe, **context)

    def _intensity(self, psf, weights, *, profile=None, **context):
        """Convolve source morphology, then sum the weighted point responses."""
        if profile is not None:
            if profile.ndim == 3 and profile.shape[0] != psf.data.shape[-3]:
                raise ValueError(
                    "A chromatic profile must match the PSF wavelength axis."
                )
            psf = psf.convolve(profile)
        return integrate_psf(psf, weights)

    def _render_source(
        self, stage, source, offset=None, stokes=None, *, bandpass=None, **context
    ):
        """Model one source, preserving the established OpticalSystem outputs."""
        wavelengths, positions = source_samples(source, bandpass, offset, **context)
        weights = self._photometry(source, bandpass, **context)
        wavefront = self._wavefront(
            wavelengths=wavelengths, positions=positions, **context
        )
        if stage == "weighted_wavefront":
            return wavefront * np.sqrt(weights.value)[..., None, None]

        psf = self._psf(wavefront=wavefront, positions=positions, stokes=stokes)
        profile = source_profile(
            source, wavefront.grid, self.propagator, wavelengths=wavelengths, **context
        )
        intensity = self._intensity(
            psf=psf, weights=weights, profile=profile, **context
        )
        if stage == "all":
            return {
                "Wavefront": wavefront,
                "PSF": psf,
                "weights": weights,
                "Intensity": intensity,
            }
        return intensity

    def model(
        self,
        source,
        return_wf=False,
        return_psf=False,
        *,
        bandpass=None,
        return_all=False,
        **context,
    ):
        """Render a Source or Scene using the same source calculations as Imager.

        Returns an array by default, or a PSF with ``return_psf=True``. Spectral
        sampling comes from ``bandpass`` or the source spectrum. No detector is
        applied. ``return_all=True`` also retains the unweighted complex fields.
        """
        if sum((return_wf, return_psf, return_all)) > 1:
            raise ValueError("Select one of return_wf, return_psf or return_all.")
        # Populate the shared owner before splitting source and optical parameters.
        system, source, bandpass = zdx.populate((self, source, bandpass))
        if return_wf and (
            not isinstance(source, Source) or isinstance(source, ResolvedSource)
        ):
            raise TypeError(
                "return_wf requires a point source; render Scene components separately."
            )
        stage = "intensity"
        if return_wf:
            stage = "weighted_wavefront"
        elif return_all:
            stage = "all"
        output = render_sources(system, stage, source, bandpass=bandpass, **context)
        if return_wf:
            if isinstance(source, PointSource):
                # Remove the singleton source axis, including compact Jones data.
                jones = output.jones
                if jones is not None and jones is not True and jones.ndim > 2:
                    jones = output._uniform_jones()[0]
                output = output.set(phasor=output.phasor[0], jones=jones)
            return output

        intensity = output["Intensity"] if return_all else output
        psf = PSF(intensity.data, intensity.grid)
        if return_all:
            return {"Wavefront": output["Wavefront"], "PSF": psf}
        return psf if return_psf else psf.data

    def debug_propagate_mono(self, wavelength, offset=None, **context):
        """Return the final wavefront and each named intermediate wavefront."""
        system = self.populate()
        return system.debug(
            system.initialise_wavefront(wavelength, offset, **context), **context
        )
