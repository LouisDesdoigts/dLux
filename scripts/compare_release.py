"""Compare numerical observations in isolated reference/compat processes.

Usage: python scripts/compare_release.py ../reference-release --output /tmp/comparison
The reference checkout is read only. No installed package or import namespace is
changed in the parent process. Each worker records import locations and public API.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def observations(dl, migrated=False):
    """Asymmetric fields and nonzero parameters expose convention differences."""
    import jax
    import jax.numpy as np
    import equinox as eqx

    dlu = dl.utils
    outputs = {}
    coords = dlu.pixel_coords(16, 1.0)
    x, y = coords
    field = (1 + x + 0.2 * y) * np.exp(1j * (x - 2 * y))
    outputs["coordinates"] = coords
    outputs["coordinate_shear"] = dlu.shear_coords(coords, np.array([0.1, -0.2]))
    outputs["coordinate_rotation"] = dlu.rotate_coords(coords, 0.3)
    outputs["circle_radius"] = dlu.circle(coords, 0.31)
    outputs["polygon_radius"] = dlu.reg_polygon(coords, 0.42, 6)
    outputs["soft_circle"] = dlu.soft_circle(coords, 0.31, 0.04)
    outputs["zernikes"] = dlu.zernike_basis([1, 4, 8, 15, 23], coords, 0.8)
    outputs["polikes"] = dlu.polike_basis(6, [1, 4, 8], coords, 0.8)
    outputs["interpolation_rotation"] = dlu.rotate(field.real, 0.2)
    outputs["interpolation_scale"] = dlu.scale(field.real, 12, 1.3)

    # Asymmetric inputs catch sign/axis changes that round or centred pupils hide.
    wavefront = dl.Wavefront.from_phasor(field, 600e-9, pixel_scale=1 / 16)
    psf = dl.PSF(np.exp(-((x - 0.1) ** 2 + (y + 0.2) ** 2) / 0.1), 1 / 16)
    for sigma, size, oversample in ((1.3, 9, 3), (0.5, 5, 2), (2.0, 7, 1)):
        operation = dl.ApplyJitter(sigma, size, oversample)
        outputs[f"jitter_{sigma}_{size}_{oversample}"] = operation(psf).data
    for method in ("linear", "cubic"):
        outputs[f"rotate_psf_{method}"] = dl.Rotate(0.2, method)(psf).data
        for complex in (False, True):
            outputs[f"rotate_field_{method}_{complex}"] = dl.Rotate(
                0.2, method, complex
            )(wavefront).phasor
    coefficients = np.arange(9.0).reshape(3, 3) * 1e-9
    outputs["fourier_basis_layer"] = dl.FourierBasis(16, 3, coefficients)(
        wavefront
    ).phasor
    for size in ((7, 9), (7, 9, 5)):
        ndim = len(size)
        outputs[f"gaussian_{ndim}d"] = dlu.gaussian(
            mean=np.array([0.2, -0.4, 0.6])[:ndim],
            std=np.array([0.7, 1.2, 0.9])[:ndim],
            npixels=size,
        )

    # Compose transmission-only apertures, retaining every local coordinate frame.
    ring = dl.CompoundAperture(
        [dl.CircularAperture(0.25), dl.CircularAperture(0.12, occulting=True)],
        transformation=dl.CoordTransform([0.07, -0.1], 0.3, [0.7, 1.1]),
    )
    square = dl.SquareAperture(0.2, transformation=dl.CoordTransform([-0.2, 0.2], -0.2))
    combined = dl.MultiAperture(
        [("ring", ring), ("square", square)],
        transformation=dl.CoordTransform([0.03, 0.02], -0.1),
    )
    outputs["nested_apertures"] = combined(wavefront).phasor
    outputs["nested_apertures_jit"] = eqx.filter_jit(lambda layer: layer(wavefront))(
        combined
    ).phasor

    outputs["fft"], outputs["fft_spacing"] = dlu.FFT(field, 600e-9, 1 / 16, pad=2)
    outputs["inverse_fft"] = dlu.FFT(field, 600e-9, 1 / 16, pad=1, inverse=True)[0]

    # Layer centring includes phases that the raw array FFT does not apply.
    for n in (9, 16):
        xy = dlu.pixel_coords(n, 1.0)
        phasor = (1 + xy[0]) * np.exp(0.3j * xy[1])
        wavefront = dl.Wavefront.from_phasor(phasor, 600e-9, pixel_scale=1 / n)
        for center in (False, True):
            for inverse in (False, True):
                for focal_length in (None, 2.0):
                    pad = 1 if n % 2 else 2
                    operation = dl.FFT(focal_length, inverse, pad, pad, center)
                    field_in = wavefront
                    if migrated and inverse and focal_length is None:
                        field_in = wavefront.set(grid=wavefront.grid.set(unit="rad"))
                    key = f"fft_layer_{n}_{center}_{inverse}_{focal_length}"
                    outputs[key] = operation(field_in).phasor
    outputs["mft"] = dlu.MFT(
        field, 600e-9, 1 / 16, 12, 2e-7, shift=np.array([0.3, -0.7])
    )
    outputs["contrast"] = dlu.fluxes_from_contrast(3.0, 0.2)
    outputs["binary_positions"] = dlu.positions_from_sep(
        np.array([0.1, -0.2]), 0.3, 0.4
    )
    outputs["pupil_factory"] = dlu.circular_aperture(16, 1.0, oversample=1)

    # Released optical/source/detector interaction, including spectral weights.
    pupil = dl.Optic(
        transmission=dlu.circle(coords, 0.45), opd=2e-8 * x, normalise=True
    )
    optics = dl.AngularOpticalSystem(16, 1.0, [pupil], 12, 0.03, 1)
    source = dl.PointSource(
        [500e-9, 700e-9], weights=[0.2, 0.8], position=[1e-7, -2e-7], flux=5.0
    )
    detector = dl.LayeredDetector(
        [
            dl.ApplyPixelResponse(np.ones((12, 12)) * 0.8),
            dl.AddConstant(0.1),
            dl.Downsample(2),
        ]
    )
    telescope = dl.Telescope(optics, source, detector)
    outputs["broadband_psf"] = optics.model(source)
    image = lambda model: model.model().data if migrated else model.model()
    outputs["detector_image"] = image(telescope)
    outputs["detector_jit"] = eqx.filter_jit(image)(telescope)
    outputs["wavelength_fields"] = optics.model(source, return_wf=True).phasor
    path = "pupil.layers.Optic.opd" if migrated else "layers.Optic.opd"
    response = lambda scale: optics.set(path, scale * x).model(source)[4, 5]
    outputs["opd_gradient"] = jax.jit(jax.grad(response))(2e-8)
    flux_path = "source.spectrum.flux" if migrated else "source.flux"
    flux_response = lambda flux: image(telescope.set(flux_path, flux)).sum()
    outputs["flux_gradient"] = jax.jit(jax.grad(flux_response))(5.0)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/dlux-release-comparison")
    )
    parser.add_argument("--worker", choices=("reference", "compat"))
    args = parser.parse_args()
    reference = args.reference.resolve()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)

    if args.worker:
        sys.path.insert(
            0, str((reference if args.worker == "reference" else repo) / "src")
        )
        import dLux
        import jax.numpy as np

        dl = dLux if args.worker == "reference" else dLux.compat
        result = {"import": dLux.__file__, "arrays": {}, "api": {}}
        for name, array in observations(dl, migrated=args.worker == "compat").items():
            array = np.asarray(array)
            result["arrays"][name] = {
                "shape": list(array.shape),
                "real": array.real.ravel().tolist(),
                "imag": array.imag.ravel().tolist(),
            }
        for name, module in [("core", dl), ("layers", dl.layers), ("utils", dl.utils)]:
            result["api"][name] = sorted(module.__all__)
        (args.output / f"{args.worker}.json").write_text(json.dumps(result, indent=2))
        return

    for backend in ("reference", "compat"):
        subprocess.run(
            [
                sys.executable,
                __file__,
                str(reference),
                "--output",
                str(args.output),
                "--worker",
                backend,
            ],
            check=True,
            env=os.environ,
        )

    import jax.numpy as np

    baseline = json.loads((args.output / "reference.json").read_text())
    current = json.loads((args.output / "compat.json").read_text())
    differences = {}
    for name, expected in baseline["arrays"].items():
        actual = current["arrays"][name]
        assert actual["shape"] == expected["shape"], name
        a = np.array(actual["real"]) + 1j * np.array(actual["imag"])
        b = np.array(expected["real"]) + 1j * np.array(expected["imag"])
        differences[name] = float(np.max(np.abs(a - b)))
        # Affine matrices reorder the released nested coordinate arithmetic.
        # Soft edges amplify float32 rounding: 1.1e-6 here, 2.2e-15 in float64.
        atol = 2e-6 if name.startswith("nested_apertures") else 1e-7
        assert np.allclose(a, b, rtol=5e-6, atol=atol), (name, differences[name])
    for module, names in baseline["api"].items():
        missing = set(names) - set(current["api"][module])
        assert not missing, (module, sorted(missing))
    (args.output / "differences.json").write_text(json.dumps(differences, indent=2))
    print(
        f"{len(differences)} numerical comparisons passed; "
        "released public names covered."
    )


if __name__ == "__main__":
    main()
