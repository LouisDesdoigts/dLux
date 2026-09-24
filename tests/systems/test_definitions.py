"""Systems consume constructor factories before modelling starts."""

import equinox as eqx
import jax.numpy as np

import dLux as dl
import dLux.utils as dlu


class DetectorDef(dl.Def):
    def __call__(self):
        return dl.Detector(gain=2.0)


def test_imager_consumes_aperture_and_detector_definitions():
    grid = dlu.Grid(n=8, diam=1.0, unit="m")
    image_grid = dlu.Grid(n=8, d=0.1, unit="arcsec")
    pupil = dl.ApertureBuilder(
        dlu.Circle(0.8), opd=dl.ZernikeDef(nolls=[1, 4], unit="nm")
    )
    model = dl.Imager(
        grid,
        image_grid,
        pupil=pupil,
        detector=DetectorDef(),
        source=dl.Source(),
        bandpass=dlu.BandPass(600e-9, n=1),
    )
    assert isinstance(model.pupil, dl.Optic)
    assert isinstance(model.pupil.opd, dlu.Basis)
    assert isinstance(model.detector, dl.Detector)
    assert model.pupil.opd.unit == "nm"
    assert np.array_equal(model.pupil.opd.coeffs, np.zeros(2))
    expected = dl.Imager(
        grid,
        image_grid,
        pupil=pupil(grid),
        detector=dl.Detector(gain=2.0),
        source=dl.Source(),
        bandpass=dlu.BandPass(600e-9, n=1),
    )
    image = eqx.filter_jit(lambda value: value.model())(model)
    assert np.allclose(image.data, expected.model().data, rtol=2e-6, atol=1e-8)
