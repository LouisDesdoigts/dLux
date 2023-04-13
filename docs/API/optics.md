# Optics

The `optics.py` script contains the general `OpticalLayer` classes. The main class is `OpticalLayer` which is the base class for all other optical layers, including those in `apertures.py`, `aberrations.py` and `propagators.py`. Unless you are creating a new optical layer, you will not need to use this class directly.

---

## Create Wavefront

This layer should be the first layer of almost all set of optical layers. It is used to initialise the wavefront object by specifying the number of pixels and the diameter of the wavefront.

??? info "CreateWavefront API"
    :::dLux.optics.CreateWavefront

---

## Normalise Wavefront

This layer is used to normalise the wavefront to a unit intensity. This simply controls at what point the wavefront is normalised to unity power. Since most calculations in optics use the collecting area of the aperture, most use cases will require this layer at some point after an aperture layer.

??? info "NormaliseWavefront API"
    :::dLux.optics.NormaliseWavefront

---

## Tilt Wavefront

This class is used to tilt the wavefront by a specified angle in radians.

??? info "TiltWavefront API"
    :::dLux.optics.TiltWavefront

---

## Transmissive Optic

This layer takes in an array of per-pixel tranmission values and multiplies the wavefront amplitude by these values. Input arrays must be the same size as the wavefront. Values should be between 0 and 1, but this is not enforced.

??? info "TransmissiveOptic API"
    :::dLux.optics.TransmissiveOptic

---

## Add Phase

This layer takes in an array of per-pixel phase values in radians and adds it to the wavefront. Input arrays must be the same size as the wavefront.

??? info "AddPhase API"
    :::dLux.optics.AddPhase

---

## Add OPD

This layer takes in an array of per-pixel Optical Path Difference (OPD) values in meters and adds it to the wavefront. Input arrays must be the same size as the wavefront.

??? info "AddOPD API"
    :::dLux.optics.AddOPD

---

## Apply Basis OPD

This layer takes in a set of basis vectors and (optionally) coefficients, which is then used to calculate the total OPD and adds the resulting OPD to the wavefront. The basis vectors should have three dimensions and the coefficients one. This can be used to add a set of (typically) Zernike polynomials to the wavefront, although any set of basis vectors can be used.

??? info "ApplyBasisOPD API"
    :::dLux.optics.ApplyBasisOPD

---

## Rotate

This layer rotates the wavefront by a specified angle in radians.

??? info "Rotate API"
    :::dLux.optics.Rotate

---

## Apply Basis CLIMB

This layer takes a set of basis vectors and (optionally) coefficients, which is then used to calculate a binary version of the resulting output in a continuous manner using the ["CLIMB" algorithm](https://arxiv.org/abs/2107.00952).

??? info "ApplyBasisCLIMB API"
    :::dLux.optics.ApplyBasisCLIMB