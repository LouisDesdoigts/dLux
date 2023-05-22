from __future__ import annotations
import jax.numpy as np
import dLux.utils as dlu
from jax import Array, vmap
import matplotlib.pyplot as plt
import dLux


__all__ = ["ApplyBasisCLIMB"]


OpticalLayer = lambda : dLux.optical_layers.OpticalLayer
BasisLayer = lambda : dLux.optical_layers.BasisLayer


class ApplyBasisCLIMB(BasisLayer()):
    """
    Adds an array of binary phase values to the input wavefront from a set of
    continuous basis vectors. This uses the CLIMB algorithm in order to
    generate the binary values in a continous manner as described in the
    paper Wong et al. 2021. The basis vectors are taken as an Optical Path
    Difference (OPD), and applied to the phase of the wavefront. The ideal
    wavelength parameter described the wavelength that will have a perfect
    anti-phase relationship given by the Optical Path Difference.

    Note: Many of the methods in the class still need doccumentation.
    Note: This currently only outputs 256 pixel arrays and uses a 3x oversample,
    therefore requiring a 768 pixel basis array.

    Attributes
    ----------
    basis: Array
        Arrays holding the continous pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    ideal_wavelength : Array
        The target wavelength at which a perfect anti-phase relationship is
        applied via the OPD.
    """
    # basis            : Array
    # coefficients     : Array
    ideal_wavelength : Array


    def __init__(self             : OpticalLayer(),
                 basis            : Array,
                 ideal_wavelength : Array,
                 coefficients     : Array = None) -> OpticalLayer():
        """
        Constructor for the ApplyBasisCLIMB class.

        Parameters
        ----------
        basis : Array
            Arrays holding the continous pre-calculated basis vectors. This must
            be a 3d array of shape (nterms, npixels, npixels), with the final
            two dimensions matching that of the wavefront at time of
            application. This is currently required to be a nx768x768 shaped
            array. 
        ideal_wavelength : Array
            The target wavelength at which a perfect anti-phase relationship is
            applied via the OPD.
        coefficients : Array = None
            The Array of coefficients to be applied to each basis vector. This
            must be a one dimensional array with leading dimension equal to the
            leading dimension of the basis vectors. Default is None which
            initialises an array of zeros.
        """
        super().__init__(basis=basis, coefficients=coefficients)
        # self.basis            = np.asarray(basis, dtype=float)
        self.ideal_wavelength = np.asarray(ideal_wavelength, dtype=float)
        # self.coefficients     = np.array(coefficients).astype(float) \
        #             if coefficients is not None else np.zeros(len(self.basis))

        # # Inputs checks
        # assert self.basis.ndim == 3, \
        # ("basis must be a 3 dimensional array, ie (nterms, npixels, npixels).")
        # assert self.basis.shape[-1] == 768, \
        # ("Basis must have shape (n, 768, 768).")
        # assert self.coefficients.ndim == 1 and \
        # self.coefficients.shape[0] == self.basis.shape[0], \
        # ("coefficients must be a 1 dimensional array with length equal to the "
        # "First dimension of the basis array.")
        # assert self.ideal_wavelength.ndim == 0, ("ideal_wavelength must be a "
        #                                          "scalar array.")


    def __call__(self : OpticalLayer(), wavefront : Wavefront) -> Wavefront:
        """
        Generates and applies the binary OPD array to the wavefront in a
        differentiable manner.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the binary OPD applied.
        """
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent, ppsz=wavefront.npixels)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavelength)
        return wavefront.add_opd(opd)


    @property
    def applied_shape(self):
        return tuple(np.array(self.basis.shape[-2:])//3)


    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel


    def phase_to_opd(self, phase, wavel):
        return phase*wavel/(2*np.pi)


    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)


    def get_total_opd(self):
        return self.get_opd(self.basis, self.coefficients)


    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent)
        return binary_phase


    def lsq_params(self, img):
        xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),
                             np.linspace(0,1,img.shape[1]))
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
        return matrix, xx, yy, A


    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix,img.ravel())


    def area(self, img, epsilon = 1e-15):
        a,b,c = self.lsq(img)
        a = np.where(a==0,epsilon,a)
        b = np.where(b==0,epsilon,b)
        c = np.where(c==0,epsilon,c)
        x1 = (-b-c)/(a) # don't divide by zero
        x2 = -c/(a) # don't divide by zero
        x1, x2 = np.min(np.array([x1,x2])), np.max(np.array([x1,x2]))
        x1, x2 = np.max(np.array([x1,0])), np.min(np.array([x2,1]))

        dummy = x1 + (-c/b)*x2-(0.5*a/b)*x2**2 - (-c/b)*x1+(0.5*a/b)*x1**2

        # Set the regions where there is a defined gradient
        dummy = np.where(dummy>=0.5,dummy,1-dummy)

        # Colour in regions
        dummy = np.where(np.mean(img)>=0,dummy,1-dummy)

        # rescale between 0 and 1?
        dummy = np.where(np.all(img>0),1,dummy)
        dummy = np.where(np.all(img<=0),0,dummy)

        # undecided region
        dummy = np.where(np.any(img==0),np.mean(dummy>0),dummy)

        # rescale between 0 and 1
        dummy = np.clip(dummy, 0, 1)

        return dummy

    def CLIMB(self, wf, ppsz = 256):
        psz = ppsz * 3
        dummy = np.array(np.split(wf, ppsz))
        dummy = np.array(np.split(np.array(dummy), ppsz, axis = 2))
        subarray = dummy[:,:,0,0]

        flat = dummy.reshape(-1, 3, 3)
        vmap_mask = vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin