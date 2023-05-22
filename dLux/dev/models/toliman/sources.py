


class MixedAlphaCen(AlphaCen):
    mixing         : float
    

    def __init__(self, 
        nwavels = 101,
        x_position = 0., # arcseconds
        y_position = 0., # arcseconds
        separation = 10., # arcseconds
        position_angle = 90, # Degrees
        weights = None,
        mixing = 0.05,
        log_flux = 5, # Photons TODO: Find true flux
        A_mag = 1.33,
        B_mag = 0.01,
        ):
        """
        
        """
        self.mixing = mixing
        weights, wavelengths = get_mixed_alpha_cen_spectra(nwavels)
        super().__init__(nwavels, x_position, y_position, separation, 
            position_angle, weights, log_flux, A_mag, B_mag)

        def plot(self):
            """
            
            """
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.title(f"Alpha Cen A, f: {f}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Weight")
            plt.plot(spec_wavelengths*1e9, MixedA, label='Mixed')
            plt.plot(spec_wavelengths*1e9, SpecA, label='Main')
            plt.plot(spec_wavelengths*1e9, SpottedA, label='Spotted')
            plt.legend()

            SpecB = (1-f)*Bspec
            SpottedB = f*Sspec
            MixedB = SpecB + SpottedB
            plt.subplot(1, 2, 2)
            plt.title("Alpha Cen B, f: {f}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Weight")
            plt.plot(spec_wavelengths*1e9, MixedB, label='Mixed')
            plt.plot(spec_wavelengths*1e9, SpecB, label='Main')
            plt.plot(spec_wavelengths*1e9, SpottedB, label='Spotted')
            plt.legend()
            plt.show()


    @property
    def norm_weights(self):
        """
        
        """
        # Get Spectra
        Spotted = self.mixing * self.weights[2]
        MixedA = (1 - self.mixing) * self.weights[0] + Spotted
        MixedB = (1 - self.mixing) * self.weights[1] + Spotted
        weights = np.array([MixedA, MixedB])

        # Normalise
        return weights / weights.sum(1)[:, None]


    def model(self      : Source,
              optics    : Optics) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Get Values
        weights = self.norm_weights
        fluxes = self.raw_fluxes
        positions = self.xy_positions

        # vmap propagator
        source_propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        propagator = vmap(source_propagator, in_axes=(None, 0))

        # Model PSF
        input_weights = weights * fluxes[:, None]
        psfs = propagator(self.wavelengths, positions)
        psfs *= input_weights[..., None, None]
        return psfs.sum((0, 1))