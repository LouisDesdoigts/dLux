class Filter(Base):
    """
    NOTE: This class is under development.

    A class for modelling optical filters.

    Attributes
    ----------
    wavelengths : Array
        The wavelengths at which the filter is defined.
    throughput : Array
        The throughput of the filter at the corresponding wavelength.
    filter_name : str
        A string identifier that can be used to initialise specific filters.
    """

    wavelengths: Array
    throughput: Array
    filter_name: str

    def __init__(
        self: Filter,
        wavelengths: Array = None,
        throughput: Array = None,
        filter_name: str = None,
    ) -> Filter:
        """
        Constructor for the Filter class. All inputs are optional and defaults
        to uniform unitary throughput. If filter_name is specified then
        wavelengths and weights must not be specified.

        Parameters
        ----------
        wavelengths : Array = None
            The wavelengths at which the filter is defined.
        throughput : Array = None
            The throughput of the filter at the corresponding wavelength.
        filter_name : str = None
            A string identifier that can be used to initialise specific
            filters. Currently no pre-built filters are implemented.
        """
        # Take the filter name as the priority input
        if filter_name is not None:
            # TODO: Pre load filters
            raise NotImplementedError("You know what this means.")
            pass

            # Check that wavelengths and throughput are not specified
            if wavelengths is not None or throughput is not None:
                raise ValueError(
                    "If filter_name is specified, wavelengths "
                    "and throughput can not be specified."
                )

        # Check that both wavelengths and throughput are specified
        elif (wavelengths is None and throughput is not None) or (
            wavelengths is not None and throughput is None
        ):
            raise ValueError(
                "If either wavelengths or throughput is "
                "specified, then both must be specified."
            )

        # Neither is specified
        elif wavelengths is None and throughput is None:
            self.wavelengths = np.array([0.0, np.inf])
            self.throughput = np.array([1.0, 1.0])
            self.filter_name = "Unitary"

        # Both wavelengths and throughputs are specified
        else:
            self.wavelengths = np.asarray(wavelengths, dtype=float)
            self.throughput = np.asarray(throughput, dtype=float)
            self.filter_name = "Custom"

            # Check bounds
            assert (
                self.wavelengths.ndim == 1 and self.throughput.ndim == 1
            ), "Both wavelengths and throughput must be 1 dimensional."
            assert (
                self.wavelengths.shape == self.throughput.shape
            ), "wavelengths and throughput must have the same length."
            assert (
                np.min(self.wavelengths) >= 0
            ), "wavelengths can not be less than 0."
            assert (self.throughput >= 0).all() and (
                self.throughput <= 1
            ).all(), "throughput must be between 0-1."
            assert np.min(wavelengths) < np.max(
                wavelengths
            ), "wavelengths must be in-order from small to large."

    def get_throughput(self: Filter, sample_wavelengths: Array) -> Array:
        """
        Gets the average throughput of the bandpass defined the differences
        between each sample wavelength, i.e. if sample wavelengths are:
            [10, 20, 30, 40],
        the bandpasses for each sample wavelength will be
            [5-15, 15-25, 25-30, 35-40].
        The throughput is calculated as the average throughput over that
        bandpass.

        Parameters
        ----------
        sample_wavelengths : Array, metres
            The wavelengths at which to sample the filter. Must contain at
            least two values.

        Returns
        -------
        throughputs : Array
            The average throughput for each bandpass defined by
            sample_wavelengths.
        """
        mids = (sample_wavelengths[1:] + sample_wavelengths[:-1]) / 2
        diffs = np.diff(sample_wavelengths)

        start = np.array([sample_wavelengths[0] - diffs[0] / 2])
        end = np.array([sample_wavelengths[-1] + diffs[-1] / 2])
        # min_val = np.array([self.wavelengths.min()])
        # max_val = np.array([self.wavelengths.max()])
        bounds = np.concatenate([start, mids, end])

        # Translate input wavelengths to indexes
        min_wavelength = self.wavelengths.min()
        max_wavelength = self.wavelengths.max()
        num_wavelength = len(self.wavelengths)
        wavelength_range = max_wavelength - min_wavelength
        bnd_indxs = (
            num_wavelength * (bounds - min_wavelength) / wavelength_range
        )
        bnd_indxs = np.clip(bnd_indxs, a_min=0, a_max=len(self.wavelengths))
        bnd_inds = np.round(bnd_indxs, decimals=0).astype(int)

        def nan_div(y, x):
            x_new = np.where(x == 0, 1, x)
            return np.where(x == 0, 0.0, y / x_new)

        def get_tp(start, end, weights, indexes):
            size = end - start
            val = np.where(
                (indexes <= start) | (indexes >= end), 0.0, weights
            ).sum()
            return nan_div(val, size)

        starts = bnd_inds[:-1]
        ends = bnd_inds[1:]
        # dwavelength = self.wavelengths[1] - self.wavelengths[0]
        indexes = np.arange(len(self.wavelengths))

        # weights = self.throughput/self.throughput.sum()
        weights = self.throughput
        out = vmap(get_tp, in_axes=(0, 0, None, None))(
            starts, ends, weights, indexes
        )
        return out

    def model(self: Filter, optics: Optics, **kwargs):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector. Users must provide optics and some form
        of source, either via a scene, sources or single source input, but not
        multiple.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector = None
            The detector to use with the observation. Defaults to the
            internally stored value.
        scene : Scene = None
            The scene to observe. Defaults to the internally stored value.
        sources : Union[dict, list, tuple) = None
            The sources to observe.
        source : dLux.sources.Source = None
            The source to observe.
        normalise_sources : bool = True
            Whether to normalise the sources before modelling. Default is True.
        flatten : bool = False
            Whether the output image should be flattened. Default is False.
        return_tree : bool = False
            Whether to return a Pytree like object with matching tree structure
            as the input scene/sources/source. Default is False.

        Returns
        -------
        image : Array, Pytree
            The image of the scene modelled through the optics with detector
            and filter effects applied if they are supplied. Returns either as
            a single array (if return_tree is false), or a pytree like object
            with matching tree structure as the input scene/sources/source.
        """
        return model(optics, filter=self, **kwargs)
